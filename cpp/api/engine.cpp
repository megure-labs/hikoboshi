#include <hikoboshi/api/engine.hpp>

#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>
#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hikoboshi::api {

struct EngineThreadingState {
  std::mutex mutex;
  std::unique_ptr<universal::detail::ThreadPool> pool;
  std::size_t pool_thread_count = 0;
  std::vector<algorithms::detail::AllVsAllWorkerWorkspace>
      pair_workspaces;
};

namespace {

constexpr std::size_t kMpnn64HiddenDimension = 64;
constexpr std::size_t kMpnn64NeighborCount = 64;
constexpr std::size_t kMpnn64RbfCount = 16;
constexpr std::size_t kMpnn64LayerCount = 3;
constexpr std::size_t kParallelPairThreshold = 45;
constexpr std::size_t kPhase1MemoryBudgetDivisor = 2;
constexpr std::size_t kPhase1MeminfoFallbackThreadCap = 2;
constexpr float kMpnn64MessageScale = 30.0F;

using PreparedMpnn64WeightsPtr =
    decltype(algorithms::PairwiseStructureRequest{}.weights);

constexpr std::string_view kPackageSchemaVersion = "0.1.0";
constexpr std::string_view kPackageId = "hikoboshi-mpnn-d64";
constexpr std::string_view kPackageArchitectureId = "hikoboshi_mpnn_v1";
constexpr std::string_view kPackageGapFamily = "Hikoboshi 0.1.0 hard-SW";
constexpr std::string_view kPackageSoftGapFamily = "Hikoboshi 0.1.0 soft-SW";
constexpr float kMpnn64SoftGapOpen = -3.21337F;
constexpr float kMpnn64SoftGapExtension = -0.111704F;

// ESM2-8M architecture identifiers. These mirror the dispatch registry record
// the `architecture_design`/`forward`/`weights-package` cascade installed;
// the api layer cannot include `hikoboshi/dispatch/registry/` per the dep
// graph (`api -> algorithms, universal`), so the registry-derived constants
// are mirrored here. Any drift between this list and the registry is caught
// by `hikoboshi_dispatch_registry_smoke` plus the
// `hikoboshi_api_engine_smoke` tests.
constexpr std::string_view kEsm2_8mPackageId = "hikoboshi-esm2-8m";
constexpr std::string_view kEsm2_8mPackageArchitectureId = "hikoboshi_esm2_v1";

// ESM2-8M descriptor constants. The descriptor is otherwise the same
// stock-FAIR ESM2-8M shape; vocab is Casey's compacted 29-row checkpoint.
constexpr std::size_t kEsm2_8mVocabSize = 29;
constexpr std::size_t kEsm2_8mHiddenDimension = 320;
constexpr std::size_t kEsm2_8mLayerCount = 6;
constexpr std::size_t kEsm2_8mHeadCount = 20;
constexpr std::size_t kEsm2_8mHeadDim = 16;
constexpr std::size_t kEsm2_8mFfnHiddenDimension = 1280;
// ESM2 uses rotary position embeddings, so there is no architectural length
// ceiling; 1024 was only the original ESM2 pretraining crop length, not a limit
// of this checkpoint. The encoder workspace allocates per actual seq_len (see
// prepare_esm2_workspace), so this value is purely a sanity guard. Set well above
// any real protein (titin ~34k) to act as no practical limit while still
// bounding the O(L^2) attention buffer against pathological inputs.
constexpr std::size_t kEsm2_8mMaxSequenceLength = 65536;

// Compacted-vocab special token ids; mirror `kEsm2TokenTable` in
// `cpp/weights/embedded_esm2_8m.cpp`. The ESM2 training pipeline (and the
// PyTorch reference under `scripts/esm2_pytorch_goldens.py`) wraps each
// AA sequence as `[<cls>, aa..., <eos>]` before the encoder forward pass;
// Hikoboshi 0.1.0's production sequence-input route mirrors that wrapping
// inside the engine so adapters can keep passing raw AA token spans.
// fe2 traced the bl5 quality regression to the missing wrap; fe3 lands
// it as a per-route engine transform that strips the resulting CLS/EOS
// embedding rows before downstream similarity / pairwise sees the data.
constexpr std::int32_t kEsm2_8mClsTokenId = 26;
constexpr std::int32_t kEsm2_8mEosTokenId = 27;
constexpr std::size_t kEsm2_8mSpecialTokenOverhead = 2U;

enum class PackageArchitectureFamily {
  Mpnn64,
  Esm2_8m,
};

// Resolve a registered architecture id into the family tag the engine
// dispatches on. Returns `false` when the id is not one of the live
// architectures.
bool resolve_architecture_family(std::string_view architecture_id,
                                 PackageArchitectureFamily& family) noexcept {
  if (architecture_id == kPackageArchitectureId) {
    family = PackageArchitectureFamily::Mpnn64;
    return true;
  }
  if (architecture_id == kEsm2_8mPackageArchitectureId) {
    family = PackageArchitectureFamily::Esm2_8m;
    return true;
  }
  return false;
}

bool is_sequence_route_family(PackageArchitectureFamily family) noexcept {
  return family == PackageArchitectureFamily::Esm2_8m;
}

template <typename T>
bool span_contains(universal::Span<const T> span, T value) noexcept {
  return std::find(span.begin(), span.end(), value) != span.end();
}

bool weights_configured(const universal::WeightsHandle& weights) noexcept {
  return weights.opaque != nullptr || weights.view != nullptr;
}

// See also: docs/architecture/CPP_TREE_OVERVIEW.md#reserved-axes. Backend
// enum names are intentionally broader than the 0.1.0 selectable set; Engine
// remains the API-side gate that accepts only Auto and Scalar here.
universal::Status validate_backend(universal::Backend backend) noexcept {
  switch (backend) {
    case universal::Backend::Auto:
    case universal::Backend::Scalar:
      return universal::ok_status();
  }
  return universal::invalid_argument_status("backend is not implemented");
}

bool scalar_planner_policy(
    const universal::PlannerPolicy& policy) noexcept {
  return policy.policy_version == universal::kScalarPlannerPolicyVersion &&
         policy.kind == universal::PlannerPolicyKind::ScalarOnly &&
         policy.default_backend == universal::Backend::Scalar &&
         policy.fusion_rule == universal::PlannerFusionRule::Disabled &&
         policy.reserved_flags == 0U &&
         policy.fallback_order.size == 1U &&
         policy.fallback_order.data != nullptr &&
         policy.fallback_order.data[0] == universal::Backend::Scalar;
}

universal::Status validate_planner_policy(
    const universal::PlannerPolicy* policy) noexcept {
  const universal::PlannerPolicy& resolved =
      universal::scalar_default_planner_policy(policy);
  if (!scalar_planner_policy(resolved)) {
    return universal::invalid_argument_status(
        "planner policy is reserved scalar-only in Hikoboshi 0.1.0");
  }
  return universal::ok_status();
}

universal::Status validate_engine_axes(
    const EngineConfig& config) noexcept {
  const universal::Status backend_status =
      validate_backend(config.execution.backend);
  if (!universal::is_ok(backend_status)) {
    return backend_status;
  }
  return validate_planner_policy(config.planner_policy);
}

// See also: docs/architecture/ROADMAP.md#reserved-but-not-01-surface. These
// predicates keep future package axes visible in descriptors while rejecting
// them before API orchestration reaches algorithms.
bool supported_input_route(universal::PackageInputKind route,
                           PackageArchitectureFamily family) noexcept {
  switch (route) {
    case universal::PackageInputKind::StructureBackboneAtoms:
    case universal::PackageInputKind::CoordsBackbone:
      return family == PackageArchitectureFamily::Mpnn64;
    case universal::PackageInputKind::ResidueEmbeddings:
      return true;
    case universal::PackageInputKind::SequenceTokens:
      return family == PackageArchitectureFamily::Esm2_8m;
    case universal::PackageInputKind::StructureAllAtom:
    case universal::PackageInputKind::DirectScoreMatrix:
      return false;
  }
  return false;
}

bool supported_preprocessing_capability(
    universal::PackagePreprocessingCapability capability,
    PackageArchitectureFamily family) noexcept {
  switch (capability) {
    case universal::PackagePreprocessingCapability::AtomInference:
    case universal::PackagePreprocessingCapability::VirtualCb:
    case universal::PackagePreprocessingCapability::CaKnn:
    case universal::PackagePreprocessingCapability::AtomPairDistances:
    case universal::PackagePreprocessingCapability::RbfExpand:
    case universal::PackagePreprocessingCapability::PositionalEncoding:
      return family == PackageArchitectureFamily::Mpnn64;
    case universal::PackagePreprocessingCapability::Tokenization:
      return family == PackageArchitectureFamily::Esm2_8m;
  }
  return false;
}

bool supported_output_kind(universal::PackageOutputKind output) noexcept {
  switch (output) {
    case universal::PackageOutputKind::ResidueEmbeddings:
      return true;
    case universal::PackageOutputKind::SubstitutionScores:
    case universal::PackageOutputKind::DirectPairScores:
      return false;
  }
  return false;
}

bool supported_backend_requirement(
    universal::PackageBackendRequirement backend) noexcept {
  switch (backend) {
    case universal::PackageBackendRequirement::CpuScalar:
      return true;
    case universal::PackageBackendRequirement::GpuCuda:
    case universal::PackageBackendRequirement::GpuMetal:
    case universal::PackageBackendRequirement::GpuHip:
      return false;
  }
  return false;
}

bool supports_raw_dot_score_semantics(
    const universal::ScoreSemantics& semantics) noexcept {
  return semantics.method == universal::ScoreMethod::RawDotV1 &&
         semantics.dtype == universal::DataType::Float32 &&
         semantics.layout ==
             universal::ScoreMatrixLayout::RowMajorQueryByTarget &&
         semantics.higher_is_better &&
         semantics.local_affine_additive &&
         semantics.normalization == universal::ScoreNormalization::None &&
         semantics.scale_family == universal::ScoreScaleFamily::RawDot;
}

universal::Status validate_input_routes(
    universal::Span<const universal::PackageInputKind> routes,
    universal::PackageInputKind required_route,
    PackageArchitectureFamily family) noexcept {
  if (routes.size == 0U) {
    return universal::invalid_argument_status(
        "selected package does not declare public Hikoboshi 0.1 input routes");
  }
  for (std::size_t index = 0; index < routes.size; ++index) {
    if (!supported_input_route(routes.data[index], family)) {
      switch (routes.data[index]) {
        case universal::PackageInputKind::StructureAllAtom:
          return universal::invalid_argument_status(
              "unsupported package input route structure_all_atom: Hikoboshi 0.1.0 does not accept all-atom structure routes");
        case universal::PackageInputKind::DirectScoreMatrix:
          return universal::invalid_argument_status(
              "unsupported package input route direct_score_matrix: Hikoboshi 0.1.0 does not accept direct score-matrix routes");
        case universal::PackageInputKind::SequenceTokens:
          return universal::invalid_argument_status(
              "unsupported package input route sequence_tokens for selected architecture: sequence_tokens is only routed by hikoboshi_esm2_v1");
        case universal::PackageInputKind::StructureBackboneAtoms:
          return universal::invalid_argument_status(
              "unsupported package input route structure_backbone_atoms for selected architecture: structure_backbone_atoms is only routed by hikoboshi_mpnn_v1");
        case universal::PackageInputKind::CoordsBackbone:
          return universal::invalid_argument_status(
              "unsupported package input route coords_backbone for selected architecture: coords_backbone is only routed by hikoboshi_mpnn_v1");
        case universal::PackageInputKind::ResidueEmbeddings:
          return universal::invalid_argument_status(
              "unsupported package input route residue_embeddings for selected architecture");
      }
      return universal::invalid_argument_status(
          "unsupported package input route for selected architecture");
    }
  }
  if (!span_contains(routes, required_route)) {
    if (required_route == universal::PackageInputKind::CoordsBackbone) {
      return universal::invalid_argument_status(
          "selected package does not support coords_backbone input route");
    }
    if (required_route == universal::PackageInputKind::SequenceTokens) {
      return universal::invalid_argument_status(
          "selected package does not support sequence_tokens input route");
    }
    return universal::invalid_argument_status(
        "selected package does not support structure_backbone_atoms input route");
  }
  return universal::ok_status();
}

universal::Status validate_preprocessing_capabilities(
    universal::Span<const universal::PackagePreprocessingCapability>
        capabilities,
    PackageArchitectureFamily family) noexcept {
  for (std::size_t index = 0; index < capabilities.size; ++index) {
    if (!supported_preprocessing_capability(capabilities.data[index],
                                            family)) {
      return universal::invalid_argument_status(
          "unsupported package preprocessing capability for selected architecture: Hikoboshi 0.1.0 routes tokenization through hikoboshi_esm2_v1 only and atom-inference / RBF / KNN through hikoboshi_mpnn_v1 only");
    }
  }
  return universal::ok_status();
}

universal::Status validate_output_kinds(
    universal::Span<const universal::PackageOutputKind> outputs) noexcept {
  if (outputs.size == 0U) {
    return universal::invalid_argument_status(
        "selected package does not declare the required residue_embeddings output");
  }
  for (std::size_t index = 0; index < outputs.size; ++index) {
    if (!supported_output_kind(outputs.data[index])) {
      return universal::invalid_argument_status(
          "unsupported package output kind: substitution scores and direct pair scores are reserved and do not create public score-only APIs in Hikoboshi 0.1.0");
    }
  }
  if (!span_contains(outputs, universal::PackageOutputKind::ResidueEmbeddings)) {
    return universal::invalid_argument_status(
        "selected package does not provide residue_embeddings output");
  }
  return universal::ok_status();
}

universal::Status validate_backend_requirements(
    universal::Span<const universal::PackageBackendRequirement> backends) noexcept {
  if (backends.size == 0U) {
    return universal::invalid_argument_status(
        "selected package does not declare the required cpu.scalar backend");
  }
  for (std::size_t index = 0; index < backends.size; ++index) {
    if (!supported_backend_requirement(backends.data[index])) {
      return universal::invalid_argument_status(
          "unsupported package backend requirement: Hikoboshi 0.1.0 supports cpu.scalar only");
    }
  }
  if (!span_contains(backends, universal::PackageBackendRequirement::CpuScalar)) {
    return universal::invalid_argument_status(
        "selected package does not support the cpu.scalar backend");
  }
  return universal::ok_status();
}

universal::Status validate_package_descriptor_route(
    const universal::PackageDescriptor& descriptor,
    universal::PackageInputKind required_route,
    PackageArchitectureFamily& family_out) noexcept {
  // Package validation is mirrored here because adapters may pass package
  // handles directly into EngineConfig; API rejects incompatible descriptors
  // before constructing algorithm requests.
  if (descriptor.identity.package_kind == universal::PackageKind::GraphIr ||
      descriptor.execution.mode == universal::PackageExecutionMode::GraphIr) {
    return universal::invalid_argument_status(
        "unsupported package execution mode graph_ir: Hikoboshi 0.1.0 supports only registered_architecture packages");
  }
  if (descriptor.identity.package_kind ==
      universal::PackageKind::SubstitutionMatrix) {
    return universal::invalid_argument_status(
        "substitution-matrix packages are reserved and do not create public score-only workflows in Hikoboshi 0.1.0");
  }
  if (descriptor.identity.package_schema_version != kPackageSchemaVersion ||
      descriptor.identity.package_kind !=
          universal::PackageKind::RegisteredArchitecture) {
    return universal::invalid_argument_status(
        "unsupported package identity: Hikoboshi 0.1.0 supports registered_architecture packages only");
  }
  if (descriptor.identity.package_id != kPackageId &&
      descriptor.identity.package_id != kEsm2_8mPackageId) {
    return universal::invalid_argument_status(
        "unsupported package identity: Hikoboshi 0.1.0 ships hikoboshi-mpnn-d64 and hikoboshi-esm2-8m only");
  }
  if (descriptor.execution.mode !=
      universal::PackageExecutionMode::RegisteredArchitecture) {
    return universal::invalid_argument_status(
        "unsupported package execution mode: Hikoboshi 0.1.0 supports registered_architecture only");
  }
  if (!resolve_architecture_family(descriptor.execution.architecture_id,
                                   family_out)) {
    return universal::invalid_argument_status(
        "unsupported registered architecture: Hikoboshi 0.1.0 supports hikoboshi_mpnn_v1 and hikoboshi_esm2_v1 only");
  }
  // Architecture identity must agree with the family the package id maps
  // to, otherwise a manifest mismatch leaks past validation.
  const bool mpnn_match =
      family_out == PackageArchitectureFamily::Mpnn64 &&
      descriptor.identity.package_id == kPackageId;
  const bool esm2_match =
      family_out == PackageArchitectureFamily::Esm2_8m &&
      descriptor.identity.package_id == kEsm2_8mPackageId;
  if (!mpnn_match && !esm2_match) {
    return universal::invalid_argument_status(
        "package identity and architecture id refer to different Hikoboshi 0.1.0 architectures");
  }

  universal::Status status = validate_input_routes(
      descriptor.inputs.routes, required_route, family_out);
  if (!universal::is_ok(status)) {
    return status;
  }
  status = validate_input_routes(descriptor.capabilities.input_routes,
                                 required_route, family_out);
  if (!universal::is_ok(status)) {
    return status;
  }
  status = validate_preprocessing_capabilities(
      descriptor.capabilities.preprocessing, family_out);
  if (!universal::is_ok(status)) {
    return status;
  }
  status = validate_output_kinds(descriptor.outputs.kinds);
  if (!universal::is_ok(status)) {
    return status;
  }
  status = validate_output_kinds(descriptor.capabilities.output_kinds);
  if (!universal::is_ok(status)) {
    return status;
  }

  if (descriptor.scoring.method == universal::ScoreMethod::CosineV1 ||
      descriptor.scoring.semantics.method == universal::ScoreMethod::CosineV1) {
    return universal::invalid_argument_status(
        "unsupported package scoring method cosine_v1: Hikoboshi 0.1.0 supports raw_dot_v1 scoring only");
  }
  if (descriptor.scoring.method != universal::ScoreMethod::RawDotV1 ||
      descriptor.scoring.output != universal::ScoreOutputKind::ScoreMatrix ||
      descriptor.scoring.inputs.size == 0U) {
    return universal::invalid_argument_status(
        "unsupported package scoring method: Hikoboshi 0.1.0 supports raw_dot_v1 residue-embedding scoring only");
  }
  for (std::size_t index = 0; index < descriptor.scoring.inputs.size; ++index) {
    if (descriptor.scoring.inputs.data[index] !=
        universal::ScoreInputKind::ResidueEmbeddings) {
      return universal::invalid_argument_status(
          "unsupported package scoring input: Hikoboshi 0.1.0 scores residue_embeddings only");
    }
  }
  if (!supports_raw_dot_score_semantics(descriptor.scoring.semantics)) {
    return universal::invalid_argument_status(
        "unsupported package score semantics: Hikoboshi 0.1.0 supports unnormalized raw-dot additive float32 score matrices only");
  }
  // Gap defaults: `gaps` remains the release-default hard-SW family and
  // `soft_gaps` is the benchmark/research companion family. Per-package values
  // diverge intentionally; provider-side validation pins the exact manifest
  // constants while the API gate enforces the shape and hard/soft split.
  if (descriptor.gaps.family != kPackageGapFamily ||
      descriptor.gaps.model != universal::GapModel::Affine ||
      descriptor.gaps.convention !=
          universal::GapConvention::GapOpenPlusKMinusOneGapExtension ||
      descriptor.gaps.calibrated_for_score_method !=
          universal::ScoreMethod::RawDotV1 ||
      descriptor.soft_gaps.family != kPackageSoftGapFamily ||
      descriptor.soft_gaps.model != universal::GapModel::Affine ||
      descriptor.soft_gaps.convention !=
          universal::GapConvention::GapOpenPlusKMinusOneGapExtension ||
      descriptor.soft_gaps.calibrated_for_score_method !=
          universal::ScoreMethod::RawDotV1) {
    return universal::invalid_argument_status(
        "unsupported package gap defaults: Hikoboshi 0.1.0 requires hard-SW and soft-SW affine raw-dot calibrated gap models");
  }
  if (family_out == PackageArchitectureFamily::Mpnn64) {
    if (descriptor.gaps.gap_open != kDefaultGapOpen ||
        descriptor.gaps.gap_extension != kDefaultGapExtension ||
        descriptor.soft_gaps.gap_open != kMpnn64SoftGapOpen ||
        descriptor.soft_gaps.gap_extension != kMpnn64SoftGapExtension) {
      return universal::invalid_argument_status(
          "unsupported hikoboshi-mpnn-d64 gap defaults: Hikoboshi 0.1.0 requires hard gaps -1.40000/-0.150000 and soft gaps -3.21337/-0.111704");
    }
  } else {
    // ESM2-8M's annealed hard-SW calibration has a positive extension value;
    // specific magnitudes are pinned by `validate_esm2_8m_package`.
    if (!std::isfinite(descriptor.gaps.gap_open) ||
        !std::isfinite(descriptor.gaps.gap_extension) ||
        !std::isfinite(descriptor.soft_gaps.gap_open) ||
        !std::isfinite(descriptor.soft_gaps.gap_extension) ||
        descriptor.gaps.gap_open >= 0.0F ||
        descriptor.soft_gaps.gap_open >= 0.0F ||
        descriptor.soft_gaps.gap_extension >= 0.0F) {
      return universal::invalid_argument_status(
          "unsupported hikoboshi-esm2-8m gap defaults: Hikoboshi 0.1.0 requires finite affine values, negative gap openings, and a negative soft-SW extension");
    }
  }
  if (descriptor.alignment.algorithm !=
      universal::AlignmentAlgorithmId::HardLocalAffineSwV1) {
    return universal::invalid_argument_status(
        "unsupported package alignment algorithm: Hikoboshi 0.1.0 supports hard_local_affine_sw_v1 only");
  }

  status = validate_backend_requirements(
      descriptor.execution.backend_requirements);
  if (!universal::is_ok(status)) {
    return status;
  }
  return validate_backend_requirements(descriptor.capabilities.backends);
}

universal::Status validate_package_descriptor_route(
    const universal::PackageDescriptor& descriptor,
    universal::PackageInputKind required_route) noexcept {
  PackageArchitectureFamily family = PackageArchitectureFamily::Mpnn64;
  return validate_package_descriptor_route(descriptor, required_route, family);
}

bool alignment_mode_runs_hard(AlignmentMode mode) noexcept {
  return mode == AlignmentMode::Hard || mode == AlignmentMode::Both;
}

bool alignment_mode_runs_soft(AlignmentMode mode) noexcept {
  return mode == AlignmentMode::Soft || mode == AlignmentMode::Both;
}

struct ResolvedAlignmentOptions {
  AlignmentOptions hard;
  AlignmentOptions soft;
};

AlignmentOptions resolve_alignment_against_gaps(
    AlignmentOptions options,
    const universal::AffineGapDefaults& gaps) noexcept {
  if (is_package_default_gap(options.gap_open)) {
    options.gap_open = gaps.gap_open;
  }
  if (is_package_default_gap(options.gap_extension)) {
    options.gap_extension = gaps.gap_extension;
  }
  return options;
}

ResolvedAlignmentOptions resolve_alignment_against_package(
    AlignmentOptions options,
    const universal::PackageDescriptor& descriptor) noexcept {
  return {resolve_alignment_against_gaps(options, descriptor.gaps),
          resolve_alignment_against_gaps(options, descriptor.soft_gaps)};
}

ResolvedAlignmentOptions resolve_alignment_against_mpnn64(
    AlignmentOptions options) noexcept {
  const universal::AffineGapDefaults hard_gaps{
      kPackageGapFamily,
      universal::GapModel::Affine,
      kDefaultGapOpen,
      kDefaultGapExtension,
      universal::GapConvention::GapOpenPlusKMinusOneGapExtension,
      universal::ScoreMethod::RawDotV1,
  };
  const universal::AffineGapDefaults soft_gaps{
      kPackageSoftGapFamily,
      universal::GapModel::Affine,
      kMpnn64SoftGapOpen,
      kMpnn64SoftGapExtension,
      universal::GapConvention::GapOpenPlusKMinusOneGapExtension,
      universal::ScoreMethod::RawDotV1,
  };
  return {resolve_alignment_against_gaps(options, hard_gaps),
          resolve_alignment_against_gaps(options, soft_gaps)};
}

algorithms::PairwiseOptions to_algorithms_options(
    const ResolvedAlignmentOptions& options) noexcept {
  algorithms::PairwiseOptions algorithm_options{};
  algorithm_options.gap_open = options.hard.gap_open;
  algorithm_options.gap_extension = options.hard.gap_extension;
  algorithm_options.soft_gap_open = options.soft.gap_open;
  algorithm_options.soft_gap_extension = options.soft.gap_extension;
  return algorithm_options;
}

algorithms::AllVsAllOptions to_algorithms_options(
    const AllVsAllOptions& options,
    const ResolvedAlignmentOptions& resolved_alignment) noexcept {
  algorithms::AllVsAllOptions algorithm_options{};
  algorithm_options.include_self = options.include_self;
  algorithm_options.pairwise = to_algorithms_options(resolved_alignment);
  algorithm_options.hard_mode = alignment_mode_runs_hard(options.mode);
  algorithm_options.soft_mode = alignment_mode_runs_soft(options.mode);
  algorithm_options.temperature = options.temperature;
  return algorithm_options;
}

universal::StructureView structure_from_coords(const CoordsInputView& coords) noexcept {
  return {coords.residue_count,
          coords.coordinates,
          coords.atom_sources,
          coords.residue_codes,
          coords.residues,
          {},
          {},
          {}};
}

std::size_t structure_coordinate_count(std::size_t residue_count) noexcept {
  return residue_count * universal::kCanonicalAtomCount *
         universal::kCoordinateAxisCount;
}

std::size_t structure_atom_source_count(std::size_t residue_count) noexcept {
  return residue_count * universal::kCanonicalAtomCount;
}

universal::Status validate_structure_view(
    const universal::StructureView& structure) noexcept {
  if (structure.residue_count == 0) {
    return universal::invalid_argument_status("structure input must contain at least one residue");
  }
  if (structure.coordinates.data == nullptr ||
      structure.coordinates.size <
          structure_coordinate_count(structure.residue_count)) {
    return universal::invalid_argument_status("structure coordinates are invalid");
  }
  if (structure.atom_sources.data == nullptr ||
      structure.atom_sources.size <
          structure_atom_source_count(structure.residue_count)) {
    return universal::invalid_argument_status("structure atom sources are invalid");
  }
  return universal::ok_status();
}

universal::Status validate_embedding_view(
    const universal::EmbeddingView& embedding) noexcept {
  if (embedding.residue_count == 0) {
    return universal::invalid_argument_status("embedding input must contain at least one residue");
  }
  if (embedding.dimension == 0) {
    return universal::invalid_argument_status("embedding dimension must be non-zero");
  }
  if (embedding.values.data == nullptr ||
      embedding.values.size < embedding.residue_count * embedding.dimension) {
    return universal::invalid_argument_status("embedding values are invalid");
  }
  return universal::ok_status();
}

bool opaque_points_at_tensor_storage(const universal::WeightsHandle& weights) noexcept {
  if (weights.opaque == nullptr || weights.view == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < weights.view->tensors.size; ++index) {
    if (weights.opaque == weights.view->tensors.data[index].data) {
      return true;
    }
  }
  return false;
}

universal::Status validate_weights_handle(
    const universal::WeightsHandle& weights) noexcept {
  if (weights.opaque == nullptr || weights.view == nullptr) {
    return universal::failed_precondition_status("structure encoding requires MPNN weights");
  }
  if (weights.view->metadata.hidden_dimension != kMpnn64HiddenDimension) {
    return universal::invalid_argument_status("MPNN weights must have hidden dimension 64");
  }
  if (opaque_points_at_tensor_storage(weights)) {
    return universal::failed_precondition_status("MPNN weights are not prepared for API encoding");
  }
  return universal::ok_status();
}

universal::Result<universal::WeightsHandle> resolve_structure_package_weights(
    const EngineConfig& config,
    universal::PackageInputKind required_route) noexcept {
  if (config.package.descriptor == nullptr) {
    if (config.package.opaque != nullptr) {
      return {universal::failed_precondition_status("selected package descriptor is missing"),
              {nullptr, nullptr}};
    }
    // See also: docs/architecture/CPP_TREE_OVERVIEW.md#api-and-package-boundary.
    // Registry defaults are adapter-owned. API receives either the selected
    // package handle or its compatibility weights view through EngineConfig.
    const universal::Status status = validate_weights_handle(config.weights);
    return {status, universal::is_ok(status) ? config.weights
                                      : universal::WeightsHandle{nullptr,
                                                                 nullptr}};
  }

  const universal::PackageDescriptor& descriptor = *config.package.descriptor;
  universal::Status status =
      validate_package_descriptor_route(descriptor, required_route);
  if (!universal::is_ok(status)) {
    return {status, {nullptr, nullptr}};
  }

  universal::WeightsHandle resolved_weights =
      descriptor.compatibility_views.weights;
  if (weights_configured(config.weights)) {
    // A package handle and a compatibility WeightsHandle must be two views of
    // the same prepared package state, not two independent model selections.
    if (config.weights.opaque != resolved_weights.opaque ||
        config.weights.view != resolved_weights.view) {
      return {universal::invalid_argument_status(
                  "EngineConfig package and weights handle refer to different prepared package state"),
              {nullptr, nullptr}};
    }
    resolved_weights = config.weights;
  }
  if (config.package.opaque != nullptr &&
      config.package.opaque != resolved_weights.opaque) {
    return {universal::invalid_argument_status(
                "selected package handle does not match its compatibility weights"),
            {nullptr, nullptr}};
  }

  status = validate_weights_handle(resolved_weights);
  return {status, universal::is_ok(status) ? resolved_weights
                                    : universal::WeightsHandle{nullptr,
                                                               nullptr}};
}

PreparedMpnn64WeightsPtr prepared_weights(
    const universal::WeightsHandle& weights) noexcept {
  return static_cast<PreparedMpnn64WeightsPtr>(weights.opaque);
}

decltype(algorithms::PairwiseStructureRequest{}.descriptor) mpnn64_descriptor()
    noexcept {
  auto descriptor = algorithms::PairwiseStructureRequest{}.descriptor;
  descriptor.hidden_dimension = kMpnn64HiddenDimension;
  descriptor.neighbor_count = kMpnn64NeighborCount;
  descriptor.rbf_count = kMpnn64RbfCount;
  descriptor.layer_count = kMpnn64LayerCount;
  descriptor.message_scale = kMpnn64MessageScale;
  return descriptor;
}

hikoboshi::modules::Esm2Descriptor esm2_8m_descriptor() noexcept {
  hikoboshi::modules::Esm2Descriptor descriptor{};
  descriptor.vocab_size = kEsm2_8mVocabSize;
  descriptor.hidden_dimension = kEsm2_8mHiddenDimension;
  descriptor.layer_count = kEsm2_8mLayerCount;
  descriptor.head_count = kEsm2_8mHeadCount;
  descriptor.head_dim = kEsm2_8mHeadDim;
  descriptor.ffn_hidden_dimension = kEsm2_8mFfnHiddenDimension;
  descriptor.max_sequence_length = kEsm2_8mMaxSequenceLength;
  return descriptor;
}

// Wrap a raw AA token span as `[<cls>, aa..., <eos>]` for the ESM2-8M
// encoder forward pass. The PyTorch training pipeline and reference
// generators (`scripts/esm2_pytorch_goldens.py`,
// `scripts/softsw_pytorch_goldens.py`) feed the encoder this exact
// wrapping; without it the encoder's positional/embedding tables index
// at the wrong residue offsets and the resulting embeddings drift from
// the PyTorch reference by ~1e-1 .. 4.7e-1 per cell (fe2 measured).
std::vector<std::int32_t> wrap_esm2_sequence_tokens(
    universal::Span<const std::int32_t> raw_tokens) {
  std::vector<std::int32_t> wrapped;
  wrapped.reserve(raw_tokens.size + kEsm2_8mSpecialTokenOverhead);
  wrapped.push_back(kEsm2_8mClsTokenId);
  if (raw_tokens.data != nullptr && raw_tokens.size != 0) {
    wrapped.insert(wrapped.end(), raw_tokens.data,
                   raw_tokens.data + raw_tokens.size);
  }
  wrapped.push_back(kEsm2_8mEosTokenId);
  return wrapped;
}

// Resolve the borrowed WeightsView the ESM2 algorithms-layer needs from a
// caller-supplied EngineConfig. Returns a NotFound-shaped status if the
// package is not an ESM2 family package; returns the ESM2 view's
// universal pointer otherwise.
universal::Result<const universal::WeightsView*>
resolve_sequence_package_weights_view(const EngineConfig& config) noexcept {
  if (config.package.descriptor == nullptr) {
    return {universal::failed_precondition_status(
                "sequence-input encode requires a selected ESM2 package; pass it through EngineConfig.package"),
            nullptr};
  }
  PackageArchitectureFamily family = PackageArchitectureFamily::Mpnn64;
  universal::Status status = validate_package_descriptor_route(
      *config.package.descriptor, universal::PackageInputKind::SequenceTokens,
      family);
  if (!universal::is_ok(status)) {
    return {status, nullptr};
  }
  if (!is_sequence_route_family(family)) {
    return {universal::invalid_argument_status(
                "selected package does not declare the sequence_tokens input route"),
            nullptr};
  }
  const universal::WeightsHandle& view_handle =
      config.package.descriptor->compatibility_views.weights;
  if (view_handle.view == nullptr) {
    return {universal::failed_precondition_status(
                "selected ESM2 package descriptor has no compatibility weights view"),
            nullptr};
  }
  return {universal::ok_status(), view_handle.view};
}

algorithms::detail::PairwiseWorkspacePlan embedding_plan(
    const universal::EmbeddingView& query,
    const universal::EmbeddingView& target,
    bool allocate_soft_sw) noexcept {
  algorithms::detail::PairwiseWorkspacePlan plan{};
  plan.max_query_length = query.residue_count;
  plan.max_target_length = target.residue_count;
  plan.embedding_dimension = query.dimension;
  plan.allocate_mpnn = false;
  plan.allocate_soft_sw = allocate_soft_sw;
  return plan;
}

algorithms::detail::PairwiseWorkspacePlan structure_plan(
    const universal::StructureView& query,
    const universal::StructureView& target,
    bool allocate_soft_sw) noexcept {
  algorithms::detail::PairwiseWorkspacePlan plan{};
  plan.max_query_length = query.residue_count;
  plan.max_target_length = target.residue_count;
  plan.embedding_dimension = kMpnn64HiddenDimension;
  plan.allocate_mpnn = true;
  plan.allocate_soft_sw = allocate_soft_sw;
  plan.mpnn_descriptor = mpnn64_descriptor();
  return plan;
}

struct DirectMpnn64Workspace {
  hikoboshi::modules::detail::Mpnn64Workspace workspace{};
  std::vector<float> ca_coordinates;
  std::vector<float> residue_features;
  std::vector<std::int32_t> neighbor_indices;
  std::vector<float> neighbor_squared_distances;
  std::vector<float> rbf_features;
  std::vector<float> residue_state;
  std::vector<float> gathered_state;
  std::vector<float> edge_state;
  std::vector<float> message_state;
  std::vector<float> projected_message_state;
  std::vector<float> residue_scratch;
  std::vector<float> ffn_hidden;
};

template <typename T>
void assign_direct_span(std::vector<T>& storage,
                        universal::Span<T>& span) noexcept {
  span = {storage.data(), storage.size()};
}

void prepare_direct_mpnn64_workspace(
    DirectMpnn64Workspace& owned,
    const hikoboshi::modules::detail::Mpnn64MemoryPlan& plan) {
  namespace pmd = hikoboshi::modules::detail;
  owned.workspace.plan = plan;
  owned.ca_coordinates.resize(pmd::mpnn64_ca_coordinate_count(plan));
  owned.residue_features.resize(pmd::mpnn64_residue_feature_count(plan));
  owned.neighbor_indices.resize(pmd::mpnn64_neighbor_slot_count(plan));
  owned.neighbor_squared_distances.resize(
      pmd::mpnn64_neighbor_slot_count(plan));
  owned.rbf_features.resize(pmd::mpnn64_neighbor_rbf_count(plan));
  owned.residue_state.resize(pmd::mpnn64_residue_hidden_count(plan));
  owned.gathered_state.resize(pmd::mpnn64_neighbor_hidden_count(plan));
  owned.edge_state.resize(pmd::mpnn64_neighbor_hidden_count(plan));
  owned.message_state.resize(pmd::mpnn64_neighbor_hidden_count(plan));
  owned.projected_message_state.resize(
      pmd::mpnn64_neighbor_hidden_count(plan));
  owned.residue_scratch.resize(pmd::mpnn64_residue_hidden_count(plan));
  owned.ffn_hidden.resize(pmd::mpnn64_ffn_hidden_count(plan));

  assign_direct_span(owned.ca_coordinates, owned.workspace.ca_coordinates);
  assign_direct_span(owned.residue_features, owned.workspace.residue_features);
  assign_direct_span(owned.neighbor_indices, owned.workspace.neighbor_indices);
  assign_direct_span(owned.neighbor_squared_distances,
                     owned.workspace.neighbor_squared_distances);
  assign_direct_span(owned.rbf_features, owned.workspace.rbf_features);
  assign_direct_span(owned.residue_state, owned.workspace.residue_state);
  assign_direct_span(owned.gathered_state, owned.workspace.gathered_state);
  assign_direct_span(owned.edge_state, owned.workspace.edge_state);
  assign_direct_span(owned.message_state, owned.workspace.message_state);
  assign_direct_span(owned.projected_message_state,
                     owned.workspace.projected_message_state);
  assign_direct_span(owned.residue_scratch, owned.workspace.residue_scratch);
  assign_direct_span(owned.ffn_hidden, owned.workspace.ffn_hidden);
}

hikoboshi::modules::detail::Mpnn64MemoryPlan direct_mpnn64_plan(
    const decltype(mpnn64_descriptor())& descriptor,
    std::size_t residue_count) noexcept {
  hikoboshi::modules::detail::Mpnn64MemoryPlan plan{};
  plan.max_residue_count = residue_count;
  plan.hidden_dimension = descriptor.hidden_dimension;
  plan.neighbor_count = descriptor.neighbor_count;
  plan.rbf_count = descriptor.rbf_count;
  plan.layer_count = descriptor.layer_count;
  return plan;
}

universal::Result<std::vector<float>> encode_mpnn64_structure_direct(
    const universal::StructureView& structure,
    const universal::WeightsHandle& weights) {
  const auto descriptor = mpnn64_descriptor();
  const std::size_t value_count =
      structure.residue_count * descriptor.hidden_dimension;
  std::vector<float> embeddings;
  DirectMpnn64Workspace workspace;
  try {
    embeddings.assign(value_count, 0.0F);
    prepare_direct_mpnn64_workspace(
        workspace, direct_mpnn64_plan(descriptor, structure.residue_count));
  } catch (const std::bad_alloc&) {
    return {universal::unavailable_status(
                "MPNN direct encode workspace allocation failed"),
            {}};
  }

  hikoboshi::modules::Mpnn64ForwardRequest request{};
  request.coordinates = structure.coordinates.data;
  request.atom_sources = structure.atom_sources.data;
  request.residue_count = structure.residue_count;
  request.descriptor = descriptor;
  request.weights = prepared_weights(weights);
  request.workspace = &workspace.workspace;

  hikoboshi::modules::Mpnn64ForwardOutput output{};
  output.embeddings = embeddings.data();
  output.residue_count = structure.residue_count;
  output.hidden_dimension = descriptor.hidden_dimension;

  const universal::Status status =
      hikoboshi::modules::mpnn64_forward_scalar(request, output);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<std::vector<float>>(status);
  }
  return {universal::ok_status(), std::move(embeddings)};
}

universal::Status validate_pairwise_alignment_mode(
    AlignmentMode mode, float temperature) noexcept {
  switch (mode) {
    case AlignmentMode::Hard:
      return universal::ok_status();
    case AlignmentMode::Soft:
    case AlignmentMode::Both:
      if (!std::isfinite(temperature) || !(temperature > 0.0F)) {
        return universal::invalid_argument_status(
            "pairwise soft/both mode temperature must be a positive finite float");
      }
      return universal::ok_status();
  }
  return universal::invalid_argument_status(
      "pairwise alignment mode is not implemented");
}

universal::Status validate_all_vs_all_alignment_mode(
    AlignmentMode mode, float temperature) noexcept {
  switch (mode) {
    case AlignmentMode::Hard:
      return universal::ok_status();
    case AlignmentMode::Soft:
    case AlignmentMode::Both:
      if (!std::isfinite(temperature) || !(temperature > 0.0F)) {
        return universal::invalid_argument_status(
            "all-vs-all soft/both mode temperature must be a positive finite float");
      }
      return universal::ok_status();
  }
  return universal::invalid_argument_status(
      "all-vs-all alignment mode is not implemented");
}

PairwiseMetrics to_api_metrics(const algorithms::MetricBlock& metrics) {
  PairwiseMetrics api_metrics{};
  api_metrics.raw_sw_score = metrics.raw_sw_score;
  api_metrics.soft_sw_score = metrics.soft_sw_score;
  api_metrics.coverage_query = metrics.coverage_query;
  api_metrics.coverage_target = metrics.coverage_target;
  api_metrics.coverage_mean = metrics.coverage_mean;
  api_metrics.identity = metrics.identity;
  api_metrics.rmsd = metrics.rmsd;
  api_metrics.tm_score_query = metrics.tm_score_query;
  api_metrics.tm_score_target = metrics.tm_score_target;
  api_metrics.lddt = metrics.lddt;
  api_metrics.lddt_byA = metrics.lddt_byA;
  api_metrics.lddt_byB = metrics.lddt_byB;
  api_metrics.lddt_aln = metrics.lddt_aln;
  api_metrics.coverage_byA = metrics.coverage_byA;
  api_metrics.coverage_byB = metrics.coverage_byB;
  api_metrics.ecs = metrics.ecs;
  return api_metrics;
}

void assign_api_path(const AlignmentPath& source,
                     AlignmentPath& destination) {
  destination.steps.resize(source.steps.size());
  std::copy(source.steps.begin(), source.steps.end(),
            destination.steps.begin());
  destination.aligned_pairs = source.aligned_pairs;
  destination.query_start = source.query_start;
  destination.query_end = source.query_end;
  destination.target_start = source.target_start;
  destination.target_end = source.target_end;
}

// Static warning messages for the Hikoboshi 0.1.0 gap-default families.
// `PackageWarning::message` is a borrowed `std::string_view`, so the storage
// must outlive each warning record; string literals satisfy that contract.
constexpr std::string_view kMpnn64HardGapOverrideMessage =
    "user-provided affine gap values override hikoboshi-mpnn-d64 hard-SW calibrated defaults gap_open=-1.40000 and gap_extension=-0.150000";
constexpr std::string_view kMpnn64SoftGapOverrideMessage =
    "user-provided affine gap values override hikoboshi-mpnn-d64 soft-SW calibrated defaults gap_open=-3.21337 and gap_extension=-0.111704";
constexpr std::string_view kMpnn64BothGapOverrideMessage =
    "user-provided affine gap values override hikoboshi-mpnn-d64 hard-SW calibrated defaults (gap_open=-1.40000, gap_extension=-0.150000) and soft-SW calibrated defaults (gap_open=-3.21337, gap_extension=-0.111704)";
constexpr std::string_view kEsm2_8mHardGapOverrideMessage =
    "user-provided affine gap values override hikoboshi-esm2-8m hard-SW calibrated defaults gap_open=-1.01982 and gap_extension=+0.225736";
constexpr std::string_view kEsm2_8mSoftGapOverrideMessage =
    "user-provided affine gap values override hikoboshi-esm2-8m soft-SW calibrated defaults gap_open=-6.72805 and gap_extension=-0.0159468";
constexpr std::string_view kEsm2_8mBothGapOverrideMessage =
    "user-provided affine gap values override hikoboshi-esm2-8m hard-SW calibrated defaults (gap_open=-1.01982, gap_extension=+0.225736) and soft-SW calibrated defaults (gap_open=-6.72805, gap_extension=-0.0159468)";

struct GapOverrideContext {
  float hard_gap_open;
  float hard_gap_extension;
  float soft_gap_open;
  float soft_gap_extension;
  AlignmentMode mode;
  std::string_view hard_warning_message;
  std::string_view soft_warning_message;
  std::string_view both_warning_message;
};

bool alignment_field_overridden(float value, float default_value) noexcept {
  // A field is treated as a user override only if it is finite (i.e. the
  // package-default sentinel did not survive through to this point) and
  // diverges from the package's calibrated value. The sentinel itself is
  // not an override — the engine resolves it to the package default before
  // this comparison runs.
  return !is_package_default_gap(value) && value != default_value;
}

bool hard_gap_defaults_overridden_against(
    const AlignmentOptions& options,
    const GapOverrideContext& ctx) noexcept {
  return alignment_mode_runs_hard(ctx.mode) &&
         (alignment_field_overridden(options.gap_open, ctx.hard_gap_open) ||
          alignment_field_overridden(options.gap_extension,
                                     ctx.hard_gap_extension));
}

bool soft_gap_defaults_overridden_against(
    const AlignmentOptions& options,
    const GapOverrideContext& ctx) noexcept {
  return alignment_mode_runs_soft(ctx.mode) &&
         (alignment_field_overridden(options.gap_open, ctx.soft_gap_open) ||
          alignment_field_overridden(options.gap_extension,
                                     ctx.soft_gap_extension));
}

bool gap_defaults_overridden_against(const AlignmentOptions& options,
                                     const GapOverrideContext& ctx) noexcept {
  return hard_gap_defaults_overridden_against(options, ctx) ||
         soft_gap_defaults_overridden_against(options, ctx);
}

GapOverrideContext mpnn64_gap_override_context(AlignmentMode mode) noexcept {
  return {kDefaultGapOpen,
          kDefaultGapExtension,
          kMpnn64SoftGapOpen,
          kMpnn64SoftGapExtension,
          mode,
          kMpnn64HardGapOverrideMessage,
          kMpnn64SoftGapOverrideMessage,
          kMpnn64BothGapOverrideMessage};
}

bool gap_defaults_overridden(const AlignmentOptions& options) noexcept {
  return gap_defaults_overridden_against(
      options, mpnn64_gap_override_context(AlignmentMode::Hard));
}

GapOverrideContext gap_override_context_for_descriptor(
    const universal::PackageDescriptor& descriptor,
    AlignmentMode mode) noexcept {
  if (descriptor.identity.package_id == kEsm2_8mPackageId ||
      descriptor.gaps.gap_open != kDefaultGapOpen ||
      descriptor.gaps.gap_extension != kDefaultGapExtension) {
    return {descriptor.gaps.gap_open,
            descriptor.gaps.gap_extension,
            descriptor.soft_gaps.gap_open,
            descriptor.soft_gaps.gap_extension,
            mode,
            kEsm2_8mHardGapOverrideMessage,
            kEsm2_8mSoftGapOverrideMessage,
            kEsm2_8mBothGapOverrideMessage};
  }
  return {descriptor.gaps.gap_open,
          descriptor.gaps.gap_extension,
          descriptor.soft_gaps.gap_open,
          descriptor.soft_gaps.gap_extension,
          mode,
          kMpnn64HardGapOverrideMessage,
          kMpnn64SoftGapOverrideMessage,
          kMpnn64BothGapOverrideMessage};
}

void set_gap_override_warning(const AlignmentOptions& options,
                              const GapOverrideContext& ctx,
                              PairwiseResult& result) {
  if (!gap_defaults_overridden_against(options, ctx)) {
    result.warnings.clear();
    return;
  }
  result.warnings.resize(1U);
  const bool hard_overridden = hard_gap_defaults_overridden_against(options, ctx);
  const bool soft_overridden = soft_gap_defaults_overridden_against(options, ctx);
  std::string_view message = ctx.both_warning_message;
  if (hard_overridden && !soft_overridden) {
    message = ctx.hard_warning_message;
  } else if (!hard_overridden && soft_overridden) {
    message = ctx.soft_warning_message;
  }
  result.warnings[0] =
      {universal::PackageWarningKind::GapDefaultsOverridden,
       universal::PackageValidationStage::GapModelDefaults,
       "gap_defaults_overridden",
       message};
}

void assign_api_result(const algorithms::PairwiseResult& source,
                       PairwiseResult& destination) {
  assign_api_path(source.path, destination.path);
  destination.metrics = to_api_metrics(source.metrics);
  destination.warnings.clear();
}

void assign_api_result(const PairwiseResult& source,
                       PairwiseResult& destination) {
  assign_api_path(source.path, destination.path);
  destination.metrics = source.metrics;
  destination.warnings.resize(source.warnings.size());
  std::copy(source.warnings.begin(), source.warnings.end(),
            destination.warnings.begin());
}

PairwiseResult to_api_result(const algorithms::PairwiseResult& result) {
  PairwiseResult api_result{};
  assign_api_result(result, api_result);
  return api_result;
}

void reserve_record_storage(PairwiseResultRecord& record,
                            std::size_t max_result_step_count,
                            bool reserve_gap_warning) {
  record.result.path.steps.reserve(max_result_step_count);
  if (reserve_gap_warning) {
    record.result.warnings.reserve(1U);
  }
}

class CollectingAllVsAllSink final : public PairwiseResultSink {
 public:
  CollectingAllVsAllSink(AllVsAllResult& result,
                         std::size_t expected_pair_count,
                         std::size_t max_result_step_count,
                         const AlignmentOptions& alignment)
      : result_(result),
        expected_pair_count_(expected_pair_count),
        max_result_step_count_(max_result_step_count),
        reserve_gap_warning_(gap_defaults_overridden(alignment)) {}

  universal::Status prepare() {
    try {
      result_.records.clear();
      result_.records.resize(expected_pair_count_);
      for (PairwiseResultRecord& record : result_.records) {
        reserve_record_storage(record, max_result_step_count_,
                               reserve_gap_warning_);
      }
    } catch (const std::bad_alloc&) {
      return universal::unavailable_status(
          "all-vs-all collected result allocation failed");
    } catch (const std::length_error&) {
      return universal::invalid_argument_status(
          "all-vs-all collected result capacity overflows");
    }
    return universal::ok_status();
  }

  universal::Status receive(const PairwiseResultRecord& record) override {
    if (next_index_ >= result_.records.size()) {
      return universal::internal_error_status(
          "all-vs-all collected result exceeded preallocated pair count");
    }
    PairwiseResultRecord& destination = result_.records[next_index_];
    destination.query_index = record.query_index;
    destination.target_index = record.target_index;
    assign_api_result(record.result, destination.result);
    ++next_index_;
    return universal::ok_status();
  }

  void finalize(universal::Status status) {
    if (!universal::is_ok(status) || next_index_ < result_.records.size()) {
      result_.records.resize(next_index_);
    }
  }

 private:
  AllVsAllResult& result_;
  std::size_t expected_pair_count_ = 0;
  std::size_t max_result_step_count_ = 0;
  bool reserve_gap_warning_ = false;
  std::size_t next_index_ = 0;
};

std::size_t all_vs_all_pair_count(std::size_t item_count,
                                  bool include_self) noexcept {
  return algorithms::detail::symmetric_pair_count(item_count, include_self);
}

std::size_t hardware_concurrency_or_one() noexcept {
  const unsigned hardware = std::thread::hardware_concurrency();
  return hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
}

std::size_t resolve_engine_thread_count(std::uint32_t requested,
                                        std::size_t pair_count) noexcept {
  return algorithms::detail::resolve_all_vs_all_auto_thread_count(
      requested, pair_count, hardware_concurrency_or_one());
}

std::size_t linux_mem_available_bytes() {
#if defined(__linux__)
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  std::size_t kibibytes = 0;
  std::string unit;
  while (meminfo >> key >> kibibytes >> unit) {
    if (key == "MemAvailable:") {
      if (kibibytes >
          std::numeric_limits<std::size_t>::max() / 1024U) {
        return std::numeric_limits<std::size_t>::max();
      }
      return kibibytes * 1024U;
    }
  }
#endif
  return 0U;
}

std::size_t phase1_workspace_budget_bytes() {
  const std::size_t available = linux_mem_available_bytes();
  return available / kPhase1MemoryBudgetDivisor;
}

std::size_t fallback_phase1_thread_count(std::size_t requested_thread_count,
                                         std::size_t item_count) noexcept {
  if (requested_thread_count <= 1U ||
      item_count <
          algorithms::detail::kAllVsAllParallelEncodingThreshold) {
    return 1U;
  }
  // If Linux MemAvailable cannot be read, keep Phase 1 bounded to two workers.
  // The exact workspace preparation still runs before worker launch, so
  // allocation failure remains a normal Unavailable status instead of a race.
  return std::min({requested_thread_count,
                   item_count,
                   kPhase1MeminfoFallbackThreadCap});
}

std::size_t select_structure_phase1_thread_count(
    std::size_t requested_thread_count,
    std::size_t item_count,
    std::size_t workspace_bytes_per_thread) {
  if (workspace_bytes_per_thread == 0U) {
    return fallback_phase1_thread_count(requested_thread_count, item_count);
  }
  const std::size_t budget_bytes = phase1_workspace_budget_bytes();
  if (budget_bytes == 0U) {
    return fallback_phase1_thread_count(requested_thread_count, item_count);
  }
  return algorithms::detail::select_all_vs_all_phase1_thread_count_for_budget(
      requested_thread_count,
      item_count,
      workspace_bytes_per_thread,
      budget_bytes);
}

std::size_t max_structure_residue_count(
    universal::Span<const universal::StructureView> structures) noexcept {
  std::size_t max_residue_count = 0;
  if (structures.data == nullptr) {
    return max_residue_count;
  }
  for (std::size_t index = 0; index < structures.size; ++index) {
    max_residue_count =
        std::max(max_residue_count, structures.data[index].residue_count);
  }
  return max_residue_count;
}

std::size_t max_coords_residue_count(
    universal::Span<const CoordsInputView> coords) noexcept {
  std::size_t max_residue_count = 0;
  if (coords.data == nullptr) {
    return max_residue_count;
  }
  for (std::size_t index = 0; index < coords.size; ++index) {
    max_residue_count =
        std::max(max_residue_count, coords.data[index].residue_count);
  }
  return max_residue_count;
}

std::size_t max_embedding_residue_count(
    universal::Span<const universal::EmbeddingView> embeddings) noexcept {
  std::size_t max_residue_count = 0;
  if (embeddings.data == nullptr) {
    return max_residue_count;
  }
  for (std::size_t index = 0; index < embeddings.size; ++index) {
    max_residue_count =
        std::max(max_residue_count, embeddings.data[index].residue_count);
  }
  return max_residue_count;
}

universal::Status max_traceback_step_count(
    std::size_t max_residue_count,
    std::size_t& max_result_step_count) noexcept {
  if (max_residue_count >
      std::numeric_limits<std::size_t>::max() / 2U) {
    return universal::invalid_argument_status(
        "all-vs-all traceback step capacity overflows");
  }
  max_result_step_count = max_residue_count * 2U;
  return universal::ok_status();
}

universal::Status pair_traceback_step_count(
    std::size_t max_query_length,
    std::size_t max_target_length,
    std::size_t& max_result_step_count) noexcept {
  if (max_query_length >
      std::numeric_limits<std::size_t>::max() - max_target_length) {
    return universal::invalid_argument_status(
        "pair-list traceback step capacity overflows");
  }
  max_result_step_count = max_query_length + max_target_length;
  return universal::ok_status();
}

struct PairListAxisMaxima {
  std::size_t max_query_length = 0;
  std::size_t max_target_length = 0;
};

template <typename LengthAt>
PairListAxisMaxima pair_list_axis_maxima(
    const std::vector<std::pair<std::size_t, std::size_t>>& pairs,
    LengthAt&& length_at) noexcept {
  PairListAxisMaxima maxima{};
  for (const std::pair<std::size_t, std::size_t>& pair : pairs) {
    maxima.max_query_length =
        std::max(maxima.max_query_length, length_at(pair.first));
    maxima.max_target_length =
        std::max(maxima.max_target_length, length_at(pair.second));
  }
  return maxima;
}

std::size_t structure_phase1_workspace_bytes_per_thread(
    std::size_t max_residue_count) noexcept {
  std::size_t bytes = 0;
  if (!algorithms::detail::
          estimate_all_vs_all_structure_encoder_workspace_bytes(
              max_residue_count, mpnn64_descriptor(), bytes)) {
    return 0U;
  }
  return bytes;
}

// Mirror of structure_phase1_workspace_bytes_per_thread for the ESM2
// sequence route. Sized from the actual maximum token count (the real data
// max), never the descriptor's kEsm2_8mMaxSequenceLength guard (65536) —
// that O(L^2) attention scratch would estimate ~17 GB/thread and starve the
// phase-1 thread budget to a single worker. Returns 0 on overflow, which the
// caller treats as "unknown" and falls back to the meminfo thread cap.
std::size_t sequence_phase1_workspace_bytes_per_thread(
    std::size_t max_token_count) noexcept {
  std::size_t bytes = 0;
  if (!algorithms::detail::
          estimate_all_vs_all_sequence_encoder_workspace_bytes(
              max_token_count, esm2_8m_descriptor(), bytes)) {
    return 0U;
  }
  return bytes;
}

struct AllVsAllThreadingPlan {
  std::size_t pair_count = 0;
  std::size_t encoding_item_count = 0;
  std::size_t encoder_workspace_bytes_per_thread = 0;
};

struct AllVsAllThreadingLease {
  std::unique_lock<std::mutex> lock{};
  universal::detail::ThreadPool* pool = nullptr;
  std::size_t thread_count = 1;
  universal::Span<algorithms::detail::AllVsAllWorkerWorkspace> workspaces{
      nullptr,
      0};
};

AllVsAllThreadingLease acquire_all_vs_all_threading(
    EngineThreadingState* state,
    std::uint32_t requested_thread_count,
    const AllVsAllThreadingPlan& plan) {
  if (!state || requested_thread_count == 1U) {
    return {};
  }

  const bool pair_phase_eligible = plan.pair_count >= kParallelPairThreshold;
  const bool encoding_phase_eligible =
      plan.encoding_item_count >=
      algorithms::detail::kAllVsAllParallelEncodingThreshold;
  if (!pair_phase_eligible && !encoding_phase_eligible) {
    return {};
  }

  std::size_t effective_thread_count =
      resolve_engine_thread_count(requested_thread_count, plan.pair_count);
  if (encoding_phase_eligible) {
    effective_thread_count = select_structure_phase1_thread_count(
        effective_thread_count,
        plan.encoding_item_count,
        plan.encoder_workspace_bytes_per_thread);
  }
  if (effective_thread_count <= 1U) {
    return {};
  }

  AllVsAllThreadingLease lease{};
  lease.lock = std::unique_lock<std::mutex>(state->mutex);
  if (!state->pool ||
      state->pool_thread_count != effective_thread_count) {
    state->pool =
        std::make_unique<universal::detail::ThreadPool>(
            effective_thread_count);
    state->pool_thread_count = effective_thread_count;
  }
  lease.pool = state->pool.get();
  lease.thread_count = lease.pool->thread_count();
  if (state->pair_workspaces.size() < lease.thread_count) {
    state->pair_workspaces.resize(lease.thread_count);
  }
  lease.workspaces = {state->pair_workspaces.data(),
                      state->pair_workspaces.size()};
  return lease;
}

void assign_api_record(
    const algorithms::PairwiseResultRecord& record,
    const AlignmentOptions& alignment,
    const GapOverrideContext& gap_override_ctx,
    PairwiseResultRecord& api_record) {
  api_record.query_index = record.query_index;
  api_record.target_index = record.target_index;
  assign_api_result(record.result, api_record.result);
  set_gap_override_warning(alignment, gap_override_ctx, api_record.result);
}

class ForwardingAllVsAllSink final : public algorithms::PairwiseResultSink {
 public:
  ForwardingAllVsAllSink(hikoboshi::api::PairwiseResultSink& sink,
                         const AlignmentOptions& alignment,
                         std::size_t max_result_step_count,
                         GapOverrideContext gap_override_ctx)
      : sink_(sink),
        alignment_(alignment),
        max_result_step_count_(max_result_step_count),
        gap_override_ctx_(gap_override_ctx) {}

  universal::Status prepare() {
    try {
      reserve_record_storage(
          api_record_, max_result_step_count_,
          gap_defaults_overridden_against(alignment_, gap_override_ctx_));
    } catch (const std::bad_alloc&) {
      return universal::unavailable_status(
          "all-vs-all forwarding result allocation failed");
    } catch (const std::length_error&) {
      return universal::invalid_argument_status(
          "all-vs-all forwarding result capacity overflows");
    }
    return universal::ok_status();
  }

  universal::Status receive(
      const algorithms::PairwiseResultRecord& record) override {
    assign_api_record(record, alignment_, gap_override_ctx_, api_record_);
    return sink_.receive(api_record_);
  }

 private:
  hikoboshi::api::PairwiseResultSink& sink_;
  const AlignmentOptions& alignment_;
  std::size_t max_result_step_count_ = 0;
  GapOverrideContext gap_override_ctx_;
  hikoboshi::api::PairwiseResultRecord api_record_{};
};

struct PairwiseExecutionCache {
  algorithms::detail::PairwiseWorkspace embedding_workspace;
  algorithms::detail::PairwiseWorkspace structure_workspace;
  algorithms::PairwiseResult embedding_result;
  algorithms::PairwiseResult structure_result;
};

PairwiseExecutionCache& pairwise_execution_cache() {
  thread_local PairwiseExecutionCache cache;
  return cache;
}

void copy_residue_metadata(const universal::StructureView& structure,
                           EncodedEmbedding& embedding) {
  if (structure.residue_codes.data != nullptr &&
      structure.residue_codes.size >= structure.residue_count) {
    embedding.residue_codes.assign(structure.residue_codes.data,
                                   structure.residue_codes.data +
                                       structure.residue_count);
  }
  if (structure.residues.data != nullptr &&
      structure.residues.size >= structure.residue_count) {
    embedding.residues.assign(structure.residues.data,
                              structure.residues.data +
                                  structure.residue_count);
  }
}

universal::Result<PairwiseResult> run_pairwise_embedding_request(
    const PairwiseEmbeddingRequest& request) {
  universal::Status status =
      validate_pairwise_alignment_mode(request.mode, request.temperature);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  status = validate_embedding_view(request.query);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  status = validate_embedding_view(request.target);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  if (request.query.dimension != request.target.dimension) {
    return universal::result_from_status<PairwiseResult>(
        universal::invalid_argument_status("pairwise embedding dimensions differ"));
  }

  const bool hard_mode = alignment_mode_runs_hard(request.mode);
  const bool soft_mode = alignment_mode_runs_soft(request.mode);
  PairwiseExecutionCache& cache = pairwise_execution_cache();
  status = cache.embedding_workspace.prepare(
      embedding_plan(request.query, request.target, soft_mode));
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }

  algorithms::PairwiseEmbeddingRequest algorithm_request{};
  algorithm_request.query_embedding = request.query;
  algorithm_request.target_embedding = request.target;
  const ResolvedAlignmentOptions resolved_alignment =
      resolve_alignment_against_mpnn64(request.alignment);
  algorithm_request.options = to_algorithms_options(resolved_alignment);
  algorithm_request.hard_mode = hard_mode;
  algorithm_request.soft_mode = soft_mode;
  algorithm_request.temperature = request.temperature;

  status = algorithms::run_pairwise_embeddings(algorithm_request,
                                               cache.embedding_workspace,
                                               cache.embedding_result);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  PairwiseResult result = to_api_result(cache.embedding_result);
  set_gap_override_warning(
      request.alignment, mpnn64_gap_override_context(request.mode), result);
  return {universal::ok_status(), result};
}

universal::Result<PairwiseResult> run_pairwise_structure_request(
    const universal::StructureView& query,
    const universal::StructureView& target,
    const AlignmentOptions& alignment,
    AlignmentMode mode,
    float temperature,
    const EngineConfig& config,
    universal::PackageInputKind required_route) {
  universal::Status status =
      validate_pairwise_alignment_mode(mode, temperature);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  universal::Result<universal::WeightsHandle> weights =
      resolve_structure_package_weights(config, required_route);
  if (!universal::is_ok(weights.status)) {
    return universal::result_from_status<PairwiseResult>(weights.status);
  }

  status = validate_structure_view(query);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  status = validate_structure_view(target);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }

  const bool hard_mode = alignment_mode_runs_hard(mode);
  const bool soft_mode = alignment_mode_runs_soft(mode);
  PairwiseExecutionCache& cache = pairwise_execution_cache();
  status =
      cache.structure_workspace.prepare(structure_plan(query, target, soft_mode));
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }

  algorithms::PairwiseStructureRequest algorithm_request{};
  algorithm_request.query = query;
  algorithm_request.target = target;
  algorithm_request.descriptor = mpnn64_descriptor();
  algorithm_request.weights = prepared_weights(weights.value);
  const ResolvedAlignmentOptions resolved_alignment =
      config.package.descriptor != nullptr
          ? resolve_alignment_against_package(alignment,
                                              *config.package.descriptor)
          : resolve_alignment_against_mpnn64(alignment);
  const GapOverrideContext gap_override_ctx =
      config.package.descriptor != nullptr
          ? gap_override_context_for_descriptor(*config.package.descriptor,
                                                mode)
          : mpnn64_gap_override_context(mode);
  algorithm_request.options = to_algorithms_options(resolved_alignment);
  algorithm_request.hard_mode = hard_mode;
  algorithm_request.soft_mode = soft_mode;
  algorithm_request.temperature = temperature;

  status = algorithms::run_pairwise_structures(algorithm_request,
                                               cache.structure_workspace,
                                               cache.structure_result);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  PairwiseResult result = to_api_result(cache.structure_result);
  set_gap_override_warning(alignment, gap_override_ctx, result);
  return {universal::ok_status(), result};
}

universal::Result<EncodeResult> encode_structure_view(
    const universal::StructureView& structure,
    const EngineConfig& config,
    universal::PackageInputKind required_route) {
  universal::Result<universal::WeightsHandle> weights =
      resolve_structure_package_weights(config, required_route);
  if (!universal::is_ok(weights.status)) {
    return universal::result_from_status<EncodeResult>(weights.status);
  }

  universal::Status status = validate_structure_view(structure);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<EncodeResult>(status);
  }

  universal::Result<std::vector<float>> encoded =
      encode_mpnn64_structure_direct(structure, weights.value);
  if (!universal::is_ok(encoded.status)) {
    return universal::result_from_status<EncodeResult>(encoded.status);
  }

  EncodeResult result{};
  result.embedding.residue_count = structure.residue_count;
  result.embedding.dimension = kMpnn64HiddenDimension;
  result.embedding.values = std::move(encoded.value);
  copy_residue_metadata(structure, result.embedding);
  return {universal::ok_status(), result};
}

algorithms::detail::PairwiseWorkspacePlan esm2_sequence_pairwise_plan(
    std::size_t query_token_count,
    std::size_t target_token_count,
    bool allocate_soft_sw) noexcept {
  algorithms::detail::PairwiseWorkspacePlan plan{};
  // The engine wraps raw AA tokens as `[<cls>, aa..., <eos>]` before
  // invoking the encoder, so workspace capacity reserved for the embedding
  // buffers includes the CLS/EOS overhead. Downstream similarity and
  // pairwise traverse only the L residue rows (CLS at row 0 and EOS at
  // row L+1 are skipped), so similarity/posterior allocations stay sized
  // by `query_token_count`/`target_token_count`.
  plan.max_query_length = query_token_count + kEsm2_8mSpecialTokenOverhead;
  plan.max_target_length = target_token_count + kEsm2_8mSpecialTokenOverhead;
  plan.embedding_dimension = kEsm2_8mHiddenDimension;
  plan.allocate_mpnn = false;
  plan.allocate_soft_sw = allocate_soft_sw;
  return plan;
}

universal::Result<PairwiseResult> run_pairwise_sequence_request(
    const PairwiseSequenceRequest& request,
    const EngineConfig& config) {
  universal::Status status =
      validate_pairwise_alignment_mode(request.mode, request.temperature);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  if (request.query_token_ids.data == nullptr ||
      request.target_token_ids.data == nullptr ||
      request.query_token_ids.size == 0 ||
      request.target_token_ids.size == 0) {
    return universal::result_from_status<PairwiseResult>(
        universal::invalid_argument_status(
            "pairwise sequence request requires non-empty token spans"));
  }

  universal::Result<const universal::WeightsView*> view =
      resolve_sequence_package_weights_view(config);
  if (!universal::is_ok(view.status)) {
    return universal::result_from_status<PairwiseResult>(view.status);
  }

  const bool hard_mode = alignment_mode_runs_hard(request.mode);
  const bool soft_mode = alignment_mode_runs_soft(request.mode);
  PairwiseExecutionCache& cache = pairwise_execution_cache();
  status = cache.embedding_workspace.prepare(esm2_sequence_pairwise_plan(
      request.query_token_ids.size, request.target_token_ids.size, soft_mode));
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }

  // Resolve gap defaults from the resolved sequence-route package descriptor
  // before handing the alignment options to the algorithms layer. Bindings
  // and CLI adapters pass the `kPackageDefaultGapSentinel` (NaN) sentinel
  // when the caller did not specify gap values; the engine substitutes the
  // package descriptor's calibrated values so each sequence package's
  // trained gap parameters propagate without each adapter having to know
  // them. fe2 traced the bl5 soft-SW pilot collapse to this path silently
  // using MPNN-64 defaults (-1.4 / -0.15) instead of hikoboshi-esm2-8m's
  // hard-SW calibrated (-1.01982 / +0.225736).
  const universal::PackageDescriptor& sequence_descriptor =
      *config.package.descriptor;
  const ResolvedAlignmentOptions resolved_alignment =
      resolve_alignment_against_package(request.alignment, sequence_descriptor);
  const GapOverrideContext gap_override_ctx =
      gap_override_context_for_descriptor(sequence_descriptor, request.mode);

  // `algorithms::run_pairwise_sequences` is responsible for wrapping raw
  // AA tokens as `[<cls>, aa..., <eos>]`, running the encoder forward on
  // the wrapped span, and exposing only the residue rows downstream. The
  // workspace plan above already reserves CLS/EOS overhead so the wrap is
  // an offset transform on the existing embedding buffer.
  algorithms::PairwiseSequenceRequest algorithm_request{};
  algorithm_request.query_token_ids = request.query_token_ids;
  algorithm_request.target_token_ids = request.target_token_ids;
  algorithm_request.descriptor = esm2_8m_descriptor();
  algorithm_request.weights_view = view.value;
  algorithm_request.options = to_algorithms_options(resolved_alignment);
  algorithm_request.hard_mode = hard_mode;
  algorithm_request.soft_mode = soft_mode;
  algorithm_request.temperature = request.temperature;

  status = algorithms::run_pairwise_sequences(algorithm_request,
                                              cache.embedding_workspace,
                                              cache.embedding_result);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  PairwiseResult result = to_api_result(cache.embedding_result);
  set_gap_override_warning(request.alignment, gap_override_ctx, result);
  return {universal::ok_status(), result};
}

universal::Result<EncodeResult> encode_sequence_view(
    universal::Span<const std::int32_t> token_ids,
    const EngineConfig& config) {
  universal::Result<const universal::WeightsView*> view =
      resolve_sequence_package_weights_view(config);
  if (!universal::is_ok(view.status)) {
    return universal::result_from_status<EncodeResult>(view.status);
  }
  if (token_ids.data == nullptr || token_ids.size == 0) {
    return universal::result_from_status<EncodeResult>(
        universal::invalid_argument_status(
            "sequence encode request token span must be non-empty"));
  }

  // The engine wraps the raw AA token span with the training-time CLS/EOS
  // tokens before invoking the encoder, then strips the corresponding
  // CLS/EOS embedding rows from the result. The public contract for
  // `encode_from_sequence` is unchanged: callers pass raw AA tokens and
  // receive `[L, hidden]` residue-level embeddings.
  const std::vector<std::int32_t> wrapped_tokens =
      wrap_esm2_sequence_tokens(token_ids);
  std::vector<float> wrapped_embeddings;
  const std::size_t wrapped_value_count =
      wrapped_tokens.size() * kEsm2_8mHiddenDimension;
  try {
    wrapped_embeddings.assign(wrapped_value_count, 0.0F);
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<EncodeResult>(
        universal::unavailable_status(
            "sequence encode wrapped-token embedding allocation failed"));
  }

  const universal::Status status = algorithms::detail::encode_esm2_sequence(
      *view.value, esm2_8m_descriptor(),
      universal::Span<const std::int32_t>{wrapped_tokens.data(),
                                          wrapped_tokens.size()},
      wrapped_embeddings.data());
  if (!universal::is_ok(status)) {
    return universal::result_from_status<EncodeResult>(status);
  }

  EncodeResult result{};
  result.embedding.residue_count = token_ids.size;
  result.embedding.dimension = kEsm2_8mHiddenDimension;
  const std::size_t value_count = token_ids.size * kEsm2_8mHiddenDimension;
  try {
    result.embedding.values.assign(value_count, 0.0F);
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<EncodeResult>(
        universal::unavailable_status(
            "sequence encode embedding allocation failed"));
  }
  // Copy residue rows (skip CLS at index 0 and EOS at the last index).
  std::copy(wrapped_embeddings.data() + kEsm2_8mHiddenDimension,
            wrapped_embeddings.data() + kEsm2_8mHiddenDimension +
                value_count,
            result.embedding.values.data());
  return {universal::ok_status(), result};
}

// ---- Pair-list (npc1b) ----------------------------------------------------
//
// The pair-list engine entries dedup the caller's (query_id, target_id)
// string pairs into a unique protein set, hand that compacted set plus the
// resolved index pairs to the algorithm layer (`run_pair_list_*`), and
// collect one record per input pair — remapping the algorithm layer's
// compacted protein indices back to the caller's source-span indices.
// Protein IDs are case-sensitive and never normalized; a pair referencing
// an ID absent from the source is a fail-fast error naming the missing ID.

// `universal::Status::detail` is a borrowed `const char*`, so a status that
// must name a caller-supplied ID needs backing storage that outlives the
// return. A thread-local string holds the most recent pair-list error
// message on each thread; `Status.detail` points into it and stays valid
// until the next pair-list error on the same thread.
const char* pair_list_error_detail(const std::string& message) {
  thread_local std::string buffer;
  buffer = message;
  return buffer.c_str();
}

// Resolve a caller pair list against `source_ids` (one identifier per
// source-span element; an empty identifier marks an element that carries
// no addressable ID). On success `original_index` lists, per unique
// referenced protein, its index in the source span — the encode-once
// unique set — and `compacted_pairs` lists each input pair as a
// `(query, target)` index into `original_index`, in input order.
universal::Status resolve_pair_list(
    universal::Span<const std::string_view> source_ids,
    const std::vector<std::pair<std::string, std::string>>& pairs,
    std::vector<std::size_t>& original_index,
    std::vector<std::pair<std::size_t, std::size_t>>& compacted_pairs) {
  original_index.clear();
  compacted_pairs.clear();
  try {
    // Build the source ID -> index table. Empty IDs are skipped: an
    // element with no identifier is simply unaddressable. A non-empty ID
    // that appears twice is ambiguous and rejected only if a pair
    // actually references it.
    std::unordered_map<std::string_view, std::size_t> id_to_index;
    std::unordered_set<std::string_view> ambiguous_ids;
    for (std::size_t index = 0; index < source_ids.size; ++index) {
      const std::string_view id = source_ids.data[index];
      if (id.empty()) {
        continue;
      }
      if (!id_to_index.emplace(id, index).second) {
        ambiguous_ids.insert(id);
      }
    }

    std::unordered_map<std::string_view, std::size_t> compacted_of;
    compacted_pairs.reserve(pairs.size());

    const auto resolve_one =
        [&](const std::string& raw_id,
            std::size_t& compacted_out) -> universal::Status {
      const std::string_view id{raw_id};
      if (ambiguous_ids.find(id) != ambiguous_ids.end()) {
        return universal::invalid_argument_status(pair_list_error_detail(
            "pair-list source contains a duplicate protein ID: '" + raw_id +
            "'"));
      }
      const auto found = id_to_index.find(id);
      if (found == id_to_index.end()) {
        return universal::invalid_argument_status(pair_list_error_detail(
            "pair-list ID not found in the input source: '" + raw_id + "'"));
      }
      const auto existing = compacted_of.find(id);
      if (existing != compacted_of.end()) {
        compacted_out = existing->second;
        return universal::ok_status();
      }
      compacted_out = original_index.size();
      original_index.push_back(found->second);
      compacted_of.emplace(id, compacted_out);
      return universal::ok_status();
    };

    for (const std::pair<std::string, std::string>& pair : pairs) {
      std::size_t query_compacted = 0;
      std::size_t target_compacted = 0;
      universal::Status status = resolve_one(pair.first, query_compacted);
      if (!universal::is_ok(status)) {
        return status;
      }
      status = resolve_one(pair.second, target_compacted);
      if (!universal::is_ok(status)) {
        return status;
      }
      compacted_pairs.push_back({query_compacted, target_compacted});
    }
  } catch (const std::bad_alloc&) {
    return universal::unavailable_status(
        "pair-list ID resolution allocation failed");
  }
  return universal::ok_status();
}

// Collecting sink for the pair-list engine entries. Receives algorithm-
// layer records (whose indices are into the compacted unique set), remaps
// them to the caller's source-span indices, converts them to api records,
// and stores one per input pair in input order. Combines the forward +
// collect roles the all-vs-all path splits across `ForwardingAllVsAllSink`
// and `CollectingAllVsAllSink`, plus the compacted -> source index remap.
class CollectingPairListSink final : public algorithms::PairwiseResultSink {
 public:
  CollectingPairListSink(AllVsAllResult& result,
                         const std::vector<std::size_t>& original_index,
                         std::size_t expected_pair_count,
                         std::size_t max_result_step_count,
                         const AlignmentOptions& alignment,
                         GapOverrideContext gap_override_ctx)
      : result_(result),
        original_index_(original_index),
        expected_pair_count_(expected_pair_count),
        max_result_step_count_(max_result_step_count),
        alignment_(alignment),
        gap_override_ctx_(gap_override_ctx) {}

  universal::Status prepare() {
    try {
      result_.records.clear();
      result_.records.resize(expected_pair_count_);
      for (PairwiseResultRecord& record : result_.records) {
        reserve_record_storage(
            record, max_result_step_count_,
            gap_defaults_overridden_against(alignment_, gap_override_ctx_));
      }
    } catch (const std::bad_alloc&) {
      return universal::unavailable_status(
          "pair-list collected result allocation failed");
    } catch (const std::length_error&) {
      return universal::invalid_argument_status(
          "pair-list collected result capacity overflows");
    }
    return universal::ok_status();
  }

  universal::Status receive(
      const algorithms::PairwiseResultRecord& record) override {
    if (next_index_ >= result_.records.size()) {
      return universal::internal_error_status(
          "pair-list collected result exceeded the input pair count");
    }
    if (record.query_index >= original_index_.size() ||
        record.target_index >= original_index_.size()) {
      return universal::internal_error_status(
          "pair-list record references an unknown compacted protein index");
    }
    PairwiseResultRecord& destination = result_.records[next_index_];
    assign_api_record(record, alignment_, gap_override_ctx_, destination);
    // Remap compacted unique-set indices back to the caller's source span.
    destination.query_index = original_index_[record.query_index];
    destination.target_index = original_index_[record.target_index];
    ++next_index_;
    return universal::ok_status();
  }

  void finalize(universal::Status status) {
    if (!universal::is_ok(status) || next_index_ < result_.records.size()) {
      result_.records.resize(next_index_);
    }
  }

 private:
  AllVsAllResult& result_;
  const std::vector<std::size_t>& original_index_;
  std::size_t expected_pair_count_ = 0;
  std::size_t max_result_step_count_ = 0;
  AlignmentOptions alignment_;
  GapOverrideContext gap_override_ctx_;
  std::size_t next_index_ = 0;
};

// Drive a resolved pair-list request through the algorithm layer and
// collect the records. `run` invokes the route's `run_pair_list_*` entry
// with the supplied sink.
template <typename RunFn>
universal::Result<AllVsAllResult> finish_pair_list(
    const std::vector<std::size_t>& original_index,
    std::size_t input_pair_count,
    std::size_t max_result_step_count,
    const AlignmentOptions& alignment,
    GapOverrideContext gap_override_ctx,
    RunFn&& run) {
  AllVsAllResult result{};
  CollectingPairListSink sink(result, original_index, input_pair_count,
                              max_result_step_count, alignment,
                              gap_override_ctx);
  universal::Status status = sink.prepare();
  if (!universal::is_ok(status)) {
    return {status, result};
  }
  status = run(sink);
  sink.finalize(status);
  return {status, result};
}

// Shared structure-route pair-list driver. The structure and coords engine
// entries differ only in how the source views and per-element IDs are
// obtained; both resolve to a `StructureView` span aligned through the
// MPNN-64 algorithm-layer pair-list entry.
universal::Result<AllVsAllResult> collect_pair_list_structures_common(
    universal::Span<const universal::StructureView> structures,
    universal::Span<const std::string_view> source_ids,
    const std::vector<std::pair<std::string, std::string>>& pairs,
    const AllVsAllOptions& options,
    const EngineConfig& config,
    EngineThreadingState* threading_state,
    universal::PackageInputKind required_route) {
  universal::Result<universal::WeightsHandle> weights =
      resolve_structure_package_weights(config, required_route);
  if (!universal::is_ok(weights.status)) {
    return universal::result_from_status<AllVsAllResult>(weights.status);
  }

  std::vector<std::size_t> original_index;
  std::vector<std::pair<std::size_t, std::size_t>> compacted_pairs;
  universal::Status status =
      resolve_pair_list(source_ids, pairs, original_index, compacted_pairs);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<AllVsAllResult>(status);
  }

  std::vector<universal::StructureView> compacted;
  std::size_t max_residue_count = 0;
  try {
    compacted.reserve(original_index.size());
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<AllVsAllResult>(
        universal::unavailable_status(
            "pair-list structure compaction allocation failed"));
  }
  for (const std::size_t original : original_index) {
    const universal::StructureView& view = structures.data[original];
    compacted.push_back(view);
    max_residue_count = std::max(max_residue_count, view.residue_count);
  }
  const PairListAxisMaxima axis_maxima = pair_list_axis_maxima(
      compacted_pairs,
      [&](std::size_t item_index) noexcept {
        return compacted[item_index].residue_count;
      });
  std::size_t max_result_step_count = 0;
  status = pair_traceback_step_count(axis_maxima.max_query_length,
                                     axis_maxima.max_target_length,
                                     max_result_step_count);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<AllVsAllResult>(status);
  }

  algorithms::PairListStructureRequest algorithm_request{};
  algorithm_request.structures = {compacted.data(), compacted.size()};
  algorithm_request.descriptor = mpnn64_descriptor();
  algorithm_request.weights = prepared_weights(weights.value);
  algorithm_request.pairs = {compacted_pairs.data(), compacted_pairs.size()};
  algorithm_request.max_query_length = axis_maxima.max_query_length;
  algorithm_request.max_target_length = axis_maxima.max_target_length;
  const ResolvedAlignmentOptions resolved_alignment =
      config.package.descriptor != nullptr
          ? resolve_alignment_against_package(options.alignment,
                                              *config.package.descriptor)
          : resolve_alignment_against_mpnn64(options.alignment);
  const GapOverrideContext gap_override_ctx =
      config.package.descriptor != nullptr
          ? gap_override_context_for_descriptor(*config.package.descriptor,
                                                options.mode)
          : mpnn64_gap_override_context(options.mode);
  algorithm_request.options = to_algorithms_options(options, resolved_alignment);

  // Acquire the shared all-vs-all pool so the encode-once pass and eligible
  // pair-list dispatch parallelize under the engine thread setting. The lease
  // must outlive `finish_pair_list`, which runs the lambda synchronously.
  AllVsAllThreadingPlan threading_plan{};
  threading_plan.pair_count = compacted_pairs.size();
  threading_plan.encoding_item_count = compacted.size();
  threading_plan.encoder_workspace_bytes_per_thread =
      structure_phase1_workspace_bytes_per_thread(max_residue_count);
  AllVsAllThreadingLease threading = acquire_all_vs_all_threading(
      threading_state,
      config.execution.thread_count,
      threading_plan);
  return finish_pair_list(
      original_index, pairs.size(), max_result_step_count, options.alignment,
      gap_override_ctx,
      [&](algorithms::PairwiseResultSink& sink) {
        return algorithms::run_pair_list_structures(algorithm_request, sink,
                                                    threading.pool,
                                                    threading.thread_count,
                                                    threading.workspaces);
      });
}

}  // namespace

Engine::Engine(EngineConfig config) : config_(config) {
  if (config_.execution.thread_count != 1U) {
    threading_ = std::make_shared<EngineThreadingState>();
  }
}

const EngineConfig& Engine::config() const noexcept {
  return config_;
}

universal::Result<EncodeResult> Engine::encode(
    const EncodeStructureRequest& request) const {
  const universal::Status status = validate_engine_axes(config_);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<EncodeResult>(status);
  }
  return encode_structure_view(
      request.structure, config_,
      universal::PackageInputKind::StructureBackboneAtoms);
}

universal::Result<EncodeResult> Engine::encode(const EncodeCoordsRequest& request) const {
  const universal::Status status = validate_engine_axes(config_);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<EncodeResult>(status);
  }
  return encode_structure_view(structure_from_coords(request.coords),
                               config_,
                               universal::PackageInputKind::CoordsBackbone);
}

universal::Result<EncodeResult> Engine::encode(
    const EncodeSequenceRequest& request) const {
  const universal::Status status = validate_engine_axes(config_);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<EncodeResult>(status);
  }
  return encode_sequence_view(request.token_ids, config_);
}

universal::Result<PairwiseResult> Engine::pairwise(
    const PairwiseStructureRequest& request) const {
  const universal::Status status = validate_engine_axes(config_);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  return run_pairwise_structure_request(request.query, request.target,
                                        request.alignment, request.mode,
                                        request.temperature, config_,
                                        universal::PackageInputKind::
                                            StructureBackboneAtoms);
}

universal::Result<PairwiseResult> Engine::pairwise(
    const PairwiseCoordsRequest& request) const {
  const universal::Status status = validate_engine_axes(config_);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  return run_pairwise_structure_request(structure_from_coords(request.query),
                                        structure_from_coords(request.target),
                                        request.alignment, request.mode,
                                        request.temperature, config_,
                                        universal::PackageInputKind::
                                            CoordsBackbone);
}

universal::Result<PairwiseResult> Engine::pairwise(
    const PairwiseEmbeddingRequest& request) const {
  const universal::Status status = validate_engine_axes(config_);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  return run_pairwise_embedding_request(request);
}

universal::Result<PairwiseResult> Engine::pairwise(
    const PairwiseSequenceRequest& request) const {
  const universal::Status status = validate_engine_axes(config_);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<PairwiseResult>(status);
  }
  return run_pairwise_sequence_request(request, config_);
}

universal::Status Engine::all_vs_all(const AllVsAllStructureRequest& request,
                                     PairwiseResultSink& sink) const {
  const universal::Status backend_status =
      validate_engine_axes(config_);
  if (!universal::is_ok(backend_status)) {
    return backend_status;
  }
  const universal::Status mode_status = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode_status)) {
    return mode_status;
  }
  const std::size_t pair_count =
      all_vs_all_pair_count(request.structures.size,
                            request.options.include_self);
  if (pair_count == 0U) {
    return universal::ok_status();
  }
  if (request.structures.data == nullptr) {
    return universal::invalid_argument_status("all-vs-all structure input span is invalid");
  }

  universal::Result<universal::WeightsHandle> weights =
      resolve_structure_package_weights(
          config_, universal::PackageInputKind::StructureBackboneAtoms);
  if (!universal::is_ok(weights.status)) {
    return weights.status;
  }

  algorithms::AllVsAllStructureRequest algorithm_request{};
  algorithm_request.structures = request.structures;
  algorithm_request.descriptor = mpnn64_descriptor();
  algorithm_request.weights = prepared_weights(weights.value);
  const ResolvedAlignmentOptions resolved_alignment =
      config_.package.descriptor != nullptr
          ? resolve_alignment_against_package(
                request.options.alignment, *config_.package.descriptor)
          : resolve_alignment_against_mpnn64(request.options.alignment);
  const GapOverrideContext gap_override_ctx =
      config_.package.descriptor != nullptr
          ? gap_override_context_for_descriptor(*config_.package.descriptor,
                                                request.options.mode)
          : mpnn64_gap_override_context(request.options.mode);
  algorithm_request.options =
      to_algorithms_options(request.options, resolved_alignment);

  const std::size_t max_residue_count =
      max_structure_residue_count(request.structures);
  std::size_t max_result_step_count = 0;
  universal::Status status =
      max_traceback_step_count(max_residue_count, max_result_step_count);
  if (!universal::is_ok(status)) {
    return status;
  }
  ForwardingAllVsAllSink algorithm_sink(sink,
                                        request.options.alignment,
                                        max_result_step_count,
                                        gap_override_ctx);
  status = algorithm_sink.prepare();
  if (!universal::is_ok(status)) {
    return status;
  }
  AllVsAllThreadingPlan threading_plan{};
  threading_plan.pair_count = pair_count;
  threading_plan.encoding_item_count = request.structures.size;
  threading_plan.encoder_workspace_bytes_per_thread =
      structure_phase1_workspace_bytes_per_thread(max_residue_count);
  AllVsAllThreadingLease threading = acquire_all_vs_all_threading(
      static_cast<EngineThreadingState*>(threading_.get()),
      config_.execution.thread_count,
      threading_plan);
  return algorithms::run_all_vs_all_structures(algorithm_request,
                                               algorithm_sink,
                                               threading.pool,
                                               threading.thread_count,
                                               threading.workspaces);
}

universal::Status Engine::all_vs_all(const AllVsAllCoordsRequest& request,
                                     PairwiseResultSink& sink) const {
  const universal::Status backend_status =
      validate_engine_axes(config_);
  if (!universal::is_ok(backend_status)) {
    return backend_status;
  }
  const universal::Status mode_status = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode_status)) {
    return mode_status;
  }
  const std::size_t pair_count =
      all_vs_all_pair_count(request.coords.size,
                            request.options.include_self);
  if (pair_count == 0U) {
    return universal::ok_status();
  }
  if (request.coords.data == nullptr) {
    return universal::invalid_argument_status("all-vs-all coordinate input span is invalid");
  }

  universal::Result<universal::WeightsHandle> weights =
      resolve_structure_package_weights(
          config_, universal::PackageInputKind::CoordsBackbone);
  if (!universal::is_ok(weights.status)) {
    return weights.status;
  }

  std::vector<universal::StructureView> structures;
  structures.reserve(request.coords.size);
  for (std::size_t index = 0; index < request.coords.size; ++index) {
    structures.push_back(structure_from_coords(request.coords.data[index]));
  }

  algorithms::AllVsAllStructureRequest algorithm_request{};
  algorithm_request.structures = {structures.data(), structures.size()};
  algorithm_request.descriptor = mpnn64_descriptor();
  algorithm_request.weights = prepared_weights(weights.value);
  const ResolvedAlignmentOptions resolved_alignment =
      config_.package.descriptor != nullptr
          ? resolve_alignment_against_package(
                request.options.alignment, *config_.package.descriptor)
          : resolve_alignment_against_mpnn64(request.options.alignment);
  const GapOverrideContext gap_override_ctx =
      config_.package.descriptor != nullptr
          ? gap_override_context_for_descriptor(*config_.package.descriptor,
                                                request.options.mode)
          : mpnn64_gap_override_context(request.options.mode);
  algorithm_request.options =
      to_algorithms_options(request.options, resolved_alignment);

  const std::size_t max_residue_count =
      max_structure_residue_count(algorithm_request.structures);
  std::size_t max_result_step_count = 0;
  universal::Status status =
      max_traceback_step_count(max_residue_count, max_result_step_count);
  if (!universal::is_ok(status)) {
    return status;
  }
  ForwardingAllVsAllSink algorithm_sink(sink,
                                        request.options.alignment,
                                        max_result_step_count,
                                        gap_override_ctx);
  status = algorithm_sink.prepare();
  if (!universal::is_ok(status)) {
    return status;
  }
  AllVsAllThreadingPlan threading_plan{};
  threading_plan.pair_count = pair_count;
  threading_plan.encoding_item_count = algorithm_request.structures.size;
  threading_plan.encoder_workspace_bytes_per_thread =
      structure_phase1_workspace_bytes_per_thread(max_residue_count);
  AllVsAllThreadingLease threading = acquire_all_vs_all_threading(
      static_cast<EngineThreadingState*>(threading_.get()),
      config_.execution.thread_count,
      threading_plan);
  return algorithms::run_all_vs_all_structures(algorithm_request,
                                               algorithm_sink,
                                               threading.pool,
                                               threading.thread_count,
                                               threading.workspaces);
}

universal::Status Engine::all_vs_all(const AllVsAllEmbeddingRequest& request,
                                     PairwiseResultSink& sink) const {
  const universal::Status backend_status =
      validate_engine_axes(config_);
  if (!universal::is_ok(backend_status)) {
    return backend_status;
  }
  const universal::Status mode_status = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode_status)) {
    return mode_status;
  }
  const std::size_t pair_count =
      all_vs_all_pair_count(request.embeddings.size,
                            request.options.include_self);
  if (pair_count == 0U) {
    return universal::ok_status();
  }
  if (request.embeddings.data == nullptr) {
    return universal::invalid_argument_status("all-vs-all embedding input span is invalid");
  }

  algorithms::AllVsAllEmbeddingRequest algorithm_request{};
  algorithm_request.embeddings = request.embeddings;
  const ResolvedAlignmentOptions resolved_alignment =
      resolve_alignment_against_mpnn64(request.options.alignment);
  const GapOverrideContext gap_override_ctx =
      mpnn64_gap_override_context(request.options.mode);
  algorithm_request.options =
      to_algorithms_options(request.options, resolved_alignment);

  std::size_t max_result_step_count = 0;
  universal::Status status =
      max_traceback_step_count(max_embedding_residue_count(request.embeddings),
                               max_result_step_count);
  if (!universal::is_ok(status)) {
    return status;
  }
  ForwardingAllVsAllSink algorithm_sink(sink,
                                        request.options.alignment,
                                        max_result_step_count,
                                        gap_override_ctx);
  status = algorithm_sink.prepare();
  if (!universal::is_ok(status)) {
    return status;
  }
  AllVsAllThreadingPlan threading_plan{};
  threading_plan.pair_count = pair_count;
  AllVsAllThreadingLease threading = acquire_all_vs_all_threading(
      static_cast<EngineThreadingState*>(threading_.get()),
      config_.execution.thread_count,
      threading_plan);
  return algorithms::run_all_vs_all_embeddings(algorithm_request,
                                               algorithm_sink,
                                               threading.pool,
                                               threading.thread_count,
                                               threading.workspaces);
}

universal::Status Engine::all_vs_all(const AllVsAllSequenceRequest& request,
                                     PairwiseResultSink& sink) const {
  const universal::Status backend_status = validate_engine_axes(config_);
  if (!universal::is_ok(backend_status)) {
    return backend_status;
  }
  const universal::Status mode_status = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode_status)) {
    return mode_status;
  }
  if (request.sequences.size != 0U && request.sequences.data == nullptr) {
    return universal::invalid_argument_status(
        "all-vs-all sequence input span is invalid");
  }
  const std::size_t pair_count = all_vs_all_pair_count(
      request.sequences.size, request.options.include_self);
  if (pair_count == 0U) {
    return universal::ok_status();
  }

  universal::Result<const universal::WeightsView*> view =
      resolve_sequence_package_weights_view(config_);
  if (!universal::is_ok(view.status)) {
    return view.status;
  }

  // Build the algorithms-layer sequence span. The api-layer SequenceEntry
  // and the algorithms-layer AllVsAllSequenceEntry share the same shape,
  // but the entries live in independent header namespaces so we copy a
  // borrowed view here.
  std::vector<algorithms::AllVsAllSequenceEntry> algorithm_sequences;
  std::size_t max_token_count = 0;
  try {
    algorithm_sequences.reserve(request.sequences.size);
  } catch (const std::bad_alloc&) {
    return universal::unavailable_status(
        "all-vs-all sequence algorithm-side span allocation failed");
  }
  for (std::size_t index = 0; index < request.sequences.size; ++index) {
    const SequenceEntry& entry = request.sequences.data[index];
    algorithms::AllVsAllSequenceEntry algorithm_entry{};
    algorithm_entry.name = entry.name;
    algorithm_entry.token_ids = entry.token_ids;
    algorithm_sequences.push_back(algorithm_entry);
    max_token_count = std::max(max_token_count, entry.token_ids.size);
  }

  std::size_t max_result_step_count = 0;
  universal::Status status =
      max_traceback_step_count(max_token_count, max_result_step_count);
  if (!universal::is_ok(status)) {
    return status;
  }
  // Resolve gap defaults from the resolved sequence-route package descriptor
  // before handing options to the algorithms layer, mirroring
  // `run_pairwise_sequence_request`. See the comment there for the full
  // rationale; without this resolution the bl5 helper invocation defaults
  // to MPNN-64 (-1.4 / -0.15) instead of hikoboshi-esm2-8m's calibrated
  // hard-SW calibrated (-1.01982 / +0.225736).
  const universal::PackageDescriptor& sequence_descriptor =
      *config_.package.descriptor;
  const ResolvedAlignmentOptions resolved_alignment =
      resolve_alignment_against_package(request.options.alignment,
                                        sequence_descriptor);
  const GapOverrideContext gap_override_ctx =
      gap_override_context_for_descriptor(sequence_descriptor,
                                          request.options.mode);
  ForwardingAllVsAllSink algorithm_sink(sink, request.options.alignment,
                                        max_result_step_count,
                                        gap_override_ctx);
  status = algorithm_sink.prepare();
  if (!universal::is_ok(status)) {
    return status;
  }

  algorithms::AllVsAllSequenceRequest algorithm_request{};
  algorithm_request.sequences = {algorithm_sequences.data(),
                                 algorithm_sequences.size()};
  algorithm_request.descriptor = esm2_8m_descriptor();
  algorithm_request.weights_view = view.value;
  algorithm_request.options =
      to_algorithms_options(request.options, resolved_alignment);

  // Mirror the structure handler: acquire the shared all-vs-all pool so the
  // sequence-route encode and per-pair dispatch parallelize under
  // `--threads`. Size the encoder workspace from the actual max_token_count,
  // not the descriptor's 65536 sequence guard.
  AllVsAllThreadingPlan threading_plan{};
  threading_plan.pair_count = pair_count;
  threading_plan.encoding_item_count = algorithm_sequences.size();
  threading_plan.encoder_workspace_bytes_per_thread =
      sequence_phase1_workspace_bytes_per_thread(max_token_count);
  AllVsAllThreadingLease threading = acquire_all_vs_all_threading(
      static_cast<EngineThreadingState*>(threading_.get()),
      config_.execution.thread_count,
      threading_plan);
  return algorithms::run_all_vs_all_sequences(algorithm_request,
                                              algorithm_sink,
                                              threading.pool,
                                              threading.thread_count,
                                              threading.workspaces);
}

universal::Result<AllVsAllResult> Engine::collect_all_vs_all(
    const AllVsAllStructureRequest& request) const {
  AllVsAllResult result{};
  const std::size_t pair_count =
      all_vs_all_pair_count(request.structures.size,
                            request.options.include_self);
  std::size_t max_result_step_count = 0;
  universal::Status status = universal::ok_status();
  if (pair_count != 0U) {
    status = max_traceback_step_count(
        max_structure_residue_count(request.structures),
        max_result_step_count);
    if (!universal::is_ok(status)) {
      return {status, result};
    }
  }
  CollectingAllVsAllSink sink(result,
                              pair_count,
                              max_result_step_count,
                              request.options.alignment);
  status = sink.prepare();
  if (!universal::is_ok(status)) {
    return {status, result};
  }
  status = all_vs_all(request, sink);
  sink.finalize(status);
  return {status, result};
}

universal::Result<AllVsAllResult> Engine::collect_all_vs_all(
    const AllVsAllCoordsRequest& request) const {
  AllVsAllResult result{};
  const std::size_t pair_count =
      all_vs_all_pair_count(request.coords.size,
                            request.options.include_self);
  std::size_t max_result_step_count = 0;
  universal::Status status = universal::ok_status();
  if (pair_count != 0U) {
    status = max_traceback_step_count(max_coords_residue_count(request.coords),
                                      max_result_step_count);
    if (!universal::is_ok(status)) {
      return {status, result};
    }
  }
  CollectingAllVsAllSink sink(result,
                              pair_count,
                              max_result_step_count,
                              request.options.alignment);
  status = sink.prepare();
  if (!universal::is_ok(status)) {
    return {status, result};
  }
  status = all_vs_all(request, sink);
  sink.finalize(status);
  return {status, result};
}

universal::Result<AllVsAllResult> Engine::collect_all_vs_all(
    const AllVsAllEmbeddingRequest& request) const {
  AllVsAllResult result{};
  const std::size_t pair_count =
      all_vs_all_pair_count(request.embeddings.size,
                            request.options.include_self);
  std::size_t max_result_step_count = 0;
  universal::Status status = universal::ok_status();
  if (pair_count != 0U) {
    status =
        max_traceback_step_count(max_embedding_residue_count(request.embeddings),
                                 max_result_step_count);
    if (!universal::is_ok(status)) {
      return {status, result};
    }
  }
  CollectingAllVsAllSink sink(result,
                              pair_count,
                              max_result_step_count,
                              request.options.alignment);
  status = sink.prepare();
  if (!universal::is_ok(status)) {
    return {status, result};
  }
  status = all_vs_all(request, sink);
  sink.finalize(status);
  return {status, result};
}

// Pair-list engine entries (npc1b). Each route dedups the caller's
// (query_id, target_id) string pairs into a unique protein set, encodes
// that set once through the algorithm layer, aligns exactly the listed
// pairs, and collects one record per input pair in input order. See
// `docs/charters/PAIR_LIST_CHARTER.md`.
universal::Result<AllVsAllResult> Engine::collect_pair_list(
    const PairListStructureRequest& request) const {
  const universal::Status axes = validate_engine_axes(config_);
  if (!universal::is_ok(axes)) {
    return universal::result_from_status<AllVsAllResult>(axes);
  }
  const universal::Status mode = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode)) {
    return universal::result_from_status<AllVsAllResult>(mode);
  }
  if (request.pairs.empty()) {
    return {universal::ok_status(), AllVsAllResult{}};
  }
  if (request.structures.size != 0U && request.structures.data == nullptr) {
    return universal::result_from_status<AllVsAllResult>(
        universal::invalid_argument_status(
            "pair-list structure input span is invalid"));
  }
  // Structure inputs carry their identifier in `StructureView::input_id`.
  std::vector<std::string_view> source_ids;
  try {
    source_ids.reserve(request.structures.size);
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<AllVsAllResult>(
        universal::unavailable_status(
            "pair-list structure id-table allocation failed"));
  }
  for (std::size_t index = 0; index < request.structures.size; ++index) {
    source_ids.push_back(request.structures.data[index].input_id);
  }
  return collect_pair_list_structures_common(
      request.structures, {source_ids.data(), source_ids.size()},
      request.pairs, request.options, config_,
      static_cast<EngineThreadingState*>(threading_.get()),
      universal::PackageInputKind::StructureBackboneAtoms);
}

universal::Result<AllVsAllResult> Engine::collect_pair_list(
    const PairListCoordsRequest& request) const {
  const universal::Status axes = validate_engine_axes(config_);
  if (!universal::is_ok(axes)) {
    return universal::result_from_status<AllVsAllResult>(axes);
  }
  const universal::Status mode = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode)) {
    return universal::result_from_status<AllVsAllResult>(mode);
  }
  if (request.pairs.empty()) {
    return {universal::ok_status(), AllVsAllResult{}};
  }
  if (request.coords.size != 0U && request.coords.data == nullptr) {
    return universal::result_from_status<AllVsAllResult>(
        universal::invalid_argument_status(
            "pair-list coordinate input span is invalid"));
  }
  // `CoordsInputView` has no dedicated identifier field; the per-residue
  // `source_id` metadata is the only protein-level ID it carries, so the
  // coords route resolves names through `residues[0].source_id`. A coords
  // input with no residue metadata is therefore unaddressable by name.
  std::vector<universal::StructureView> structures;
  std::vector<std::string_view> source_ids;
  try {
    structures.reserve(request.coords.size);
    source_ids.reserve(request.coords.size);
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<AllVsAllResult>(
        universal::unavailable_status(
            "pair-list coordinate id-table allocation failed"));
  }
  for (std::size_t index = 0; index < request.coords.size; ++index) {
    const CoordsInputView& coords = request.coords.data[index];
    structures.push_back(structure_from_coords(coords));
    source_ids.push_back(coords.residues.size != 0U
                             ? coords.residues.data[0].source_id
                             : std::string_view{});
  }
  return collect_pair_list_structures_common(
      {structures.data(), structures.size()},
      {source_ids.data(), source_ids.size()}, request.pairs, request.options,
      config_, static_cast<EngineThreadingState*>(threading_.get()),
      universal::PackageInputKind::CoordsBackbone);
}

universal::Result<AllVsAllResult> Engine::collect_pair_list(
    const PairListEmbeddingRequest& request) const {
  const universal::Status axes = validate_engine_axes(config_);
  if (!universal::is_ok(axes)) {
    return universal::result_from_status<AllVsAllResult>(axes);
  }
  const universal::Status mode = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode)) {
    return universal::result_from_status<AllVsAllResult>(mode);
  }
  if (request.pairs.empty()) {
    return {universal::ok_status(), AllVsAllResult{}};
  }
  if (request.embeddings.size != 0U && request.embeddings.data == nullptr) {
    return universal::result_from_status<AllVsAllResult>(
        universal::invalid_argument_status(
            "pair-list embedding input span is invalid"));
  }
  // `EmbeddingView` has no dedicated identifier field; like the coords
  // route the embedding route resolves names through the per-residue
  // `source_id` metadata. An embedding with no residue metadata is
  // unaddressable by name.
  std::vector<std::string_view> source_ids;
  try {
    source_ids.reserve(request.embeddings.size);
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<AllVsAllResult>(
        universal::unavailable_status(
            "pair-list embedding id-table allocation failed"));
  }
  for (std::size_t index = 0; index < request.embeddings.size; ++index) {
    const universal::EmbeddingView& embedding =
        request.embeddings.data[index];
    source_ids.push_back(embedding.residues.size != 0U
                             ? embedding.residues.data[0].source_id
                             : std::string_view{});
  }

  std::vector<std::size_t> original_index;
  std::vector<std::pair<std::size_t, std::size_t>> compacted_pairs;
  universal::Status status = resolve_pair_list(
      {source_ids.data(), source_ids.size()}, request.pairs, original_index,
      compacted_pairs);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<AllVsAllResult>(status);
  }

  std::vector<universal::EmbeddingView> compacted;
  try {
    compacted.reserve(original_index.size());
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<AllVsAllResult>(
        universal::unavailable_status(
            "pair-list embedding compaction allocation failed"));
  }
  for (const std::size_t original : original_index) {
    const universal::EmbeddingView& view = request.embeddings.data[original];
    compacted.push_back(view);
  }
  const PairListAxisMaxima axis_maxima = pair_list_axis_maxima(
      compacted_pairs,
      [&](std::size_t item_index) noexcept {
        return compacted[item_index].residue_count;
      });
  std::size_t max_result_step_count = 0;
  status = pair_traceback_step_count(axis_maxima.max_query_length,
                                     axis_maxima.max_target_length,
                                     max_result_step_count);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<AllVsAllResult>(status);
  }

  algorithms::PairListEmbeddingRequest algorithm_request{};
  algorithm_request.embeddings = {compacted.data(), compacted.size()};
  algorithm_request.pairs = {compacted_pairs.data(), compacted_pairs.size()};
  algorithm_request.max_query_length = axis_maxima.max_query_length;
  algorithm_request.max_target_length = axis_maxima.max_target_length;
  const ResolvedAlignmentOptions resolved_alignment =
      resolve_alignment_against_mpnn64(request.options.alignment);
  const GapOverrideContext gap_override_ctx =
      mpnn64_gap_override_context(request.options.mode);
  algorithm_request.options =
      to_algorithms_options(request.options, resolved_alignment);

  return finish_pair_list(
      original_index, request.pairs.size(), max_result_step_count,
      request.options.alignment, gap_override_ctx,
      [&](algorithms::PairwiseResultSink& sink) {
        return algorithms::run_pair_list_embeddings(algorithm_request, sink);
      });
}

universal::Result<AllVsAllResult> Engine::collect_pair_list(
    const PairListSequenceRequest& request) const {
  const universal::Status axes = validate_engine_axes(config_);
  if (!universal::is_ok(axes)) {
    return universal::result_from_status<AllVsAllResult>(axes);
  }
  const universal::Status mode = validate_all_vs_all_alignment_mode(
      request.options.mode, request.options.temperature);
  if (!universal::is_ok(mode)) {
    return universal::result_from_status<AllVsAllResult>(mode);
  }
  if (request.pairs.empty()) {
    return {universal::ok_status(), AllVsAllResult{}};
  }
  if (request.sequences.size != 0U && request.sequences.data == nullptr) {
    return universal::result_from_status<AllVsAllResult>(
        universal::invalid_argument_status(
            "pair-list sequence input span is invalid"));
  }

  universal::Result<const universal::WeightsView*> view =
      resolve_sequence_package_weights_view(config_);
  if (!universal::is_ok(view.status)) {
    return universal::result_from_status<AllVsAllResult>(view.status);
  }

  // Sequence inputs carry their identifier in `SequenceEntry::name`.
  std::vector<std::string_view> source_ids;
  try {
    source_ids.reserve(request.sequences.size);
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<AllVsAllResult>(
        universal::unavailable_status(
            "pair-list sequence id-table allocation failed"));
  }
  for (std::size_t index = 0; index < request.sequences.size; ++index) {
    source_ids.push_back(request.sequences.data[index].name);
  }

  std::vector<std::size_t> original_index;
  std::vector<std::pair<std::size_t, std::size_t>> compacted_pairs;
  universal::Status status = resolve_pair_list(
      {source_ids.data(), source_ids.size()}, request.pairs, original_index,
      compacted_pairs);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<AllVsAllResult>(status);
  }

  std::vector<algorithms::AllVsAllSequenceEntry> compacted;
  std::size_t max_token_count = 0;
  try {
    compacted.reserve(original_index.size());
  } catch (const std::bad_alloc&) {
    return universal::result_from_status<AllVsAllResult>(
        universal::unavailable_status(
            "pair-list sequence compaction allocation failed"));
  }
  for (const std::size_t original : original_index) {
    const SequenceEntry& entry = request.sequences.data[original];
    algorithms::AllVsAllSequenceEntry algorithm_entry{};
    algorithm_entry.name = entry.name;
    algorithm_entry.token_ids = entry.token_ids;
    compacted.push_back(algorithm_entry);
    max_token_count = std::max(max_token_count, entry.token_ids.size);
  }
  const PairListAxisMaxima axis_maxima = pair_list_axis_maxima(
      compacted_pairs,
      [&](std::size_t item_index) noexcept {
        return compacted[item_index].token_ids.size;
      });
  std::size_t max_result_step_count = 0;
  status = pair_traceback_step_count(axis_maxima.max_query_length,
                                     axis_maxima.max_target_length,
                                     max_result_step_count);
  if (!universal::is_ok(status)) {
    return universal::result_from_status<AllVsAllResult>(status);
  }

  // Resolve the package-default gap sentinel against the ESM2 descriptor,
  // mirroring `Engine::all_vs_all(AllVsAllSequenceRequest)`. The sink keeps
  // the caller's unresolved alignment so the gap-override warning still
  // compares the caller's values against the package defaults.
  const universal::PackageDescriptor& sequence_descriptor =
      *config_.package.descriptor;
  const ResolvedAlignmentOptions resolved_alignment =
      resolve_alignment_against_package(request.options.alignment,
                                        sequence_descriptor);
  const GapOverrideContext gap_override_ctx =
      gap_override_context_for_descriptor(sequence_descriptor,
                                          request.options.mode);

  algorithms::PairListSequenceRequest algorithm_request{};
  algorithm_request.sequences = {compacted.data(), compacted.size()};
  algorithm_request.descriptor = esm2_8m_descriptor();
  algorithm_request.weights_view = view.value;
  algorithm_request.pairs = {compacted_pairs.data(), compacted_pairs.size()};
  algorithm_request.max_query_length = axis_maxima.max_query_length;
  algorithm_request.max_target_length = axis_maxima.max_target_length;
  algorithm_request.options =
      to_algorithms_options(request.options, resolved_alignment);

  // Acquire the shared all-vs-all pool so the encode-once pass and eligible
  // pair-list dispatch parallelize under `--threads`. `encoding_item_count`
  // is the deduped unique-protein count, sized from the actual max_token_count.
  // The lease (and its lock) must outlive `finish_pair_list`, which runs the
  // lambda synchronously.
  AllVsAllThreadingPlan threading_plan{};
  threading_plan.pair_count = compacted_pairs.size();
  threading_plan.encoding_item_count = compacted.size();
  threading_plan.encoder_workspace_bytes_per_thread =
      sequence_phase1_workspace_bytes_per_thread(max_token_count);
  AllVsAllThreadingLease threading = acquire_all_vs_all_threading(
      static_cast<EngineThreadingState*>(threading_.get()),
      config_.execution.thread_count,
      threading_plan);
  return finish_pair_list(
      original_index, request.pairs.size(), max_result_step_count,
      request.options.alignment, gap_override_ctx,
      [&](algorithms::PairwiseResultSink& sink) {
        return algorithms::run_pair_list_sequences(algorithm_request, sink,
                                                   threading.pool,
                                                   threading.thread_count,
                                                   threading.workspaces);
      });
}

VersionInfo Engine::version_info() const noexcept {
  return hikoboshi::api::version_info();
}

BackendCapabilities Engine::backend_capabilities() const noexcept {
  return hikoboshi::api::backend_capabilities();
}

namespace {

// Hikoboshi 0.1.0 CLI compatibility: render `raw_sw_score` (a double) and metric
// values with six significant digits to match `render_all_vs_all_summary` and
// keep the streaming TSV path bit-identical to the legacy renderer for parity
// `diff` checks.
constexpr int kSummaryDoublePrecision = 6;
constexpr int kSummaryMetricPrecision = 6;
// Mirrors `hikoboshi::errors::kMetricNotAvailable`. Inlined here because the
// public `api` layer is not permitted to depend on `errors` per the
// dependency graph.
constexpr const char* kMetricNotAvailable = "NA";

std::string format_summary_double(double value) {
  std::ostringstream out;
  out << std::setprecision(kSummaryDoublePrecision) << value;
  return out.str();
}

std::string format_summary_metric(universal::MetricValue metric) {
  if (!metric.valid) {
    return kMetricNotAvailable;
  }
  int significant_digits = kSummaryMetricPrecision;
  if (significant_digits < 1) {
    significant_digits = 1;
  }
  if (significant_digits > std::numeric_limits<double>::max_digits10) {
    significant_digits = std::numeric_limits<double>::max_digits10;
  }
  std::ostringstream out;
  out << std::setprecision(significant_digits) << metric.value;
  return out.str();
}

universal::MetricValue summary_invalid_metric(
    universal::MetricInvalidReason reason) noexcept {
  return {0.0, false, reason};
}

universal::MetricValue summary_valid_metric(double value) noexcept {
  return {value, true, universal::MetricInvalidReason::None};
}

universal::MetricValue summary_sw_per_aligned(
    const PairwiseResult& result) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return summary_invalid_metric(universal::MetricInvalidReason::
                                      ZeroDenominator);
  }
  return summary_valid_metric(
      result.metrics.raw_sw_score /
      static_cast<double>(result.path.aligned_pairs));
}

universal::MetricValue summary_sw_per_length(
    const PairwiseResult& result,
    universal::MetricValue coverage) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return summary_invalid_metric(universal::MetricInvalidReason::
                                      ZeroDenominator);
  }
  if (!coverage.valid) {
    return summary_invalid_metric(coverage.reason);
  }
  return summary_valid_metric(
      (result.metrics.raw_sw_score * coverage.value) /
      static_cast<double>(result.path.aligned_pairs));
}

void write_summary_callback_column(std::ostream& out,
                                   std::string (*callback)(std::size_t,
                                                           std::size_t,
                                                           void*),
                                   std::size_t query_index,
                                   std::size_t target_index,
                                   void* user_data) {
  if (callback != nullptr) {
    out << callback(query_index, target_index, user_data);
  }
}

void write_summary_record(std::ostream& out,
                          const PairwiseResultRecord& record,
                          const TsvStreamingAllVsAllSink::Callbacks& cb) {
  const PairwiseMetrics& metrics = record.result.metrics;
  out << record.query_index << '\t' << record.target_index << '\t';
  write_summary_callback_column(out, cb.pair_id, record.query_index,
                                record.target_index, cb.user_data);
  out << '\t' << format_summary_double(metrics.raw_sw_score);
  if (cb.include_dual_score_schema) {
    out << '\t' << format_summary_metric(metrics.soft_sw_score) << '\t'
        << format_summary_metric(
               summary_sw_per_length(record.result, metrics.coverage_query))
        << '\t'
        << format_summary_metric(
               summary_sw_per_length(record.result, metrics.coverage_target))
        << '\t' << format_summary_metric(summary_sw_per_aligned(record.result));
  }
  out << '\t'
      << record.result.path.aligned_pairs << '\t'
      << format_summary_metric(metrics.coverage_query) << '\t'
      << format_summary_metric(metrics.coverage_target) << '\t'
      << format_summary_metric(metrics.coverage_mean) << '\t'
      << format_summary_metric(metrics.identity) << '\t'
      << format_summary_metric(metrics.rmsd) << '\t'
      << format_summary_metric(metrics.tm_score_query) << '\t'
      << format_summary_metric(metrics.tm_score_target) << '\t'
      << format_summary_metric(metrics.lddt) << '\t'
      << format_summary_metric(metrics.lddt_byA) << '\t'
      << format_summary_metric(metrics.lddt_byB) << '\t'
      << format_summary_metric(metrics.lddt_aln) << '\t'
      << format_summary_metric(metrics.coverage_byA) << '\t'
      << format_summary_metric(metrics.coverage_byB) << '\t'
      << format_summary_metric(metrics.ecs) << '\t';
  write_summary_callback_column(out, cb.fasta_path, record.query_index,
                                record.target_index, cb.user_data);
  out << '\t';
  write_summary_callback_column(out, cb.pdb_path, record.query_index,
                                record.target_index, cb.user_data);
  out << '\n';
}

}  // namespace

void TsvStreamingAllVsAllSink::write_header(std::ostream& out) {
  out << "query_index\ttarget_index\tpair_id\traw_sw_score\taligned_pairs"
      << "\tcoverage_query\tcoverage_target\tcoverage_mean\tidentity\trmsd"
      << "\ttm_score_query\ttm_score_target"
      << "\tlddt\tlddt_byA\tlddt_byB\tlddt_aln\tcoverage_byA\tcoverage_byB"
      << "\tecs\tfasta_path\tpdb_path\n";
}

void write_tsv_streaming_header(std::ostream& out,
                                bool include_dual_score_schema) {
  if (!include_dual_score_schema) {
    TsvStreamingAllVsAllSink::write_header(out);
    return;
  }
  out << "query_index\ttarget_index\tpair_id\traw_sw_score";
  if (include_dual_score_schema) {
    out << "\tsoft_sw_score"
        << "\tsw_per_query_len\tsw_per_target_len\tsw_per_aligned";
  }
  out << "\taligned_pairs"
      << "\tcoverage_query\tcoverage_target\tcoverage_mean\tidentity\trmsd"
      << "\ttm_score_query\ttm_score_target"
      << "\tlddt\tlddt_byA\tlddt_byB\tlddt_aln\tcoverage_byA\tcoverage_byB"
      << "\tecs\tfasta_path\tpdb_path\n";
}

TsvStreamingAllVsAllSink::TsvStreamingAllVsAllSink(
    std::vector<std::ostream*> outputs,
    Callbacks callbacks)
    : outputs_(std::move(outputs)), callbacks_(callbacks) {
  for (std::ostream* out : outputs_) {
    if (out != nullptr) {
      write_tsv_streaming_header(*out, callbacks_.include_dual_score_schema);
    }
  }
}

TsvStreamingAllVsAllSink::TsvStreamingAllVsAllSink(std::ostream& output,
                                                   Callbacks callbacks)
    : TsvStreamingAllVsAllSink(std::vector<std::ostream*>{&output}, callbacks) {}

universal::Status TsvStreamingAllVsAllSink::receive(
    const PairwiseResultRecord& record) {
  for (std::ostream* out : outputs_) {
    if (out == nullptr) {
      continue;
    }
    write_summary_record(*out, record, callbacks_);
    if (!out->good()) {
      return {universal::StatusCode::Unavailable,
              "all-vs-all summary write failed"};
    }
  }
  ++emitted_;
  return universal::ok_status();
}

universal::Status stream_all_vs_all(const Engine& engine,
                                    const AllVsAllStructureRequest& request,
                                    PairwiseResultSink& sink) {
  return engine.all_vs_all(request, sink);
}

universal::Status stream_all_vs_all(const Engine& engine,
                                    const AllVsAllCoordsRequest& request,
                                    PairwiseResultSink& sink) {
  return engine.all_vs_all(request, sink);
}

universal::Status stream_all_vs_all(const Engine& engine,
                                    const AllVsAllEmbeddingRequest& request,
                                    PairwiseResultSink& sink) {
  return engine.all_vs_all(request, sink);
}

universal::Status stream_all_vs_all(const Engine& engine,
                                    const AllVsAllSequenceRequest& request,
                                    PairwiseResultSink& sink) {
  return engine.all_vs_all(request, sink);
}

}  // namespace hikoboshi::api

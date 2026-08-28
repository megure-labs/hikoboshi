#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/backend.hpp>
#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/execution_options.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>
#include <hikoboshi/universal/tensor.hpp>
#include <hikoboshi/universal/types.hpp>
#include <hikoboshi/universal/version.hpp>
#include <hikoboshi/universal/weights.hpp>

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace hiko_u = hikoboshi::universal;

static_assert(std::is_standard_layout<hiko_u::Span<const float>>::value,
              "Span must stay a plain view");
static_assert(std::is_trivially_copyable<hiko_u::Span<const float>>::value,
              "Span must stay trivially copyable");
static_assert(std::is_standard_layout<hiko_u::Status>::value,
              "Status must stay a plain status record");
static_assert(std::is_trivially_copyable<hiko_u::MetricValue>::value,
              "MetricValue must stay trivially copyable");
static_assert(std::is_standard_layout<hiko_u::PackageDescriptor>::value,
              "PackageDescriptor must stay a plain descriptor record");
static_assert(std::is_trivially_copyable<hiko_u::PackageHandle>::value,
              "PackageHandle must stay a plain handle");
static_assert(std::is_trivially_copyable<hiko_u::ScoreSemantics>::value,
              "ScoreSemantics must stay trivially copyable");
static_assert(hiko_u::kVersionMajor == 0, "version major is part of the public contract");
static_assert(hiko_u::kVersionMinor == 1, "version minor is part of the public contract");
static_assert(hiko_u::kVersionPatch == 0, "version patch is part of the public contract");
static_assert(hiko_u::kAlignmentGapSentinel == -1,
              "hard-SW path uses -1 as the public gap sentinel");
static_assert(hiko_u::kCanonicalAtomCount == 5, "structure views use N, CA, C, O, CB");
static_assert(hiko_u::kCoordinateAxisCount == 3, "structure coordinates are xyz triples");
static_assert(static_cast<unsigned>(hiko_u::Backend::Auto) == 0,
              "Backend::Auto enum value is part of the public contract");
static_assert(static_cast<unsigned>(hiko_u::Backend::Scalar) == 1,
              "Backend::Scalar enum value is part of the public contract");
static_assert(static_cast<unsigned>(hiko_u::Backend::Sse4) == 2,
              "Backend::Sse4 reserves the sse4 user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Avx2) == 3,
              "Backend::Avx2 reserves the avx2 user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Avx512) == 4,
              "Backend::Avx512 reserves the avx512 user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Neon) == 5,
              "Backend::Neon reserves the neon user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Sve) == 6,
              "Backend::Sve reserves the sve user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Cuda) == 7,
              "Backend::Cuda reserves the cuda user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Hip) == 8,
              "Backend::Hip reserves the hip user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Metal) == 9,
              "Backend::Metal reserves the metal user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::Vulkan) == 10,
              "Backend::Vulkan reserves the vulkan user-facing backend name");
static_assert(static_cast<unsigned>(hiko_u::Backend::OpenCl) == 11,
              "Backend::OpenCl reserves the opencl user-facing backend name");

int main() {
  constexpr std::size_t residue_count = 2;
  constexpr std::size_t embedding_dimension = 64;
  const float coordinates[residue_count * hiko_u::kCanonicalAtomCount *
                          hiko_u::kCoordinateAxisCount] = {};
  const hiko_u::AtomSource atom_sources[residue_count * hiko_u::kCanonicalAtomCount] = {};
  const char residue_codes[residue_count] = {'A', 'G'};
  const hiko_u::ResidueMetadataView residues[residue_count] = {
      {'A', "ALA", "A", "1", 1, 1, '\0', "query", 0, "query.pdb", 10},
      {'G', "GLY", "A", "1", 1, 2, '\0', "query", 1, "query.pdb", 20},
  };
  const hiko_u::ChainBreakView chain_breaks[1] = {{0}};

  const hiko_u::StructureView structure{
      residue_count,
      {coordinates, residue_count * hiko_u::kCanonicalAtomCount *
                        hiko_u::kCoordinateAxisCount},
      {atom_sources, residue_count * hiko_u::kCanonicalAtomCount},
      {residue_codes, residue_count},
      {residues, residue_count},
      "query",
      "query.pdb",
      {chain_breaks, 1},
  };

  hiko_u::AlignmentPath path{};
  path.steps.push_back({0, 0, 1.0F});
  path.steps.push_back({1, hiko_u::kAlignmentGapSentinel, 0.0F});

  const float embedding_values[residue_count * embedding_dimension] = {};
  const hiko_u::EmbeddingView embedding{
      residue_count,
      embedding_dimension,
      {embedding_values, residue_count * embedding_dimension},
      {residue_codes, residue_count},
      {residues, residue_count},
  };

  const std::size_t tensor_shape[2] = {residue_count, embedding_dimension};
  const hiko_u::TensorView tensor{
      embedding_values,
      {tensor_shape, 2},
      {nullptr, 0},
      hiko_u::DataType::Float32,
      "embedding",
  };

  const hiko_u::ModelMetadataView metadata{
      "Hikoboshi-MPNN-64",
      "mpnn",
      "0.1.0",
      embedding_dimension,
      "test-fixture",
      "checksum-placeholder",
  };
  const hiko_u::TensorView tensors[1] = {tensor};
  const hiko_u::WeightsView weights{metadata, {tensors, 1}};
  const hiko_u::WeightsHandle handle{nullptr, &weights};

  const hiko_u::PackageInputKind input_routes[3] = {
      hiko_u::PackageInputKind::StructureBackboneAtoms,
      hiko_u::PackageInputKind::CoordsBackbone,
      hiko_u::PackageInputKind::ResidueEmbeddings,
  };
  const hiko_u::PackagePreprocessingCapability preprocessing[6] = {
      hiko_u::PackagePreprocessingCapability::AtomInference,
      hiko_u::PackagePreprocessingCapability::VirtualCb,
      hiko_u::PackagePreprocessingCapability::CaKnn,
      hiko_u::PackagePreprocessingCapability::AtomPairDistances,
      hiko_u::PackagePreprocessingCapability::RbfExpand,
      hiko_u::PackagePreprocessingCapability::PositionalEncoding,
  };
  const hiko_u::PackageOutputKind output_kinds[1] = {
      hiko_u::PackageOutputKind::ResidueEmbeddings,
  };
  const hiko_u::DataType package_dtypes[1] = {hiko_u::DataType::Float32};
  const hiko_u::PackageTensorLayout package_layouts[1] = {
      hiko_u::PackageTensorLayout::RowMajor,
  };
  const hiko_u::PackageBackendRequirement package_backends[1] = {
      hiko_u::PackageBackendRequirement::CpuScalar,
  };
  const std::string_view package_aliases[2] = {"mpnn64", "mpnn-64"};
  const hiko_u::ScoreInputKind score_inputs[1] = {
      hiko_u::ScoreInputKind::ResidueEmbeddings,
  };
  const hiko_u::TracebackPolicy traceback_policies[2] = {
      hiko_u::TracebackPolicy::RequiredForPublicPairwise,
      hiko_u::TracebackPolicy::RequiredForPublicAllVsAll,
  };
  const hiko_u::PackageCapabilityFlags package_flags =
      static_cast<hiko_u::PackageCapabilityFlags>(
          hiko_u::PackageCapabilityFlag::StructureBackboneAtoms) |
      static_cast<hiko_u::PackageCapabilityFlags>(
          hiko_u::PackageCapabilityFlag::ResidueEmbeddings) |
      static_cast<hiko_u::PackageCapabilityFlags>(
          hiko_u::PackageCapabilityFlag::OutputResidueEmbeddings) |
      static_cast<hiko_u::PackageCapabilityFlags>(
          hiko_u::PackageCapabilityFlag::BackendCpuScalar);

  const hiko_u::PackageDescriptor package_descriptor{
      {"0.1.0", "Hikoboshi-MPNN-64", "Hikoboshi-MPNN", "mpnn64-compiled-v1",
       hiko_u::PackageKind::RegisteredArchitecture, {package_aliases, 2}},
      {hiko_u::PackageExecutionMode::RegisteredArchitecture, "hikoboshi_mpnn_v1",
       {package_backends, 1}},
      {package_flags,
       {input_routes, 3},
       {preprocessing, 6},
       {output_kinds, 1},
       {package_dtypes, 1},
       {package_layouts, 1},
       {package_backends, 1}},
      {{input_routes, 3}},
      {{output_kinds, 1}},
      {hiko_u::ScoreMethod::RawDotV1,
       {score_inputs, 1},
       hiko_u::ScoreOutputKind::ScoreMatrix,
       {hiko_u::DataType::Float32, hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget,
        true, true, hiko_u::ScoreNormalization::None,
        hiko_u::ScoreScaleFamily::RawDot}},
        {"Hikoboshi 0.1.0 hard-SW",
         hiko_u::GapModel::Affine,
         -1.4F,
         -0.15F,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         hiko_u::ScoreMethod::RawDotV1},
        {"Hikoboshi 0.1.0 soft-SW",
         hiko_u::GapModel::Affine,
         -3.21337F,
         -0.111704F,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         hiko_u::ScoreMethod::RawDotV1},
        {hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1,
       {traceback_policies, 2}},
      {handle},
  };
  const hiko_u::PackageHandle package{nullptr, &package_descriptor};
  const hiko_u::PackageWarning warnings[1] = {
      {hiko_u::PackageWarningKind::GapDefaultsOverridden,
       hiko_u::PackageValidationStage::GapModelDefaults,
       "gap_defaults_overridden",
       "gap defaults were overridden"},
  };
  const hiko_u::PackageValidationDiagnostic diagnostics[1] = {
      {hiko_u::PackageDiagnosticSeverity::Warning,
       hiko_u::PackageValidationStage::GapModelDefaults,
       "gap_defaults_overridden",
       "gap defaults were overridden"},
  };
  const hiko_u::PackageValidationReport validation_report{
      package,
      {diagnostics, 1},
      {warnings, 1},
      1ull << static_cast<unsigned>(hiko_u::PackageValidationStage::SchemaVersion),
      0,
      true,
  };

  const hiko_u::ExecutionOptions execution{hiko_u::Backend::Auto, 0};
  const hiko_u::MetricValue metric{0.0, false,
                                hiko_u::MetricInvalidReason::InsufficientAlignedPairs};
  const hiko_u::Result<hiko_u::MetricValue> result{
      {hiko_u::StatusCode::Ok, nullptr},
      metric,
  };
  const hiko_u::VersionView version = hiko_u::kVersion;

  return structure.residue_count == residue_count &&
                 structure.chain_breaks.size == 1 &&
                 structure.residues.data[0].source_record_index == 10 &&
                 path.steps[1].target_index == hiko_u::kAlignmentGapSentinel &&
                 embedding.dimension == embedding_dimension &&
                 handle.view == &weights &&
                 package.descriptor == &package_descriptor &&
                 package.descriptor->identity.package_id == "Hikoboshi-MPNN-64" &&
                 package.descriptor->identity.aliases.size == 2 &&
                 package.descriptor->identity.aliases.data[1] == "mpnn-64" &&
                 package.descriptor->execution.mode ==
                     hiko_u::PackageExecutionMode::RegisteredArchitecture &&
                   package.descriptor->scoring.semantics.scale_family ==
                       hiko_u::ScoreScaleFamily::RawDot &&
                   package.descriptor->gaps.gap_open == -1.4F &&
                   package.descriptor->soft_gaps.gap_open == -3.21337F &&
                   package.descriptor->compatibility_views.weights.view == &weights &&
                 validation_report.ok && validation_report.warnings.size == 1 &&
                 execution.backend == hiko_u::Backend::Auto &&
                 result.status.code == hiko_u::StatusCode::Ok &&
                 !result.value.valid &&
                 version.minor == 1
             ? 0
             : 1;
}

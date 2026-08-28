#include <hikoboshi/weights/provider.hpp>
#include <hikoboshi/weights/provider_register.hpp>

#include "embedded_esm2_8m.hpp"
#include "embedded_mpnn_d64.hpp"
#include "embedded_proteinmpnn_v48_eps020.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <hikoboshi/dispatch/registry/alignment.hpp>
#include <hikoboshi/dispatch/registry/architecture.hpp>
#include <hikoboshi/dispatch/registry/scoring.hpp>
#include <hikoboshi/dispatch/registry/validation_core.hpp>
#include <hikoboshi/universal/package_validation_codes.hpp>

namespace hikoboshi::weights {
namespace {

namespace hiko_u = hikoboshi::universal;
namespace hiko_dr = hikoboshi::dispatch::registry;

constexpr std::string_view kDefaultMpnnD64Aliases[] = {
    "mpnn64",
    "mpnn-64",
    "Hikoboshi-MPNN-64",
};

constexpr std::string_view kDefaultEsm2_8mAliases[] = {
    "esm2-8m",
    "esm2_8m",
    "Hikoboshi-ESM2-8M",
};

constexpr std::string_view kDefaultProteinMpnnV48Eps020Aliases[] = {
    "v_48_020",
    "Hikoboshi-ProteinMPNN-v48-020",
    "proteinmpnn-v48-020",
    "proteinmpnn",
};

constexpr char kUnknownPackageDiagnostic[] =
    "unknown Hikoboshi package id; available compiled package IDs/aliases: "
    "hikoboshi-mpnn-d64, mpnn64, mpnn-64, Hikoboshi-MPNN-64, "
    "hikoboshi-esm2-8m, esm2-8m, esm2_8m, Hikoboshi-ESM2-8M, "
    "proteinmpnn-v48-eps020, v_48_020, Hikoboshi-ProteinMPNN-v48-020, "
    "proteinmpnn-v48-020, proteinmpnn";

// MPNN-64-specific input-route/preprocessing helpers (see
// `validate_mpnn64_package`) reject ESM2-style sequence inputs by design.
// ESM2-8M validation reuses the canonical 10-stage pipeline plus its own
// package-specific helpers below, so we no longer carry a
// "not runtime-available" sentinel string.

constexpr hiko_u::TracebackPolicy kTracebackPolicies[] = {
    hiko_u::TracebackPolicy::RequiredForPublicPairwise,
    hiko_u::TracebackPolicy::RequiredForPublicAllVsAll,
};

constexpr hiko_u::PackageInputKind kProteinMpnnV48020InputRoutes[] = {
    hiko_u::PackageInputKind::StructureBackboneAtoms,
    hiko_u::PackageInputKind::CoordsBackbone,
};

constexpr hiko_u::PackagePreprocessingCapability
    kProteinMpnnV48020Preprocessing[] = {
        hiko_u::PackagePreprocessingCapability::AtomInference,
        hiko_u::PackagePreprocessingCapability::VirtualCb,
        hiko_u::PackagePreprocessingCapability::CaKnn,
        hiko_u::PackagePreprocessingCapability::AtomPairDistances,
        hiko_u::PackagePreprocessingCapability::RbfExpand,
        hiko_u::PackagePreprocessingCapability::PositionalEncoding,
};

constexpr hiko_u::PackageOutputKind kProteinMpnnV48020OutputKinds[] = {
    hiko_u::PackageOutputKind::DirectPairScores,
};

constexpr hiko_u::DataType kProteinMpnnV48020Dtypes[] = {
    hiko_u::DataType::Float32,
};

constexpr hiko_u::PackageTensorLayout kProteinMpnnV48020Layouts[] = {
    hiko_u::PackageTensorLayout::RowMajor,
};

constexpr hiko_u::PackageBackendRequirement kProteinMpnnV48020Backends[] = {
    hiko_u::PackageBackendRequirement::CpuScalar,
};

constexpr hiko_u::ScoreInputKind kProteinMpnnV48020ScoreInputs[] = {
    hiko_u::ScoreInputKind::DirectPairInput,
};

constexpr hiko_u::PackageCapabilityFlags kProteinMpnnV48020CapabilityFlags =
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::StructureBackboneAtoms) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::CoordsBackbone) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::AtomInference) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::VirtualCb) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::CaKnn) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::AtomPairDistances) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::RbfExpand) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::PositionalEncoding) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::BackendCpuScalar);

constexpr std::uint64_t stage_flag(
    const hiko_u::PackageValidationStage stage) noexcept {
  return 1ull << static_cast<unsigned>(stage);
}

char ascii_lower(const char value) noexcept {
  if (value >= 'A' && value <= 'Z') {
    return static_cast<char>(value - 'A' + 'a');
  }
  return value;
}

bool ascii_case_equal(const std::string_view lhs,
                      const std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (ascii_lower(lhs[index]) != ascii_lower(rhs[index])) {
      return false;
    }
  }
  return true;
}

bool matches_default_mpnn64_id(const std::string_view package_id) noexcept {
  if (package_id == kDefaultMpnnD64ModelName) {
    return true;
  }
  for (const std::string_view alias : kDefaultMpnnD64Aliases) {
    if (ascii_case_equal(package_id, alias)) {
      return true;
    }
  }
  return false;
}

bool matches_default_esm2_8m_id(const std::string_view package_id) noexcept {
  if (package_id == kDefaultEsm2_8mModelName) {
    return true;
  }
  for (const std::string_view alias : kDefaultEsm2_8mAliases) {
    if (ascii_case_equal(package_id, alias)) {
      return true;
    }
  }
  return false;
}

bool matches_proteinmpnn_v48_eps020_id(
    const std::string_view package_id) noexcept {
  if (package_id == kDefaultProteinMpnnV48Eps020ModelName) {
    return true;
  }
  for (const std::string_view alias : kDefaultProteinMpnnV48Eps020Aliases) {
    if (ascii_case_equal(package_id, alias)) {
      return true;
    }
  }
  return false;
}

// Per-MPNN-64-package input-route check used by validate_mpnn64_package.
// Other architectures (e.g. hikoboshi-esm2-8m) accept additional input
// kinds, but those are validated through the canonical staged pipeline's
// io_contract_binding stage rather than this MPNN-64-specific helper.
bool supported_input(const hiko_u::PackageInputKind input) noexcept {
  switch (input) {
    case hiko_u::PackageInputKind::StructureBackboneAtoms:
    case hiko_u::PackageInputKind::CoordsBackbone:
    case hiko_u::PackageInputKind::ResidueEmbeddings:
      return true;
    case hiko_u::PackageInputKind::StructureAllAtom:
    case hiko_u::PackageInputKind::SequenceTokens:
    case hiko_u::PackageInputKind::DirectScoreMatrix:
      return false;
  }
  return false;
}

// Per-MPNN-64-package preprocessing check used by
// validate_mpnn64_package. Tokenization is reserved for the ESM2-8M
// (and other PLM) architecture path; the canonical staged pipeline lets
// it through for descriptors that declare a PLM architecture id.
bool supported_preprocessing(
    const hiko_u::PackagePreprocessingCapability capability) noexcept {
  switch (capability) {
    case hiko_u::PackagePreprocessingCapability::AtomInference:
    case hiko_u::PackagePreprocessingCapability::VirtualCb:
    case hiko_u::PackagePreprocessingCapability::CaKnn:
    case hiko_u::PackagePreprocessingCapability::AtomPairDistances:
    case hiko_u::PackagePreprocessingCapability::RbfExpand:
    case hiko_u::PackagePreprocessingCapability::PositionalEncoding:
      return true;
    case hiko_u::PackagePreprocessingCapability::Tokenization:
      return false;
  }
  return false;
}

bool supported_output(const hiko_u::PackageOutputKind output) noexcept {
  switch (output) {
    case hiko_u::PackageOutputKind::ResidueEmbeddings:
      return true;
    case hiko_u::PackageOutputKind::SubstitutionScores:
    case hiko_u::PackageOutputKind::DirectPairScores:
      return false;
  }
  return false;
}

bool supported_backend(
    const hiko_u::PackageBackendRequirement backend) noexcept {
  switch (backend) {
    case hiko_u::PackageBackendRequirement::CpuScalar:
      return true;
    case hiko_u::PackageBackendRequirement::GpuCuda:
    case hiko_u::PackageBackendRequirement::GpuMetal:
    case hiko_u::PackageBackendRequirement::GpuHip:
      return false;
  }
  return false;
}

void add_error(PackageValidationBuffer& buffer,
               std::size_t& diagnostic_count,
               const hiko_u::PackageValidationStage stage,
               const std::string_view code,
               const std::string_view message) noexcept {
  if (diagnostic_count < kPackageValidationDiagnosticCapacity) {
    buffer.diagnostics[diagnostic_count++] = {
        hiko_u::PackageDiagnosticSeverity::Error,
        stage,
        code,
        message,
    };
  }
}

void mark_stage(bool stage_ok,
                const hiko_u::PackageValidationStage stage,
                std::uint64_t& passed_stage_flags,
                std::uint64_t& failed_stage_flags) noexcept {
  if (stage_ok) {
    passed_stage_flags |= stage_flag(stage);
  } else {
    failed_stage_flags |= stage_flag(stage);
  }
}

bool validate_input_span(const hiko_u::Span<const hiko_u::PackageInputKind>& routes,
                         PackageValidationBuffer& buffer,
                         std::size_t& diagnostic_count) noexcept {
  bool ok = routes.size != 0;
  if (routes.size == 0) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::InputRoute,
              "missing_input_route",
              "Hikoboshi 0.1 package validation requires at least one input route");
  }
  for (std::size_t index = 0; index < routes.size; ++index) {
    if (!supported_input(routes.data[index])) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::InputRoute,
                "unsupported_input_modality",
                "Hikoboshi 0.1 supports structure_backbone_atoms, coords_backbone, and residue_embeddings inputs only");
    }
  }
  return ok;
}

bool validate_preprocessing_span(
    const hiko_u::Span<const hiko_u::PackagePreprocessingCapability>& capabilities,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = true;
  for (std::size_t index = 0; index < capabilities.size; ++index) {
    if (!supported_preprocessing(capabilities.data[index])) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::PreprocessingCapabilities,
                "unsupported_preprocessing_capability",
                "Hikoboshi 0.1 supports only the compiled hikoboshi-mpnn-d64 preprocessing capabilities");
    }
  }
  return ok;
}

bool validate_output_span(
    const hiko_u::Span<const hiko_u::PackageOutputKind>& outputs,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = outputs.size != 0;
  if (outputs.size == 0) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::WorkflowCompatibility,
              "missing_output_kind",
              "Hikoboshi 0.1 package validation requires residue_embeddings output");
  }
  for (std::size_t index = 0; index < outputs.size; ++index) {
    if (!supported_output(outputs.data[index])) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::WorkflowCompatibility,
                "unsupported_output_kind",
                "Hikoboshi 0.1 supports residue_embeddings package output only");
    }
  }
  return ok;
}

bool validate_dtype_span(const hiko_u::Span<const hiko_u::DataType>& dtypes,
                         PackageValidationBuffer& buffer,
                         std::size_t& diagnostic_count) noexcept {
  bool ok = dtypes.size != 0;
  if (dtypes.size == 0) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
              "missing_dtype",
              "Hikoboshi 0.1 package validation requires float32 tensor dtype");
  }
  for (std::size_t index = 0; index < dtypes.size; ++index) {
    if (dtypes.data[index] != hiko_u::DataType::Float32) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "unsupported_dtype",
                "Hikoboshi 0.1 supports float32 tensor dtype only");
    }
  }
  return ok;
}

bool validate_layout_span(
    const hiko_u::Span<const hiko_u::PackageTensorLayout>& layouts,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = layouts.size != 0;
  if (layouts.size == 0) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
              "missing_tensor_layout",
              "Hikoboshi 0.1 package validation requires row_major tensor layout");
  }
  for (std::size_t index = 0; index < layouts.size; ++index) {
    if (layouts.data[index] != hiko_u::PackageTensorLayout::RowMajor) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "unsupported_tensor_layout",
                "Hikoboshi 0.1 supports row_major tensor layout only");
    }
  }
  return ok;
}

bool validate_backend_span(
    const hiko_u::Span<const hiko_u::PackageBackendRequirement>& backends,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = backends.size != 0;
  if (backends.size == 0) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::WorkflowCompatibility,
              "missing_backend_requirement",
              "Hikoboshi 0.1 package validation requires cpu.scalar backend");
  }
  for (std::size_t index = 0; index < backends.size; ++index) {
    if (!supported_backend(backends.data[index])) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::WorkflowCompatibility,
                "unsupported_backend_requirement",
                "Hikoboshi 0.1 supports cpu.scalar backend requirements only");
    }
  }
  return ok;
}

const hiko_u::PackageDescriptor& default_descriptor() noexcept {
  static const hiko_u::PackageDescriptor descriptor = [] {
    const WeightManifestView& manifest = default_mpnn_d64_manifest();
    const hiko_dr::RegisteredArchitectureRecord* arch =
        hiko_dr::find_architecture(kDefaultMpnn64ArchitectureId);
    const hiko_dr::RegisteredScoringRecord* scoring =
        hiko_dr::find_scoring(hiko_u::ScoreMethod::RawDotV1);
    const hiko_dr::RegisteredAlignmentRecord* alignment =
        hiko_dr::find_alignment(hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1);
    return hiko_u::PackageDescriptor{
        {manifest.schema_version,
         manifest.model_name,
         manifest.model_family,
         manifest.model_version,
         hiko_u::PackageKind::RegisteredArchitecture,
         {kDefaultMpnnD64Aliases,
          sizeof(kDefaultMpnnD64Aliases) / sizeof(kDefaultMpnnD64Aliases[0])}},
        {hiko_u::PackageExecutionMode::RegisteredArchitecture,
         kDefaultMpnn64ArchitectureId,
         arch->capability_descriptor->backends},
        *arch->capability_descriptor,
        *arch->io_contract,
        {arch->capability_descriptor->output_kinds},
        {scoring->kind,
         scoring->inputs,
         hiko_u::ScoreOutputKind::ScoreMatrix,
         {scoring->output_dtype,
          hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget,
          true,
          true,
          hiko_u::ScoreNormalization::None,
          hiko_u::ScoreScaleFamily::RawDot,
          scoring->kind}},
        {manifest.gap_parameter_family,
         alignment->gap_families_supported.data[0],
         manifest.gap_open,
         manifest.gap_extension,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         scoring->kind},
        {manifest.soft_gap_parameter_family,
         alignment->gap_families_supported.data[0],
         manifest.soft_gap_open,
         manifest.soft_gap_extension,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         scoring->kind},
        {alignment->kind, {kTracebackPolicies, 2}},
        {detail::embedded_mpnn_d64_handle()},
    };
  }();
  return descriptor;
}

const hiko_u::PackageHandle& default_package_handle() noexcept {
  static const hiko_u::PackageHandle handle{
      detail::embedded_mpnn_d64_handle().opaque,
      &default_descriptor(),
  };
  return handle;
}

const hiko_u::PackageDescriptor& esm2_8m_descriptor() noexcept {
  static const hiko_u::PackageDescriptor descriptor = [] {
    const WeightManifestView& manifest = default_esm2_8m_manifest();
    const hiko_dr::RegisteredArchitectureRecord* arch =
        hiko_dr::find_architecture(kDefaultEsm2_8mArchitectureId);
    const hiko_dr::RegisteredScoringRecord* scoring =
        hiko_dr::find_scoring(hiko_u::ScoreMethod::RawDotV1);
    const hiko_dr::RegisteredAlignmentRecord* alignment =
        hiko_dr::find_alignment(hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1);
    return hiko_u::PackageDescriptor{
        {manifest.schema_version,
         manifest.model_name,
         manifest.model_family,
         manifest.model_version,
         hiko_u::PackageKind::RegisteredArchitecture,
         {kDefaultEsm2_8mAliases,
          sizeof(kDefaultEsm2_8mAliases) / sizeof(kDefaultEsm2_8mAliases[0])}},
        {hiko_u::PackageExecutionMode::RegisteredArchitecture,
         kDefaultEsm2_8mArchitectureId,
         arch->capability_descriptor->backends},
        *arch->capability_descriptor,
        *arch->io_contract,
        {arch->capability_descriptor->output_kinds},
        {scoring->kind,
         scoring->inputs,
         hiko_u::ScoreOutputKind::ScoreMatrix,
         {scoring->output_dtype,
          hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget,
          true,
          true,
          hiko_u::ScoreNormalization::None,
          hiko_u::ScoreScaleFamily::RawDot,
          scoring->kind}},
        {manifest.gap_parameter_family,
         alignment->gap_families_supported.data[0],
         manifest.gap_open,
         manifest.gap_extension,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         scoring->kind},
        {manifest.soft_gap_parameter_family,
         alignment->gap_families_supported.data[0],
         manifest.soft_gap_open,
         manifest.soft_gap_extension,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         scoring->kind},
        {alignment->kind, {kTracebackPolicies, 2}},
        {detail::embedded_esm2_8m_handle()},
    };
  }();
  return descriptor;
}

const hiko_u::PackageHandle& esm2_8m_package_handle() noexcept {
  static const hiko_u::PackageHandle handle{
      detail::embedded_esm2_8m_handle().opaque,
      &esm2_8m_descriptor(),
  };
  return handle;
}

const hiko_u::PackageDescriptor& proteinmpnn_v48_eps020_descriptor() noexcept {
  static const hiko_u::PackageDescriptor descriptor = [] {
    const WeightManifestView& manifest =
        default_proteinmpnn_v48_eps020_manifest();
    return hiko_u::PackageDescriptor{
        {manifest.schema_version,
         manifest.model_name,
         manifest.model_family,
         manifest.model_version,
         hiko_u::PackageKind::GraphIr,
         {kDefaultProteinMpnnV48Eps020Aliases,
          sizeof(kDefaultProteinMpnnV48Eps020Aliases) /
              sizeof(kDefaultProteinMpnnV48Eps020Aliases[0])}},
        {hiko_u::PackageExecutionMode::GraphIr,
         kDefaultProteinMpnnV48Eps020ArchitectureId,
         {kProteinMpnnV48020Backends,
          sizeof(kProteinMpnnV48020Backends) /
              sizeof(kProteinMpnnV48020Backends[0])}},
        {kProteinMpnnV48020CapabilityFlags,
         {kProteinMpnnV48020InputRoutes,
          sizeof(kProteinMpnnV48020InputRoutes) /
              sizeof(kProteinMpnnV48020InputRoutes[0])},
         {kProteinMpnnV48020Preprocessing,
          sizeof(kProteinMpnnV48020Preprocessing) /
              sizeof(kProteinMpnnV48020Preprocessing[0])},
         {kProteinMpnnV48020OutputKinds,
          sizeof(kProteinMpnnV48020OutputKinds) /
              sizeof(kProteinMpnnV48020OutputKinds[0])},
         {kProteinMpnnV48020Dtypes,
          sizeof(kProteinMpnnV48020Dtypes) /
              sizeof(kProteinMpnnV48020Dtypes[0])},
         {kProteinMpnnV48020Layouts,
          sizeof(kProteinMpnnV48020Layouts) /
              sizeof(kProteinMpnnV48020Layouts[0])},
         {kProteinMpnnV48020Backends,
          sizeof(kProteinMpnnV48020Backends) /
              sizeof(kProteinMpnnV48020Backends[0])}},
        {{kProteinMpnnV48020InputRoutes,
          sizeof(kProteinMpnnV48020InputRoutes) /
              sizeof(kProteinMpnnV48020InputRoutes[0])}},
        {{kProteinMpnnV48020OutputKinds,
          sizeof(kProteinMpnnV48020OutputKinds) /
              sizeof(kProteinMpnnV48020OutputKinds[0])}},
        {hiko_u::ScoreMethod::LearnedPairScorerV1,
         {kProteinMpnnV48020ScoreInputs,
          sizeof(kProteinMpnnV48020ScoreInputs) /
              sizeof(kProteinMpnnV48020ScoreInputs[0])},
         hiko_u::ScoreOutputKind::ScoreMatrix,
         {hiko_u::DataType::Float32,
          hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget,
          true,
          false,
          hiko_u::ScoreNormalization::PackageSpecific,
          hiko_u::ScoreScaleFamily::LearnedLogit,
          hiko_u::ScoreMethod::LearnedPairScorerV1}},
        {manifest.gap_parameter_family,
         hiko_u::GapModel::Affine,
         manifest.gap_open,
         manifest.gap_extension,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         hiko_u::ScoreMethod::LearnedPairScorerV1},
        {manifest.soft_gap_parameter_family,
         hiko_u::GapModel::Affine,
         manifest.soft_gap_open,
         manifest.soft_gap_extension,
         hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
         hiko_u::ScoreMethod::LearnedPairScorerV1},
        {hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1, {nullptr, 0}},
        {detail::embedded_proteinmpnn_v48_eps020_handle()},
    };
  }();
  return descriptor;
}

const hiko_u::PackageHandle& proteinmpnn_v48_eps020_package_handle() noexcept {
  static const hiko_u::PackageHandle handle{
      detail::embedded_proteinmpnn_v48_eps020_handle().opaque,
      &proteinmpnn_v48_eps020_descriptor(),
  };
  return handle;
}

hiko_u::PackageValidationReport validate_proteinmpnn_v48_eps020_package(
    PackageHandle package,
    PackageValidationBuffer& buffer) noexcept {
  std::size_t diagnostic_count = 0;
  std::size_t warning_count = 0;
  std::uint64_t passed_stage_flags = 0;
  std::uint64_t failed_stage_flags = 0;

  if (package.descriptor == nullptr) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::SchemaVersion,
              "missing_package_descriptor",
              "ProteinMPNN v_48_020 package validation requires a descriptor");
    failed_stage_flags |= stage_flag(hiko_u::PackageValidationStage::SchemaVersion);
    return {{nullptr, nullptr},
            {buffer.diagnostics, diagnostic_count},
            {buffer.warnings, warning_count},
            passed_stage_flags,
            failed_stage_flags,
            false};
  }

  const hiko_u::PackageDescriptor& descriptor = *package.descriptor;
  const bool generated_tensor_ok =
      detail::validate_proteinmpnn_v48_eps020_generated_tensors(
          {generated::proteinmpnn_v48_eps020::kRuntimeTensors,
           generated::proteinmpnn_v48_eps020::kRuntimeTensorCount},
          buffer, diagnostic_count);

  const bool schema_ok =
      descriptor.identity.package_schema_version == std::string_view{"0.1.0"} &&
      descriptor.identity.package_id == kDefaultProteinMpnnV48Eps020ModelName &&
      descriptor.identity.package_family ==
          kDefaultProteinMpnnV48020ModelFamily &&
      descriptor.identity.package_kind == hiko_u::PackageKind::GraphIr &&
      descriptor.execution.mode == hiko_u::PackageExecutionMode::GraphIr &&
      descriptor.execution.architecture_id ==
          kDefaultProteinMpnnV48Eps020ArchitectureId;
  if (!schema_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::SchemaVersion,
              "unsupported_package_identity",
              "ProteinMPNN v_48_020 descriptor must use the reserved inverse-folding package identity");
  }
  mark_stage(schema_ok, hiko_u::PackageValidationStage::SchemaVersion,
             passed_stage_flags, failed_stage_flags);

  const bool storage_ok =
      detail::embedded_proteinmpnn_v48_eps020_manifest_matches() &&
      descriptor.compatibility_views.weights.view != nullptr &&
      descriptor.compatibility_views.weights.view->metadata.model_name ==
          kDefaultProteinMpnnV48Eps020ModelName &&
      descriptor.compatibility_views.weights.view->metadata.checksum ==
          default_proteinmpnn_v48_eps020_manifest().checksum;
  if (!storage_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::StorageChecksum,
              "storage_manifest_mismatch",
              "compiled ProteinMPNN v_48_020 storage does not match the embedded manifest");
  }
  mark_stage(storage_ok, hiko_u::PackageValidationStage::StorageChecksum,
             passed_stage_flags, failed_stage_flags);

  bool tensor_ok = generated_tensor_ok;
  tensor_ok = validate_dtype_span(descriptor.capabilities.dtypes, buffer,
                                  diagnostic_count) &&
              tensor_ok;
  tensor_ok = validate_layout_span(descriptor.capabilities.layouts, buffer,
                                   diagnostic_count) &&
              tensor_ok;
  mark_stage(tensor_ok,
             hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
             passed_stage_flags, failed_stage_flags);

  const bool non_aligner_gaps_ok =
      descriptor.gaps.family == kInverseFoldingGapFamily &&
      descriptor.gaps.model == hiko_u::GapModel::Affine &&
      descriptor.gaps.gap_open == 0.0F &&
      descriptor.gaps.gap_extension == 0.0F &&
      descriptor.soft_gaps.family == kInverseFoldingGapFamily &&
      descriptor.soft_gaps.model == hiko_u::GapModel::Affine &&
      descriptor.soft_gaps.gap_open == 0.0F &&
      descriptor.soft_gaps.gap_extension == 0.0F;
  if (!non_aligner_gaps_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::GapModelDefaults,
              "unsupported_gap_defaults",
              "ProteinMPNN v_48_020 gaps must remain explicit non-aligner sentinels");
  }
  mark_stage(non_aligner_gaps_ok,
             hiko_u::PackageValidationStage::GapModelDefaults,
             passed_stage_flags, failed_stage_flags);

  const bool prepared_ok = failed_stage_flags == 0;
  mark_stage(prepared_ok, hiko_u::PackageValidationStage::PreparedStateBuild,
             passed_stage_flags, failed_stage_flags);
  passed_stage_flags &= ~failed_stage_flags;

  return {prepared_ok ? package : hiko_u::PackageHandle{nullptr, nullptr},
          {buffer.diagnostics, diagnostic_count},
          {buffer.warnings, warning_count},
          passed_stage_flags,
          failed_stage_flags,
          prepared_ok};
}

const PackageRegistryRecord* package_registry_data() noexcept {
  static const PackageRegistryRecord records[] = {
      available_compiled_package_record(default_package_handle()),
      available_compiled_package_record(esm2_8m_package_handle()),
      available_compiled_package_record(proteinmpnn_v48_eps020_package_handle()),
  };
  return records;
}

}  // namespace

universal::Span<const PackageRegistryRecord> compiled_packages() noexcept {
  return {package_registry_data(), 3};
}

universal::Result<PackageHandle> default_package(
    const std::string_view package_id) noexcept {
  if (matches_default_mpnn64_id(package_id)) {
    PackageValidationBuffer buffer{};
    const PackageHandle package = package_registry_data()[0].package;
    const hiko_u::PackageValidationReport report =
        validate_mpnn_d64_package(package, buffer);
    if (!report.ok) {
      return {{hiko_u::StatusCode::FailedPrecondition,
               "default hikoboshi-mpnn-d64 package validation failed"},
              {nullptr, nullptr}};
    }
    return {{hiko_u::StatusCode::Ok, ""}, report.accepted_handle};
  }
  if (matches_default_esm2_8m_id(package_id)) {
    PackageValidationBuffer buffer{};
    const PackageHandle package = package_registry_data()[1].package;
    const hiko_u::PackageValidationReport report =
        validate_esm2_8m_package(package, buffer);
    if (!report.ok) {
      return {{hiko_u::StatusCode::FailedPrecondition,
               "default hikoboshi-esm2-8m package validation failed"},
              {nullptr, nullptr}};
    }
    return {{hiko_u::StatusCode::Ok, ""}, report.accepted_handle};
  }
  if (matches_proteinmpnn_v48_eps020_id(package_id)) {
    PackageValidationBuffer buffer{};
    const PackageHandle package = package_registry_data()[2].package;
    const hiko_u::PackageValidationReport report =
        validate_proteinmpnn_v48_eps020_package(package, buffer);
    if (!report.ok) {
      return {{hiko_u::StatusCode::FailedPrecondition,
               "default proteinmpnn-v48-eps020 package validation failed"},
              {nullptr, nullptr}};
    }
    return {{hiko_u::StatusCode::Ok, ""}, report.accepted_handle};
  }
  return {{hiko_u::StatusCode::InvalidArgument, kUnknownPackageDiagnostic},
          {nullptr, nullptr}};
}

universal::PackageValidationReport validate_mpnn_d64_package(
    PackageHandle package,
    PackageValidationBuffer& buffer) noexcept {
  std::size_t diagnostic_count = 0;
  std::size_t warning_count = 0;
  std::uint64_t passed_stage_flags = 0;
  std::uint64_t failed_stage_flags = 0;

  if (package.descriptor == nullptr) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::SchemaVersion,
              "missing_package_descriptor",
              "Hikoboshi package validation requires a package descriptor");
    failed_stage_flags |= stage_flag(hiko_u::PackageValidationStage::SchemaVersion);
    return {{nullptr, nullptr},
            {buffer.diagnostics, diagnostic_count},
            {buffer.warnings, warning_count},
            passed_stage_flags,
            failed_stage_flags,
            false};
  }

  const hiko_u::PackageDescriptor& descriptor = *package.descriptor;

  // Stage 4 + Stage 9 evidence: per-tensor slot, shape, alignment, and
  // historical-tensor checks against the embedded blob. Emits legacy
  // diagnostic codes that the existing weights_package_provider_tests
  // assertions match against; the staged pipeline adds canonical codes.
  const bool generated_tensor_ok = detail::validate_mpnn_d64_generated_tensors(
      {generated::mpnn_d64::kRuntimeTensors,
       generated::mpnn_d64::kRuntimeTensorCount},
      buffer, diagnostic_count);

  // Drive the canonical ten-stage validator from
  // cpp/dispatch/registry/validation_core. The pipeline consults the closed
  // architecture / scoring / alignment / capability registries and emits
  // the decision-report rejection codes from
  // hikoboshi/universal/package_validation_codes.hpp. Diagnostics produced
  // here coexist with the legacy provider-specific codes appended below.
  hiko_dr::ValidationDiagnosticsBuffer pipeline_buffer{};
  const WeightManifestView& manifest_for_pipeline = default_mpnn_d64_manifest();
  const hiko_dr::TensorChecksumView pipeline_tensor_checksums[] = {
      // Hikoboshi 0.1.0 surfaces the embedded payload checksum via the
      // manifest; per-tensor SHA-256 records live alongside each tensor
      // record in `kRuntimeTensors` and are walked individually by
      // validate_mpnn_d64_generated_tensors above. Carry the manifest-level
      // entry through provenance so the canonical validator sees a
      // consistent `tensor_checksums` span.
      {manifest_for_pipeline.model_name, manifest_for_pipeline.checksum},
  };
  const hiko_dr::PackageProvenance pipeline_provenance{
      manifest_for_pipeline.checksum,
      manifest_for_pipeline.checksum_algorithm,
      manifest_for_pipeline.source_checkpoint,
      {pipeline_tensor_checksums,
       sizeof(pipeline_tensor_checksums) /
           sizeof(pipeline_tensor_checksums[0])},
  };
  const hiko_u::Span<const std::uint8_t> pipeline_payload{
      reinterpret_cast<const std::uint8_t*>(
          generated::mpnn_d64::kSafetensorsBlob),
      generated::mpnn_d64::kSafetensorsBlobLength,
  };
  const hiko_u::PackageValidationReport pipeline_report =
      hiko_dr::validate_package(descriptor, pipeline_provenance,
                             pipeline_payload, pipeline_buffer);
  // Copy pipeline diagnostics into the caller-owned buffer. Diagnostic
  // entries are PODs whose `code`/`message` string_views reference static
  // storage (constexpr in `package_validation_codes.hpp`), so copying the
  // structs preserves the data once `pipeline_buffer` goes out of scope.
  for (std::size_t index = 0; index < pipeline_report.diagnostics.size; ++index) {
    if (diagnostic_count < kPackageValidationDiagnosticCapacity) {
      buffer.diagnostics[diagnostic_count++] =
          pipeline_report.diagnostics.data[index];
    }
  }
  for (std::size_t index = 0; index < pipeline_report.warnings.size; ++index) {
    if (warning_count < kPackageValidationWarningCapacity) {
      buffer.warnings[warning_count++] = pipeline_report.warnings.data[index];
    }
  }
  passed_stage_flags |= pipeline_report.passed_stage_flags;
  failed_stage_flags |= pipeline_report.failed_stage_flags;

  const bool schema_ok =
      descriptor.identity.package_schema_version == std::string_view{"0.1.0"} &&
      descriptor.identity.package_id == kDefaultMpnnD64ModelName &&
      descriptor.identity.package_family == kDefaultMpnn64ModelFamily &&
      descriptor.identity.package_kind ==
          hiko_u::PackageKind::RegisteredArchitecture;
  if (!schema_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::SchemaVersion,
              "unsupported_package_identity",
              "Hikoboshi 0.1 supports the registered hikoboshi-mpnn-d64 package descriptor only");
  }
  mark_stage(schema_ok, hiko_u::PackageValidationStage::SchemaVersion,
             passed_stage_flags, failed_stage_flags);

  const bool storage_ok =
      detail::embedded_mpnn_d64_manifest_matches() &&
      descriptor.compatibility_views.weights.view != nullptr &&
      descriptor.compatibility_views.weights.view->metadata.model_name ==
          kDefaultMpnnD64ModelName &&
      descriptor.compatibility_views.weights.view->metadata.checksum ==
          default_mpnn_d64_manifest().checksum;
  if (!storage_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::StorageChecksum,
              "storage_manifest_mismatch",
              "compiled hikoboshi-mpnn-d64 storage does not match the embedded manifest");
  }
  mark_stage(storage_ok, hiko_u::PackageValidationStage::StorageChecksum,
             passed_stage_flags, failed_stage_flags);

  bool architecture_ok = true;
  if (descriptor.execution.mode !=
      hiko_u::PackageExecutionMode::RegisteredArchitecture) {
    architecture_ok = false;
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ArchitectureRegistration,
              "unsupported_execution_mode",
              "Hikoboshi 0.1 supports registered_architecture execution only");
  }
  if (descriptor.execution.architecture_id != kDefaultMpnn64ArchitectureId) {
    architecture_ok = false;
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ArchitectureRegistration,
              "unsupported_architecture",
              "Hikoboshi 0.1 supports the hikoboshi_mpnn_v1 registered architecture only");
  }
  mark_stage(architecture_ok,
             hiko_u::PackageValidationStage::ArchitectureRegistration,
             passed_stage_flags, failed_stage_flags);

  bool tensor_ok = generated_tensor_ok;
  tensor_ok = validate_dtype_span(descriptor.capabilities.dtypes, buffer,
                                  diagnostic_count) &&
              tensor_ok;
  tensor_ok = validate_layout_span(descriptor.capabilities.layouts, buffer,
                                   diagnostic_count) &&
              tensor_ok;
  mark_stage(tensor_ok,
             hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
             passed_stage_flags, failed_stage_flags);

  bool input_ok = true;
  input_ok = validate_input_span(descriptor.inputs.routes, buffer,
                                 diagnostic_count) &&
             input_ok;
  input_ok = validate_input_span(descriptor.capabilities.input_routes, buffer,
                                 diagnostic_count) &&
             input_ok;
  mark_stage(input_ok, hiko_u::PackageValidationStage::InputRoute,
             passed_stage_flags, failed_stage_flags);

  const bool preprocessing_ok = validate_preprocessing_span(
      descriptor.capabilities.preprocessing, buffer, diagnostic_count);
  mark_stage(preprocessing_ok,
             hiko_u::PackageValidationStage::PreprocessingCapabilities,
             passed_stage_flags, failed_stage_flags);

  bool scoring_ok = descriptor.scoring.method == hiko_u::ScoreMethod::RawDotV1 &&
                    descriptor.scoring.output ==
                        hiko_u::ScoreOutputKind::ScoreMatrix &&
                    descriptor.scoring.inputs.size != 0;
  for (std::size_t index = 0; index < descriptor.scoring.inputs.size; ++index) {
    if (descriptor.scoring.inputs.data[index] !=
        hiko_u::ScoreInputKind::ResidueEmbeddings) {
      scoring_ok = false;
    }
  }
  if (!scoring_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ScoringMethod,
              "unsupported_score_method",
              "Hikoboshi 0.1 supports raw_dot_v1 residue-embedding scoring only");
  }
  mark_stage(scoring_ok, hiko_u::PackageValidationStage::ScoringMethod,
             passed_stage_flags, failed_stage_flags);

  bool semantics_ok =
      descriptor.scoring.semantics.dtype == hiko_u::DataType::Float32 &&
      descriptor.scoring.semantics.layout ==
          hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget &&
      descriptor.scoring.semantics.higher_is_better &&
      descriptor.scoring.semantics.local_affine_additive &&
      descriptor.scoring.semantics.normalization ==
          hiko_u::ScoreNormalization::None &&
      descriptor.scoring.semantics.scale_family ==
          hiko_u::ScoreScaleFamily::RawDot;
  if (descriptor.scoring.semantics.dtype != hiko_u::DataType::Float32) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ScoreMatrixSemantics,
              "unsupported_score_dtype",
              "Hikoboshi 0.1 supports float32 score matrices only");
  }
  if (descriptor.scoring.semantics.layout !=
      hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ScoreMatrixSemantics,
              "unsupported_score_layout",
              "Hikoboshi 0.1 supports row_major_query_by_target score matrices only");
  }
  if (!semantics_ok &&
      descriptor.scoring.semantics.dtype == hiko_u::DataType::Float32 &&
      descriptor.scoring.semantics.layout ==
          hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ScoreMatrixSemantics,
              "unsupported_score_semantics",
              "Hikoboshi 0.1 supports raw-dot additive score semantics only");
  }
  mark_stage(semantics_ok, hiko_u::PackageValidationStage::ScoreMatrixSemantics,
             passed_stage_flags, failed_stage_flags);

  const bool gaps_ok =
      descriptor.gaps.family == kHardSwGapFamily &&
      descriptor.gaps.model == hiko_u::GapModel::Affine &&
      descriptor.gaps.gap_open == kHardSwDefaultGapOpen &&
      descriptor.gaps.gap_extension == kHardSwDefaultGapExtension &&
      descriptor.gaps.convention ==
          hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension &&
      descriptor.gaps.calibrated_for_score_method ==
          hiko_u::ScoreMethod::RawDotV1 &&
      descriptor.soft_gaps.family == kSoftSwGapFamily &&
      descriptor.soft_gaps.model == hiko_u::GapModel::Affine &&
      descriptor.soft_gaps.gap_open == kSoftSwMpnn64GapOpen &&
      descriptor.soft_gaps.gap_extension == kSoftSwMpnn64GapExtension &&
      descriptor.soft_gaps.convention ==
          hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension &&
      descriptor.soft_gaps.calibrated_for_score_method ==
          hiko_u::ScoreMethod::RawDotV1;
  if (!gaps_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::GapModelDefaults,
              "unsupported_gap_defaults",
              "hikoboshi-mpnn-d64 descriptor gaps must match calibrated hard-SW and soft-SW defaults");
  }
  mark_stage(gaps_ok, hiko_u::PackageValidationStage::GapModelDefaults,
             passed_stage_flags, failed_stage_flags);

  const bool alignment_ok =
      descriptor.alignment.algorithm ==
      hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1;
  if (!alignment_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::AlignmentAlgorithm,
              "unsupported_alignment_algorithm",
              "Hikoboshi 0.1 supports hard_local_affine_sw_v1 alignment only");
  }
  mark_stage(alignment_ok, hiko_u::PackageValidationStage::AlignmentAlgorithm,
             passed_stage_flags, failed_stage_flags);

  bool workflow_ok = true;
  workflow_ok = validate_backend_span(descriptor.execution.backend_requirements,
                                      buffer, diagnostic_count) &&
                workflow_ok;
  workflow_ok = validate_backend_span(descriptor.capabilities.backends, buffer,
                                      diagnostic_count) &&
                workflow_ok;
  workflow_ok = validate_output_span(descriptor.outputs.kinds, buffer,
                                     diagnostic_count) &&
                workflow_ok;
  workflow_ok = validate_output_span(descriptor.capabilities.output_kinds,
                                     buffer, diagnostic_count) &&
                workflow_ok;
  mark_stage(workflow_ok, hiko_u::PackageValidationStage::WorkflowCompatibility,
             passed_stage_flags, failed_stage_flags);

  const bool prepared_ok = failed_stage_flags == 0;
  mark_stage(prepared_ok, hiko_u::PackageValidationStage::PreparedStateBuild,
             passed_stage_flags, failed_stage_flags);
  // After merging legacy and canonical-pipeline flag sets, failure always
  // dominates passing for any stage bit that both sides flipped. Clearing
  // overlapping passed bits keeps the report shape consistent: each stage
  // appears in exactly one of `passed_stage_flags` or `failed_stage_flags`.
  passed_stage_flags &= ~failed_stage_flags;

  return {prepared_ok ? package : hiko_u::PackageHandle{nullptr, nullptr},
          {buffer.diagnostics, diagnostic_count},
          {buffer.warnings, warning_count},
          passed_stage_flags,
          failed_stage_flags,
          prepared_ok};
}

universal::PackageValidationReport validate_mpnn64_package(
    PackageHandle package,
    PackageValidationBuffer& buffer) noexcept {
  return validate_mpnn_d64_package(package, buffer);
}

universal::Result<PackageHandle> default_mpnn_d64_package() noexcept {
  return default_package(kDefaultMpnnD64ModelName);
}

universal::Result<PackageHandle> default_mpnn64_package() noexcept {
  return default_mpnn_d64_package();
}

universal::Result<universal::WeightsHandle> default_mpnn_d64() noexcept {
  const universal::Result<PackageHandle> package = default_mpnn_d64_package();
  if (package.status.code != universal::StatusCode::Ok ||
      package.value.descriptor == nullptr) {
    return {package.status, {nullptr, nullptr}};
  }
  return {package.status,
          package.value.descriptor->compatibility_views.weights};
}

universal::Result<universal::WeightsHandle> default_mpnn64() noexcept {
  return default_mpnn_d64();
}

namespace {

// ESM2-8M descriptor scoring/input/preprocessing rules. The canonical
// 10-stage pipeline already enforces these against the registered
// capability records; the helpers below add legacy diagnostic codes
// that mirror the MPNN-64 surface so the package-specific validator
// produces a coherent error set per package.
bool esm2_supported_input(const hiko_u::PackageInputKind input) noexcept {
  switch (input) {
    case hiko_u::PackageInputKind::SequenceTokens:
    case hiko_u::PackageInputKind::ResidueEmbeddings:
      return true;
    case hiko_u::PackageInputKind::StructureBackboneAtoms:
    case hiko_u::PackageInputKind::CoordsBackbone:
    case hiko_u::PackageInputKind::StructureAllAtom:
    case hiko_u::PackageInputKind::DirectScoreMatrix:
      return false;
  }
  return false;
}

bool esm2_supported_preprocessing(
    const hiko_u::PackagePreprocessingCapability capability) noexcept {
  switch (capability) {
    case hiko_u::PackagePreprocessingCapability::Tokenization:
      return true;
    case hiko_u::PackagePreprocessingCapability::AtomInference:
    case hiko_u::PackagePreprocessingCapability::VirtualCb:
    case hiko_u::PackagePreprocessingCapability::CaKnn:
    case hiko_u::PackagePreprocessingCapability::AtomPairDistances:
    case hiko_u::PackagePreprocessingCapability::RbfExpand:
    case hiko_u::PackagePreprocessingCapability::PositionalEncoding:
      return false;
  }
  return false;
}

bool esm2_validate_input_span(
    const hiko_u::Span<const hiko_u::PackageInputKind>& routes,
    PackageValidationBuffer& buffer, std::size_t& diagnostic_count) noexcept {
  bool ok = routes.size != 0;
  if (routes.size == 0) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::InputRoute,
              "missing_input_route",
              "hikoboshi-esm2-8m validation requires at least one input route");
  }
  for (std::size_t index = 0; index < routes.size; ++index) {
    if (!esm2_supported_input(routes.data[index])) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::InputRoute,
                "unsupported_input_modality",
                "hikoboshi-esm2-8m supports sequence_tokens and "
                "residue_embeddings inputs only");
    }
  }
  return ok;
}

bool esm2_validate_preprocessing_span(
    const hiko_u::Span<const hiko_u::PackagePreprocessingCapability>& capabilities,
    PackageValidationBuffer& buffer, std::size_t& diagnostic_count) noexcept {
  bool ok = true;
  for (std::size_t index = 0; index < capabilities.size; ++index) {
    if (!esm2_supported_preprocessing(capabilities.data[index])) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::PreprocessingCapabilities,
                "unsupported_preprocessing_capability",
                "hikoboshi-esm2-8m supports the tokenization preprocessing "
                "capability only");
    }
  }
  return ok;
}

}  // namespace

universal::PackageValidationReport validate_esm2_8m_package(
    PackageHandle package,
    PackageValidationBuffer& buffer) noexcept {
  std::size_t diagnostic_count = 0;
  std::size_t warning_count = 0;
  std::uint64_t passed_stage_flags = 0;
  std::uint64_t failed_stage_flags = 0;

  if (package.descriptor == nullptr) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::SchemaVersion,
              "missing_package_descriptor",
              "Hikoboshi package validation requires a package descriptor");
    failed_stage_flags |= stage_flag(hiko_u::PackageValidationStage::SchemaVersion);
    return {{nullptr, nullptr},
            {buffer.diagnostics, diagnostic_count},
            {buffer.warnings, warning_count},
            passed_stage_flags,
            failed_stage_flags,
            false};
  }

  const hiko_u::PackageDescriptor& descriptor = *package.descriptor;

  // Stage 4 evidence: per-tensor slot/shape/alignment check against the
  // embedded ESM2-8M blob. Diagnostic codes mirror the MPNN-64 surface
  // so package_provider_tests can match against them.
  const bool generated_tensor_ok =
      detail::validate_esm2_8m_generated_tensors(
          {generated::esm2_8m::kRuntimeTensors,
           generated::esm2_8m::kRuntimeTensorCount},
          buffer, diagnostic_count);

  // Drive the canonical 10-stage validator from validation_core. The
  // pipeline consumes the registered architecture / scoring /
  // alignment / capability records and emits the decision-report
  // rejection codes when any stage fails.
  hiko_dr::ValidationDiagnosticsBuffer pipeline_buffer{};
  const WeightManifestView& manifest_for_pipeline = default_esm2_8m_manifest();
  const hiko_dr::TensorChecksumView pipeline_tensor_checksums[] = {
      {manifest_for_pipeline.model_name, manifest_for_pipeline.checksum},
  };
  const hiko_dr::PackageProvenance pipeline_provenance{
      manifest_for_pipeline.checksum,
      manifest_for_pipeline.checksum_algorithm,
      manifest_for_pipeline.source_checkpoint,
      {pipeline_tensor_checksums,
       sizeof(pipeline_tensor_checksums) /
           sizeof(pipeline_tensor_checksums[0])},
  };
  const hiko_u::Span<const std::uint8_t> pipeline_payload{
      reinterpret_cast<const std::uint8_t*>(
          generated::esm2_8m::kSafetensorsBlob),
      generated::esm2_8m::kSafetensorsBlobLength,
  };
  const hiko_u::PackageValidationReport pipeline_report =
      hiko_dr::validate_package(descriptor, pipeline_provenance,
                             pipeline_payload, pipeline_buffer);
  for (std::size_t index = 0; index < pipeline_report.diagnostics.size;
       ++index) {
    if (diagnostic_count < kPackageValidationDiagnosticCapacity) {
      buffer.diagnostics[diagnostic_count++] =
          pipeline_report.diagnostics.data[index];
    }
  }
  for (std::size_t index = 0; index < pipeline_report.warnings.size; ++index) {
    if (warning_count < kPackageValidationWarningCapacity) {
      buffer.warnings[warning_count++] = pipeline_report.warnings.data[index];
    }
  }
  passed_stage_flags |= pipeline_report.passed_stage_flags;
  failed_stage_flags |= pipeline_report.failed_stage_flags;

  const bool schema_ok =
      descriptor.identity.package_schema_version == std::string_view{"0.1.0"} &&
      descriptor.identity.package_id == kDefaultEsm2_8mModelName &&
      descriptor.identity.package_family == kDefaultEsm2_8mModelFamily &&
      descriptor.identity.package_kind ==
          hiko_u::PackageKind::RegisteredArchitecture;
  if (!schema_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::SchemaVersion,
              "unsupported_package_identity",
              "hikoboshi-esm2-8m descriptor must declare the registered "
              "hikoboshi-esm2-8m identity");
  }
  mark_stage(schema_ok, hiko_u::PackageValidationStage::SchemaVersion,
             passed_stage_flags, failed_stage_flags);

  const bool storage_ok =
      detail::embedded_esm2_8m_manifest_matches() &&
      descriptor.compatibility_views.weights.view != nullptr &&
      descriptor.compatibility_views.weights.view->metadata.model_name ==
          kDefaultEsm2_8mModelName &&
      descriptor.compatibility_views.weights.view->metadata.checksum ==
          default_esm2_8m_manifest().checksum;
  if (!storage_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::StorageChecksum,
              "storage_manifest_mismatch",
              "compiled hikoboshi-esm2-8m storage does not match the embedded "
              "manifest");
  }
  mark_stage(storage_ok, hiko_u::PackageValidationStage::StorageChecksum,
             passed_stage_flags, failed_stage_flags);

  bool architecture_ok = true;
  if (descriptor.execution.mode !=
      hiko_u::PackageExecutionMode::RegisteredArchitecture) {
    architecture_ok = false;
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ArchitectureRegistration,
              "unsupported_execution_mode",
              "hikoboshi-esm2-8m supports registered_architecture execution "
              "only");
  }
  if (descriptor.execution.architecture_id != kDefaultEsm2_8mArchitectureId) {
    architecture_ok = false;
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ArchitectureRegistration,
              "unsupported_architecture",
              "hikoboshi-esm2-8m descriptor must reference hikoboshi_esm2_v1");
  }
  mark_stage(architecture_ok,
             hiko_u::PackageValidationStage::ArchitectureRegistration,
             passed_stage_flags, failed_stage_flags);

  bool tensor_ok = generated_tensor_ok;
  tensor_ok = validate_dtype_span(descriptor.capabilities.dtypes, buffer,
                                  diagnostic_count) &&
              tensor_ok;
  tensor_ok = validate_layout_span(descriptor.capabilities.layouts, buffer,
                                   diagnostic_count) &&
              tensor_ok;
  mark_stage(tensor_ok,
             hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
             passed_stage_flags, failed_stage_flags);

  bool input_ok = true;
  input_ok = esm2_validate_input_span(descriptor.inputs.routes, buffer,
                                      diagnostic_count) &&
             input_ok;
  input_ok = esm2_validate_input_span(descriptor.capabilities.input_routes,
                                      buffer, diagnostic_count) &&
             input_ok;
  mark_stage(input_ok, hiko_u::PackageValidationStage::InputRoute,
             passed_stage_flags, failed_stage_flags);

  const bool preprocessing_ok = esm2_validate_preprocessing_span(
      descriptor.capabilities.preprocessing, buffer, diagnostic_count);
  mark_stage(preprocessing_ok,
             hiko_u::PackageValidationStage::PreprocessingCapabilities,
             passed_stage_flags, failed_stage_flags);

  bool scoring_ok =
      descriptor.scoring.method == hiko_u::ScoreMethod::RawDotV1 &&
      descriptor.scoring.output == hiko_u::ScoreOutputKind::ScoreMatrix &&
      descriptor.scoring.inputs.size != 0;
  for (std::size_t index = 0; index < descriptor.scoring.inputs.size; ++index) {
    if (descriptor.scoring.inputs.data[index] !=
        hiko_u::ScoreInputKind::ResidueEmbeddings) {
      scoring_ok = false;
    }
  }
  if (!scoring_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ScoringMethod,
              "unsupported_score_method",
              "hikoboshi-esm2-8m supports raw_dot_v1 residue-embedding "
              "scoring only");
  }
  mark_stage(scoring_ok, hiko_u::PackageValidationStage::ScoringMethod,
             passed_stage_flags, failed_stage_flags);

  const bool semantics_ok =
      descriptor.scoring.semantics.dtype == hiko_u::DataType::Float32 &&
      descriptor.scoring.semantics.layout ==
          hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget &&
      descriptor.scoring.semantics.higher_is_better &&
      descriptor.scoring.semantics.local_affine_additive &&
      descriptor.scoring.semantics.normalization ==
          hiko_u::ScoreNormalization::None &&
      descriptor.scoring.semantics.scale_family ==
          hiko_u::ScoreScaleFamily::RawDot;
  if (!semantics_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::ScoreMatrixSemantics,
              "unsupported_score_semantics",
              "hikoboshi-esm2-8m supports raw-dot additive score semantics "
              "only");
  }
  mark_stage(semantics_ok, hiko_u::PackageValidationStage::ScoreMatrixSemantics,
             passed_stage_flags, failed_stage_flags);

  // ESM2-8M's hard defaults come from the recorded near-zero-temperature
  // gap anneal; its soft defaults come from the T=1 checkpoint tensors.
  // Both differ from MPNN-64 because the score scales and calibrations differ.
  const bool gaps_ok =
      descriptor.gaps.family == kHardSwGapFamily &&
      descriptor.gaps.model == hiko_u::GapModel::Affine &&
      descriptor.gaps.convention ==
          hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension &&
      descriptor.gaps.calibrated_for_score_method ==
          hiko_u::ScoreMethod::RawDotV1 &&
      descriptor.gaps.gap_open == default_esm2_8m_manifest().gap_open &&
      descriptor.gaps.gap_extension == default_esm2_8m_manifest().gap_extension &&
      descriptor.soft_gaps.family == kSoftSwGapFamily &&
      descriptor.soft_gaps.model == hiko_u::GapModel::Affine &&
      descriptor.soft_gaps.convention ==
          hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension &&
      descriptor.soft_gaps.calibrated_for_score_method ==
          hiko_u::ScoreMethod::RawDotV1 &&
      descriptor.soft_gaps.gap_open ==
          default_esm2_8m_manifest().soft_gap_open &&
      descriptor.soft_gaps.gap_extension ==
          default_esm2_8m_manifest().soft_gap_extension;
  if (!gaps_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::GapModelDefaults,
              "unsupported_gap_defaults",
              "hikoboshi-esm2-8m descriptor gaps must match the calibrated hard "
              "and soft manifest values");
  }
  mark_stage(gaps_ok, hiko_u::PackageValidationStage::GapModelDefaults,
             passed_stage_flags, failed_stage_flags);

  const bool alignment_ok =
      descriptor.alignment.algorithm ==
      hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1;
  if (!alignment_ok) {
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::AlignmentAlgorithm,
              "unsupported_alignment_algorithm",
              "hikoboshi-esm2-8m supports hard_local_affine_sw_v1 alignment "
              "only");
  }
  mark_stage(alignment_ok, hiko_u::PackageValidationStage::AlignmentAlgorithm,
             passed_stage_flags, failed_stage_flags);

  bool workflow_ok = true;
  workflow_ok = validate_backend_span(descriptor.execution.backend_requirements,
                                      buffer, diagnostic_count) &&
                workflow_ok;
  workflow_ok = validate_backend_span(descriptor.capabilities.backends, buffer,
                                      diagnostic_count) &&
                workflow_ok;
  workflow_ok = validate_output_span(descriptor.outputs.kinds, buffer,
                                     diagnostic_count) &&
                workflow_ok;
  workflow_ok = validate_output_span(descriptor.capabilities.output_kinds,
                                     buffer, diagnostic_count) &&
                workflow_ok;
  mark_stage(workflow_ok, hiko_u::PackageValidationStage::WorkflowCompatibility,
             passed_stage_flags, failed_stage_flags);

  const bool prepared_ok = failed_stage_flags == 0;
  mark_stage(prepared_ok, hiko_u::PackageValidationStage::PreparedStateBuild,
             passed_stage_flags, failed_stage_flags);
  passed_stage_flags &= ~failed_stage_flags;

  return {prepared_ok ? package : hiko_u::PackageHandle{nullptr, nullptr},
          {buffer.diagnostics, diagnostic_count},
          {buffer.warnings, warning_count},
          passed_stage_flags,
          failed_stage_flags,
          prepared_ok};
}

universal::Result<PackageHandle> default_esm2_8m_package() noexcept {
  return default_package(kDefaultEsm2_8mModelName);
}

universal::Result<universal::WeightsHandle> default_esm2_8m() noexcept {
  const universal::Result<PackageHandle> package = default_esm2_8m_package();
  if (package.status.code != universal::StatusCode::Ok ||
      package.value.descriptor == nullptr) {
    return {package.status, {nullptr, nullptr}};
  }
  return {package.status,
          package.value.descriptor->compatibility_views.weights};
}

}  // namespace hikoboshi::weights

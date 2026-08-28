// Tests the canonical ten-stage package validation pipeline exposed by
// `hikoboshi::dispatch::registry::validate_package`. Each block builds a
// descriptor fixture that exercises one stage (pass or fail) and asserts
// the corresponding stage flag and canonical rejection code from
// hikoboshi/universal/package_validation_codes.hpp.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <hikoboshi/dispatch/registry/validation_core.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/package_validation_codes.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>
#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

namespace hiko_u = hikoboshi::universal;
namespace hiko_dr = hikoboshi::dispatch::registry;
namespace codes = hikoboshi::universal::package_validation_codes;

namespace {

constexpr std::uint64_t stage_flag(
    const hiko_u::PackageValidationStage stage) noexcept {
  return 1ULL << static_cast<unsigned>(stage);
}

bool has_diagnostic(const hiko_u::PackageValidationReport& report,
                    const hiko_u::PackageValidationStage stage,
                    const std::string_view code) noexcept {
  for (std::size_t index = 0; index < report.diagnostics.size; ++index) {
    const hiko_u::PackageValidationDiagnostic& diagnostic =
        report.diagnostics.data[index];
    if (diagnostic.severity == hiko_u::PackageDiagnosticSeverity::Error &&
        diagnostic.stage == stage && diagnostic.code == code) {
      return true;
    }
  }
  return false;
}

bool has_warning(const hiko_u::PackageValidationReport& report,
                 const std::string_view code) noexcept {
  for (std::size_t index = 0; index < report.warnings.size; ++index) {
    if (report.warnings.data[index].code == code) {
      return true;
    }
  }
  return false;
}

// Base "all-stages-pass" fixture: a synthetic MPNN-64-shaped descriptor
// that the staged validator accepts cleanly. Each per-stage failure test
// copies this fixture and mutates the field that should trip the stage.
struct ValidFixture {
  hiko_u::PackageInputKind input_routes[3] = {
      hiko_u::PackageInputKind::StructureBackboneAtoms,
      hiko_u::PackageInputKind::CoordsBackbone,
      hiko_u::PackageInputKind::ResidueEmbeddings,
  };
  hiko_u::PackagePreprocessingCapability preprocessing[6] = {
      hiko_u::PackagePreprocessingCapability::AtomInference,
      hiko_u::PackagePreprocessingCapability::VirtualCb,
      hiko_u::PackagePreprocessingCapability::CaKnn,
      hiko_u::PackagePreprocessingCapability::AtomPairDistances,
      hiko_u::PackagePreprocessingCapability::RbfExpand,
      hiko_u::PackagePreprocessingCapability::PositionalEncoding,
  };
  hiko_u::PackageOutputKind output_kinds[1] = {
      hiko_u::PackageOutputKind::ResidueEmbeddings,
  };
  hiko_u::DataType dtypes[1] = {hiko_u::DataType::Float32};
  hiko_u::PackageTensorLayout layouts[1] = {hiko_u::PackageTensorLayout::RowMajor};
  hiko_u::PackageBackendRequirement backends[1] = {
      hiko_u::PackageBackendRequirement::CpuScalar};
  hiko_u::ScoreInputKind scoring_inputs[1] = {
      hiko_u::ScoreInputKind::ResidueEmbeddings};
  hiko_u::TracebackPolicy traceback_policies[2] = {
      hiko_u::TracebackPolicy::RequiredForPublicPairwise,
      hiko_u::TracebackPolicy::RequiredForPublicAllVsAll,
  };
  hiko_u::ModelMetadataView metadata{};
  hiko_u::WeightsView weights_view{};
  hiko_dr::TensorChecksumView tensor_checksums[1]{};
  hiko_u::PackageDescriptor descriptor{};
  hiko_dr::PackageProvenance provenance{};
  std::uint8_t payload_bytes[16]{};

  ValidFixture() {
    metadata = hiko_u::ModelMetadataView{
        hikoboshi::weights::kDefaultMpnnD64ModelName,
        hikoboshi::weights::kDefaultMpnn64ModelFamily,
        std::string_view{"test"},
        hikoboshi::weights::kDefaultMpnn64HiddenDim,
        std::string_view{"fixture-source"},
        std::string_view{"fixture-checksum"},
    };
    weights_view = hiko_u::WeightsView{metadata, {nullptr, 0}};

    descriptor.identity = {
        std::string_view{"0.1.0"},
        hikoboshi::weights::kDefaultMpnnD64ModelName,
        hikoboshi::weights::kDefaultMpnn64ModelFamily,
        std::string_view{"test"},
        hiko_u::PackageKind::RegisteredArchitecture,
        {nullptr, 0},
    };
    descriptor.execution = {
        hiko_u::PackageExecutionMode::RegisteredArchitecture,
        hikoboshi::weights::kDefaultMpnn64ArchitectureId,
        {backends, 1},
    };
    descriptor.capabilities = {
        0,
        {input_routes, 3},
        {preprocessing, 6},
        {output_kinds, 1},
        {dtypes, 1},
        {layouts, 1},
        {backends, 1},
    };
    descriptor.inputs = {{input_routes, 3}};
    descriptor.outputs = {{output_kinds, 1}};
    descriptor.scoring = {
        hiko_u::ScoreMethod::RawDotV1,
        {scoring_inputs, 1},
        hiko_u::ScoreOutputKind::ScoreMatrix,
        {hiko_u::DataType::Float32,
         hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget,
         true,
         true,
         hiko_u::ScoreNormalization::None,
         hiko_u::ScoreScaleFamily::RawDot,
         hiko_u::ScoreMethod::RawDotV1},
    };
      descriptor.gaps = {
          std::string_view{"Hikoboshi 0.1.0 hard-SW"},
          hiko_u::GapModel::Affine,
          hikoboshi::weights::kHardSwDefaultGapOpen,
          hikoboshi::weights::kHardSwDefaultGapExtension,
          hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
          hiko_u::ScoreMethod::RawDotV1,
      };
      descriptor.soft_gaps = {
          std::string_view{"Hikoboshi 0.1.0 soft-SW"},
          hiko_u::GapModel::Affine,
          hikoboshi::weights::kSoftSwMpnn64GapOpen,
          hikoboshi::weights::kSoftSwMpnn64GapExtension,
          hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
          hiko_u::ScoreMethod::RawDotV1,
      };
    descriptor.alignment = {
        hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1,
        {traceback_policies, 2},
    };
    descriptor.compatibility_views = {{nullptr, &weights_view}};

    tensor_checksums[0] = {hikoboshi::weights::kDefaultMpnnD64ModelName,
                           metadata.checksum};
    provenance = {metadata.checksum,
                  std::string_view{"sha256"},
                  std::string_view{"fixture-source"},
                  {tensor_checksums, 1}};
  }
};

hiko_u::PackageValidationReport run_validate(
    const hiko_u::PackageDescriptor& descriptor,
    const hiko_dr::PackageProvenance& provenance,
    const hiko_u::Span<const std::uint8_t> payload,
    hiko_dr::ValidationDiagnosticsBuffer& buffer) {
  return hiko_dr::validate_package(descriptor, provenance, payload, buffer);
}

}  // namespace

int main() {
  // Stage 0 — base fixture should validate cleanly with zero diagnostics
  // and every stage marked passed.
  {
    ValidFixture fixture;
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (!report.ok || report.diagnostics.size != 0 ||
        report.failed_stage_flags != 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::SchemaVersion)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::ArchitectureRegistration)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::WorkflowCompatibility)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::PreparedStateBuild)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::ScoringMethod)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::ScoreMatrixSemantics)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::AlignmentAlgorithm)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::GapModelDefaults)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::InputRoute)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::PreprocessingCapabilities)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::StorageChecksum)) == 0 ||
        report.accepted_handle.descriptor != &fixture.descriptor) {
      return 1;
    }
  }

  // Stage 1 — schema_validation: unsupported package schema version emits
  // `package_schema_version_unsupported`.
  {
    ValidFixture fixture;
    fixture.descriptor.identity.package_schema_version =
        std::string_view{"0.9.0-future"};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::SchemaVersion)) == 0 ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::SchemaVersion,
                        codes::kPackageSchemaVersionUnsupported)) {
      return 2;
    }
  }

  // Stage 1 — unknown package kind tag.
  {
    ValidFixture fixture;
    fixture.descriptor.identity.package_kind = hiko_u::PackageKind::Unknown;
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::SchemaVersion,
                        codes::kArchitectureKindUnknown)) {
      return 3;
    }
  }

  // Stage 1 — empty architecture id for registered-architecture mode.
  {
    ValidFixture fixture;
    fixture.descriptor.execution.architecture_id = std::string_view{};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::SchemaVersion,
                        codes::kArchitectureParamsInvalid)) {
      return 4;
    }
  }

  // Stage 2 — architecture_binding: id not in registry emits
  // `architecture_kind_not_registered`.
  {
    ValidFixture fixture;
    fixture.descriptor.execution.architecture_id =
        std::string_view{"hikoboshi_unknown_v1"};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::ArchitectureRegistration)) == 0 ||
        !has_diagnostic(report,
                        hiko_u::PackageValidationStage::ArchitectureRegistration,
                        codes::kArchitectureKindNotRegistered)) {
      return 5;
    }
  }

  // Stage 3 — capability_handshake: unimplemented backend without host
  // fallback emits `backend_unavailable` as an error.
  {
    ValidFixture fixture;
    const hiko_u::PackageBackendRequirement gpu_only[1] = {
        hiko_u::PackageBackendRequirement::GpuCuda};
    fixture.descriptor.capabilities.backends = {gpu_only, 1};
    fixture.descriptor.execution.backend_requirements = {gpu_only, 1};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::WorkflowCompatibility)) == 0 ||
        !has_diagnostic(report,
                        hiko_u::PackageValidationStage::WorkflowCompatibility,
                        codes::kBackendUnavailable)) {
      return 6;
    }
  }

  // Stage 3 — capability_handshake fallback policy: unimplemented backend
  // co-declared with a CpuScalar host fallback emits a warning, not an
  // error, and the stage remains passed.
  {
    ValidFixture fixture;
    const hiko_u::PackageBackendRequirement backends_with_fallback[2] = {
        hiko_u::PackageBackendRequirement::GpuCuda,
        hiko_u::PackageBackendRequirement::CpuScalar,
    };
    fixture.descriptor.capabilities.backends = {backends_with_fallback, 2};
    fixture.descriptor.execution.backend_requirements = {
        backends_with_fallback, 2};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (!report.ok ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::WorkflowCompatibility)) == 0 ||
        !has_warning(report, codes::kBackendUnavailable) ||
        has_diagnostic(report,
                       hiko_u::PackageValidationStage::WorkflowCompatibility,
                       codes::kBackendUnavailable)) {
      return 7;
    }
  }

  // Stage 3 — capability_handshake: unsupported dtype on backend.
  {
    ValidFixture fixture;
    const hiko_u::DataType bad_dtypes[1] = {hiko_u::DataType::Float64};
    fixture.descriptor.capabilities.dtypes = {bad_dtypes, 1};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        !has_diagnostic(report,
                        hiko_u::PackageValidationStage::WorkflowCompatibility,
                        codes::kDtypeUnsupportedOnBackend)) {
      return 8;
    }
  }

  // Stage 4 — tensor_table_validation: dtype outside the registered
  // capability axis emits `tensor_slot_dtype_mismatch`.
  {
    ValidFixture fixture;
    const hiko_u::DataType bad_dtypes[1] = {hiko_u::DataType::UInt8};
    fixture.descriptor.capabilities.dtypes = {bad_dtypes, 1};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes)) == 0 ||
        !has_diagnostic(
            report,
            hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
            codes::kTensorSlotDtypeMismatch)) {
      return 9;
    }
  }

  // Stage 4 — required tensor slot missing.
  {
    ValidFixture fixture;
    fixture.descriptor.compatibility_views.weights.view = nullptr;
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        !has_diagnostic(
            report,
            hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
            codes::kTensorRequiredSlotMissing)) {
      return 10;
    }
  }

  // Stage 5 — prepared_state_plan_check: when stages 1/2/4/9 all fail,
  // PreparedStateBuild must end up in failed flags too.
  {
    ValidFixture fixture;
    fixture.descriptor.execution.architecture_id =
        std::string_view{"hikoboshi_unknown_v1"};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::PreparedStateBuild)) == 0 ||
        (report.passed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::PreparedStateBuild)) != 0) {
      return 11;
    }
  }

  // Stage 6 — scoring_binding: unregistered scoring kind.
  {
    ValidFixture fixture;
    fixture.descriptor.scoring.method = hiko_u::ScoreMethod::CosineV1;
    fixture.descriptor.scoring.semantics.method = hiko_u::ScoreMethod::CosineV1;
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::ScoringMethod)) == 0 ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::ScoringMethod,
                        codes::kScoringKindNotRegistered)) {
      return 12;
    }
  }

  // Stage 6 — scoring input not accepted by the registered scoring method.
  {
    ValidFixture fixture;
    const hiko_u::ScoreInputKind sequence_input[1] = {
        hiko_u::ScoreInputKind::SequenceTokens};
    fixture.descriptor.scoring.inputs = {sequence_input, 1};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::ScoringMethod,
                        codes::kScoringInputIncompatibleWithArchitecture)) {
      return 13;
    }
  }

  // Stage 7 — alignment_binding: unregistered alignment id.
  {
    ValidFixture fixture;
    fixture.descriptor.alignment.algorithm =
        hiko_u::AlignmentAlgorithmId::GlobalAffineSwV1;
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::AlignmentAlgorithm)) == 0 ||
        !has_diagnostic(report,
                        hiko_u::PackageValidationStage::AlignmentAlgorithm,
                        codes::kAlignmentKindNotRegistered)) {
      return 14;
    }
  }

  // Stage 8 — io_contract_binding: input route not part of the
  // architecture IO contract.
  {
    ValidFixture fixture;
    const hiko_u::PackageInputKind sequence_inputs[1] = {
        hiko_u::PackageInputKind::SequenceTokens};
    fixture.descriptor.inputs.routes = {sequence_inputs, 1};
    fixture.descriptor.capabilities.input_routes = {sequence_inputs, 1};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::InputRoute)) == 0 ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::InputRoute,
                        codes::kIoContractIncompatible)) {
      return 15;
    }
  }

  // Stage 8 — io_contract_binding: descriptor names an
  // architecture_kind that is not present in architecture_registry().
  // Replaces the prior Tokenization-as-reserved fixture (Tokenization is
  // now implemented in Hikoboshi 0.1.0 because the ESM2-8M architecture
  // ships a concrete tokenizer table). The lead's escalation decision
  // under the released scalar architecture contract.
  // re-targets this slot at the unknown-architecture-kind negative case;
  // stage 2 emits architecture_kind_not_registered for the unknown id.
  {
    ValidFixture fixture;
    fixture.descriptor.execution.architecture_id =
        std::string_view{"hikoboshi_gvp_v1"};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::ArchitectureRegistration)) == 0 ||
        !has_diagnostic(report,
                        hiko_u::PackageValidationStage::ArchitectureRegistration,
                        codes::kArchitectureKindNotRegistered)) {
      return 16;
    }
  }

  // Stage 9 — provenance_check: payload checksum mismatch between the
  // compatibility weights view and the caller-supplied provenance.
  {
    ValidFixture fixture;
    fixture.provenance.payload_checksum = std::string_view{"different"};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        (report.failed_stage_flags &
         stage_flag(hiko_u::PackageValidationStage::StorageChecksum)) == 0 ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::StorageChecksum,
                        codes::kPayloadChecksumMismatch)) {
      return 17;
    }
  }

  // Stage 9 — empty per-tensor checksum entry.
  {
    ValidFixture fixture;
    fixture.tensor_checksums[0] = {std::string_view{}, std::string_view{}};
    fixture.provenance.tensor_checksums = {fixture.tensor_checksums, 1};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok ||
        !has_diagnostic(report, hiko_u::PackageValidationStage::StorageChecksum,
                        codes::kTensorChecksumMismatch)) {
      return 18;
    }
  }

  // Stage 10 — validation_result: the report POD is well-formed even on
  // failure (diagnostics span populated, accepted_handle null).
  {
    ValidFixture fixture;
    fixture.descriptor.execution.architecture_id =
        std::string_view{"hikoboshi_unknown_v1"};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = run_validate(
        fixture.descriptor, fixture.provenance,
        {fixture.payload_bytes, sizeof(fixture.payload_bytes)}, buffer);
    if (report.ok || report.accepted_handle.descriptor != nullptr ||
        report.accepted_handle.opaque != nullptr ||
        report.diagnostics.size == 0 ||
        report.diagnostics.data != buffer.diagnostics) {
      return 19;
    }
  }

  // Default compiled Hikoboshi-MPNN-64 package must validate cleanly through
  // the canonical pipeline driven by the real registries.
  {
    const hiko_u::Result<hikoboshi::weights::PackageHandle> package_result =
        hikoboshi::weights::default_mpnn_d64_package();
    if (package_result.status.code != hiko_u::StatusCode::Ok ||
        package_result.value.descriptor == nullptr) {
      return 20;
    }
    const hikoboshi::weights::WeightManifestView& manifest =
        hikoboshi::weights::default_mpnn_d64_manifest();
    const hiko_dr::TensorChecksumView tensor_checksums[1] = {
        {hikoboshi::weights::kDefaultMpnnD64ModelName, manifest.checksum}};
    const hiko_dr::PackageProvenance provenance{
        manifest.checksum, manifest.checksum_algorithm,
        manifest.source_checkpoint, {tensor_checksums, 1}};
    const std::uint8_t payload[16]{};
    hiko_dr::ValidationDiagnosticsBuffer buffer{};
    const hiko_u::PackageValidationReport report = hiko_dr::validate_package(
        *package_result.value.descriptor, provenance, {payload, 16}, buffer);
    if (!report.ok || report.diagnostics.size != 0 ||
        report.failed_stage_flags != 0 ||
        report.accepted_handle.descriptor != package_result.value.descriptor) {
      return 21;
    }
  }

  return 0;
}

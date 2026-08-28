#include <hikoboshi/dispatch/registry/validation_core.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <hikoboshi/dispatch/registry/alignment.hpp>
#include <hikoboshi/dispatch/registry/architecture.hpp>
#include <hikoboshi/dispatch/registry/capability.hpp>
#include <hikoboshi/dispatch/registry/scoring.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/package_validation_codes.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::dispatch::registry {
namespace {

namespace hiko_u = hikoboshi::universal;
namespace codes = hikoboshi::universal::package_validation_codes;

constexpr std::string_view kSupportedPackageSchemaVersion{"0.1.0"};

constexpr std::uint64_t stage_flag(
    const hiko_u::PackageValidationStage stage) noexcept {
  return 1ULL << static_cast<unsigned>(stage);
}

void emit_error(ValidationDiagnosticsBuffer& buffer,
                std::size_t& diagnostic_count,
                const hiko_u::PackageValidationStage stage,
                const std::string_view code,
                const std::string_view message) noexcept {
  if (diagnostic_count < kValidationDiagnosticCapacity) {
    buffer.diagnostics[diagnostic_count++] = {
        hiko_u::PackageDiagnosticSeverity::Error,
        stage,
        code,
        message,
    };
  }
}

void emit_warning(ValidationDiagnosticsBuffer& buffer,
                  std::size_t& warning_count,
                  const hiko_u::PackageWarningKind kind,
                  const hiko_u::PackageValidationStage stage,
                  const std::string_view code,
                  const std::string_view message) noexcept {
  if (warning_count < kValidationWarningCapacity) {
    buffer.warnings[warning_count++] = {
        kind,
        stage,
        code,
        message,
    };
  }
}

void mark_pass(bool ok, const hiko_u::PackageValidationStage stage,
               std::uint64_t& passed, std::uint64_t& failed) noexcept {
  if (ok) {
    passed |= stage_flag(stage);
  } else {
    failed |= stage_flag(stage);
  }
}

template <typename T>
const RegisteredCapabilityRecord<T>* find_capability_record(
    const universal::Span<const RegisteredCapabilityRecord<T>>& records,
    const T value) noexcept {
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].value == value) {
      return &records.data[index];
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Stage 1 — schema_validation
//
// Reject vocabulary:
//   package_schema_version_unsupported, architecture_kind_unknown,
//   architecture_params_invalid, tensor_role_unknown
// ---------------------------------------------------------------------------
bool stage_schema_validation(const hiko_u::PackageDescriptor& descriptor,
                             ValidationDiagnosticsBuffer& buffer,
                             std::size_t& diagnostic_count) noexcept {
  bool ok = true;
  if (descriptor.identity.package_schema_version !=
      kSupportedPackageSchemaVersion) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::SchemaVersion,
               codes::kPackageSchemaVersionUnsupported,
               "package schema version is not the 0.1.0 supported value");
  }
  if (descriptor.identity.package_kind == hiko_u::PackageKind::Unknown) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::SchemaVersion,
               codes::kArchitectureKindUnknown,
               "package kind is unknown vocabulary");
  }
  if (descriptor.execution.mode == hiko_u::PackageExecutionMode::Unknown) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::SchemaVersion,
               codes::kArchitectureKindUnknown,
               "package execution mode is unknown vocabulary");
  }
  if (descriptor.execution.mode ==
          hiko_u::PackageExecutionMode::RegisteredArchitecture &&
      descriptor.execution.architecture_id.empty()) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::SchemaVersion,
               codes::kArchitectureParamsInvalid,
               "registered-architecture packages must declare a non-empty architecture id");
  }
  if (descriptor.identity.package_id.empty()) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::SchemaVersion,
               codes::kArchitectureParamsInvalid,
               "package descriptor must declare a non-empty package id");
  }
  // tensor_role_unknown — at the public descriptor surface, tensor slots are
  // anonymous spans of dtype/layout entries. Reject the empty case here; the
  // per-tensor slot mismatches are caught in stage 4.
  if (descriptor.capabilities.dtypes.size == 0 ||
      descriptor.capabilities.layouts.size == 0) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::SchemaVersion,
               codes::kTensorRoleUnknown,
               "package capabilities must declare at least one tensor dtype and layout");
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Stage 2 — architecture_binding
//
// Reject vocabulary:
//   architecture_kind_not_registered
// ---------------------------------------------------------------------------
bool stage_architecture_binding(
    const hiko_u::PackageDescriptor& descriptor,
    const RegisteredArchitectureRecord*& out_record,
    ValidationDiagnosticsBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  out_record = nullptr;
  if (descriptor.execution.mode !=
      hiko_u::PackageExecutionMode::RegisteredArchitecture) {
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::ArchitectureRegistration,
               codes::kArchitectureKindNotRegistered,
               "Hikoboshi 0.1 supports the registered_architecture execution mode only");
    return false;
  }
  out_record = find_architecture(descriptor.execution.architecture_id);
  if (out_record == nullptr) {
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::ArchitectureRegistration,
               codes::kArchitectureKindNotRegistered,
               "package architecture id is not present in architecture_registry()");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Stage 3 — capability_handshake
//
// Reject vocabulary:
//   capability_handshake_failed, backend_unavailable,
//   dtype_unsupported_on_backend, layout_unsupported_on_backend
//
// Per the decision report's fallback_policy, an architecture supported on
// host but not on a requested device should emit a host-fallback warning
// instead of rejecting. Hikoboshi 0.1.0 ships only the scalar CPU backend,
// so any non-CpuScalar requirement that the registry exposes as
// unimplemented becomes a warning when the CpuScalar fallback is also
// declared, and a hard rejection otherwise.
// ---------------------------------------------------------------------------
bool stage_capability_handshake(
    const hiko_u::PackageDescriptor& descriptor,
    ValidationDiagnosticsBuffer& buffer,
    std::size_t& diagnostic_count,
    std::size_t& warning_count) noexcept {
  bool ok = true;

  const universal::Span<const RegisteredCapabilityRecord<hiko_u::DataType>>
      dtypes = dtypes_registry();
  const universal::Span<
      const RegisteredCapabilityRecord<hiko_u::PackageTensorLayout>>
      layouts = layouts_registry();
  const universal::Span<
      const RegisteredCapabilityRecord<hiko_u::PackageBackendRequirement>>
      backends = backends_registry();

  if (descriptor.capabilities.backends.size == 0 ||
      descriptor.execution.backend_requirements.size == 0) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::WorkflowCompatibility,
               codes::kCapabilityHandshakeFailed,
               "package must declare at least one backend requirement");
  }

  bool has_host_fallback = false;
  for (std::size_t index = 0; index < descriptor.capabilities.backends.size;
       ++index) {
    const hiko_u::PackageBackendRequirement backend =
        descriptor.capabilities.backends.data[index];
    if (backend == hiko_u::PackageBackendRequirement::CpuScalar) {
      has_host_fallback = true;
    }
  }
  for (std::size_t index = 0; index < descriptor.execution.backend_requirements.size;
       ++index) {
    const hiko_u::PackageBackendRequirement backend =
        descriptor.execution.backend_requirements.data[index];
    if (backend == hiko_u::PackageBackendRequirement::CpuScalar) {
      has_host_fallback = true;
    }
  }

  // Backend availability + per-dtype / per-layout check.
  for (std::size_t index = 0; index < descriptor.capabilities.backends.size;
       ++index) {
    const hiko_u::PackageBackendRequirement backend =
        descriptor.capabilities.backends.data[index];
    const auto* record = find_capability_record(backends, backend);
    if (record == nullptr) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::WorkflowCompatibility,
                 codes::kBackendUnavailable,
                 "package declares a backend that is not in backends_registry()");
    } else if (!record->implemented_in_0_1_0) {
      if (has_host_fallback) {
        emit_warning(buffer, warning_count,
                     hiko_u::PackageWarningKind::UnsupportedReservedCapability,
                     hiko_u::PackageValidationStage::WorkflowCompatibility,
                     codes::kBackendUnavailable,
                     "requested backend not available in this build; CpuScalar host fallback declared and selected");
      } else {
        ok = false;
        emit_error(buffer, diagnostic_count,
                   hiko_u::PackageValidationStage::WorkflowCompatibility,
                   codes::kBackendUnavailable,
                   "package backend is not implemented in Hikoboshi 0.1.0 and no host fallback is declared");
      }
    }
  }
  for (std::size_t index = 0;
       index < descriptor.execution.backend_requirements.size; ++index) {
    const hiko_u::PackageBackendRequirement backend =
        descriptor.execution.backend_requirements.data[index];
    const auto* record = find_capability_record(backends, backend);
    if (record == nullptr) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::WorkflowCompatibility,
                 codes::kBackendUnavailable,
                 "package execution declares a backend that is not in backends_registry()");
    } else if (!record->implemented_in_0_1_0 && !has_host_fallback) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::WorkflowCompatibility,
                 codes::kBackendUnavailable,
                 "package execution backend is not implemented and no host fallback is declared");
    }
  }

  for (std::size_t index = 0; index < descriptor.capabilities.dtypes.size;
       ++index) {
    const hiko_u::DataType dtype = descriptor.capabilities.dtypes.data[index];
    const auto* record = find_capability_record(dtypes, dtype);
    if (record == nullptr || !record->implemented_in_0_1_0) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::WorkflowCompatibility,
                 codes::kDtypeUnsupportedOnBackend,
                 "package dtype is not implemented on the scalar CPU backend");
    }
  }
  for (std::size_t index = 0; index < descriptor.capabilities.layouts.size;
       ++index) {
    const hiko_u::PackageTensorLayout layout =
        descriptor.capabilities.layouts.data[index];
    const auto* record = find_capability_record(layouts, layout);
    if (record == nullptr || !record->implemented_in_0_1_0) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::WorkflowCompatibility,
                 codes::kLayoutUnsupportedOnBackend,
                 "package layout is not implemented on the scalar CPU backend");
    }
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Stage 4 — tensor_table_validation
//
// Reject vocabulary:
//   tensor_slot_unknown, tensor_slot_shape_mismatch,
//   tensor_slot_dtype_mismatch, tensor_slot_layout_mismatch,
//   tensor_required_slot_missing
//
// The public PackageDescriptor surface exposes tensor descriptors only via
// `capabilities.dtypes` and `capabilities.layouts`. Per-tensor slot,
// shape, and checksum validation runs in the provider-specific path
// (e.g. `validate_mpnn64_generated_tensors`) against the embedded blob.
// At validation_core's tier we ensure those declared dtype/layout values
// are registered + implemented in 0.1.0 and that the descriptor declares
// a `compatibility_views.weights.view` (the public compatibility surface
// that holds the prepared tensor table).
// ---------------------------------------------------------------------------
bool stage_tensor_table_validation(
    const hiko_u::PackageDescriptor& descriptor,
    const RegisteredArchitectureRecord* architecture_record,
    ValidationDiagnosticsBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = true;

  const universal::Span<const RegisteredCapabilityRecord<hiko_u::DataType>>
      dtypes = dtypes_registry();
  const universal::Span<
      const RegisteredCapabilityRecord<hiko_u::PackageTensorLayout>>
      layouts = layouts_registry();

  if (descriptor.capabilities.dtypes.size == 0) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
               codes::kTensorRequiredSlotMissing,
               "package capabilities must list at least one tensor dtype slot");
  }
  if (descriptor.capabilities.layouts.size == 0) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
               codes::kTensorRequiredSlotMissing,
               "package capabilities must list at least one tensor layout slot");
  }

  for (std::size_t index = 0; index < descriptor.capabilities.dtypes.size;
       ++index) {
    const hiko_u::DataType dtype = descriptor.capabilities.dtypes.data[index];
    const auto* record = find_capability_record(dtypes, dtype);
    if (record == nullptr) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                 codes::kTensorSlotUnknown,
                 "package declares a tensor dtype that is not in dtypes_registry()");
    } else if (!record->implemented_in_0_1_0) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                 codes::kTensorSlotDtypeMismatch,
                 "package declares a tensor dtype that is not implemented in Hikoboshi 0.1.0");
    }
  }
  for (std::size_t index = 0; index < descriptor.capabilities.layouts.size;
       ++index) {
    const hiko_u::PackageTensorLayout layout =
        descriptor.capabilities.layouts.data[index];
    const auto* record = find_capability_record(layouts, layout);
    if (record == nullptr) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                 codes::kTensorSlotUnknown,
                 "package declares a tensor layout that is not in layouts_registry()");
    } else if (!record->implemented_in_0_1_0) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                 codes::kTensorSlotLayoutMismatch,
                 "package declares a tensor layout that is not implemented in Hikoboshi 0.1.0");
    }
  }

  if (descriptor.compatibility_views.weights.view == nullptr) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
               codes::kTensorRequiredSlotMissing,
               "package must declare a compatibility tensor table view");
  }
  // The architecture record describes which tensor families an architecture
  // composes (via its required_module_op_ids); if the record is missing we
  // already rejected at stage 2 and this stage runs against the descriptor's
  // self-declared capabilities only.
  (void)architecture_record;
  return ok;
}

// ---------------------------------------------------------------------------
// Stage 5 — prepared_state_plan_check
//
// Reject vocabulary: prepared_state_plan_invalid
// ---------------------------------------------------------------------------
bool stage_prepared_state_plan_check(
    const hiko_u::PackageDescriptor& descriptor,
    const RegisteredArchitectureRecord* architecture_record,
    ValidationDiagnosticsBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  if (architecture_record == nullptr) {
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::PreparedStateBuild,
               codes::kPreparedStatePlanInvalid,
               "cannot build prepared-state plan without a registered architecture record");
    return false;
  }
  if (architecture_record->prepared_state_kind.empty()) {
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::PreparedStateBuild,
               codes::kPreparedStatePlanInvalid,
               "architecture record declares an empty prepared-state kind");
    return false;
  }
  if (descriptor.compatibility_views.weights.view == nullptr) {
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::PreparedStateBuild,
               codes::kPreparedStatePlanInvalid,
               "prepared-state plan requires a non-null compatibility weights view");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Stage 6 — scoring_binding
//
// Reject vocabulary:
//   scoring_kind_not_registered, scoring_input_incompatible_with_architecture
// ---------------------------------------------------------------------------
bool stage_scoring_binding(const hiko_u::PackageDescriptor& descriptor,
                           ValidationDiagnosticsBuffer& buffer,
                           std::size_t& diagnostic_count,
                           bool& scoring_method_ok_out,
                           bool& scoring_semantics_ok_out) noexcept {
  bool method_ok = true;
  bool semantics_ok = true;
  const RegisteredScoringRecord* scoring_record =
      find_scoring(descriptor.scoring.method);
  if (scoring_record == nullptr) {
    method_ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::ScoringMethod,
               codes::kScoringKindNotRegistered,
               "package scoring method is not present in scoring_registry()");
  } else {
    // Score-input compatibility: every declared scoring input must be one of
    // the inputs the registered scoring record accepts.
    if (descriptor.scoring.inputs.size == 0) {
      method_ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::ScoringMethod,
                 codes::kScoringInputIncompatibleWithArchitecture,
                 "package scoring descriptor must declare at least one input");
    }
    for (std::size_t index = 0; index < descriptor.scoring.inputs.size;
         ++index) {
      const hiko_u::ScoreInputKind input = descriptor.scoring.inputs.data[index];
      bool found = false;
      for (std::size_t inner = 0; inner < scoring_record->inputs.size; ++inner) {
        if (scoring_record->inputs.data[inner] == input) {
          found = true;
          break;
        }
      }
      if (!found) {
        method_ok = false;
        emit_error(buffer, diagnostic_count,
                   hiko_u::PackageValidationStage::ScoringMethod,
                   codes::kScoringInputIncompatibleWithArchitecture,
                   "package scoring input is not accepted by the registered scoring method");
      }
    }
    // Output dtype must match what the scoring record emits.
    if (descriptor.scoring.semantics.dtype != scoring_record->output_dtype) {
      semantics_ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::ScoreMatrixSemantics,
                 codes::kScoringInputIncompatibleWithArchitecture,
                 "package scoring semantics dtype does not match the registered output dtype");
    }
    // Score method coherence between scoring descriptor and semantics.
    if (descriptor.scoring.semantics.method != descriptor.scoring.method) {
      semantics_ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::ScoreMatrixSemantics,
                 codes::kScoringInputIncompatibleWithArchitecture,
                 "package scoring semantics method tag does not match scoring descriptor method");
    }
  }
  scoring_method_ok_out = method_ok;
  scoring_semantics_ok_out = semantics_ok;
  return method_ok && semantics_ok;
}

// ---------------------------------------------------------------------------
// Stage 7 — alignment_binding
//
// Reject vocabulary:
//   alignment_kind_not_registered, gap_family_not_supported_by_alignment
// ---------------------------------------------------------------------------
bool stage_alignment_binding(const hiko_u::PackageDescriptor& descriptor,
                             ValidationDiagnosticsBuffer& buffer,
                             std::size_t& diagnostic_count,
                             bool& alignment_ok_out,
                             bool& gap_ok_out) noexcept {
  bool alignment_ok = true;
  bool gap_ok = true;
  const RegisteredAlignmentRecord* alignment_record =
      find_alignment(descriptor.alignment.algorithm);
  if (alignment_record == nullptr) {
    alignment_ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::AlignmentAlgorithm,
               codes::kAlignmentKindNotRegistered,
               "package alignment algorithm is not present in alignment_registry()");
  } else {
    const auto supports_gap_model = [&](const hiko_u::GapModel model) noexcept {
      for (std::size_t index = 0;
           index < alignment_record->gap_families_supported.size; ++index) {
        if (alignment_record->gap_families_supported.data[index] == model) {
          return true;
        }
      }
      return false;
    };
    if (!supports_gap_model(descriptor.gaps.model) ||
        !supports_gap_model(descriptor.soft_gaps.model)) {
      gap_ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::GapModelDefaults,
                 codes::kGapFamilyNotSupportedByAlignment,
                 "package gap families are not supported by the registered alignment algorithm");
    }
  }
  alignment_ok_out = alignment_ok;
  gap_ok_out = gap_ok;
  return alignment_ok && gap_ok;
}

// ---------------------------------------------------------------------------
// Stage 8 — io_contract_binding
//
// Reject vocabulary: io_contract_incompatible
// ---------------------------------------------------------------------------
bool stage_io_contract_binding(
    const hiko_u::PackageDescriptor& descriptor,
    const RegisteredArchitectureRecord* architecture_record,
    ValidationDiagnosticsBuffer& buffer,
    std::size_t& diagnostic_count,
    bool& input_route_ok_out,
    bool& preprocessing_ok_out) noexcept {
  bool input_route_ok = true;
  bool preprocessing_ok = true;
  if (descriptor.inputs.routes.size == 0) {
    input_route_ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::InputRoute,
               codes::kIoContractIncompatible,
               "package descriptor must declare at least one input route");
  }
  if (architecture_record != nullptr &&
      architecture_record->io_contract != nullptr) {
    const universal::Span<const hiko_u::PackageInputKind>& contract_routes =
        architecture_record->io_contract->routes;
    for (std::size_t index = 0; index < descriptor.inputs.routes.size;
         ++index) {
      const hiko_u::PackageInputKind route = descriptor.inputs.routes.data[index];
      bool route_in_contract = false;
      for (std::size_t inner = 0; inner < contract_routes.size; ++inner) {
        if (contract_routes.data[inner] == route) {
          route_in_contract = true;
          break;
        }
      }
      if (!route_in_contract) {
        input_route_ok = false;
        emit_error(buffer, diagnostic_count,
                   hiko_u::PackageValidationStage::InputRoute,
                   codes::kIoContractIncompatible,
                   "package input route is not declared by the architecture IO contract");
      }
    }
  }
  // capability_descriptor.input_routes should mirror inputs.routes.
  for (std::size_t index = 0;
       index < descriptor.capabilities.input_routes.size; ++index) {
    bool found = false;
    for (std::size_t inner = 0; inner < descriptor.inputs.routes.size;
         ++inner) {
      if (descriptor.inputs.routes.data[inner] ==
          descriptor.capabilities.input_routes.data[index]) {
        found = true;
        break;
      }
    }
    if (!found) {
      input_route_ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::InputRoute,
                 codes::kIoContractIncompatible,
                 "package capabilities declare an input route that is not in the descriptor inputs");
    }
  }

  const universal::Span<
      const RegisteredCapabilityRecord<hiko_u::PackagePreprocessingCapability>>
      preprocessing = preprocessing_kinds_registry();
  for (std::size_t index = 0;
       index < descriptor.capabilities.preprocessing.size; ++index) {
    const hiko_u::PackagePreprocessingCapability capability =
        descriptor.capabilities.preprocessing.data[index];
    const auto* record = find_capability_record(preprocessing, capability);
    if (record == nullptr || !record->implemented_in_0_1_0) {
      preprocessing_ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::PreprocessingCapabilities,
                 codes::kIoContractIncompatible,
                 "package preprocessing capability is not implemented in Hikoboshi 0.1.0");
    }
  }

  // Output-kind cross-checks fold into the legacy provider path that emits
  // `unsupported_output_kind` at `WorkflowCompatibility`; keeping that single
  // diagnostic avoids duplicate emissions and bounds the per-call diagnostic
  // count inside `PackageValidationBuffer`'s capacity.

  input_route_ok_out = input_route_ok;
  preprocessing_ok_out = preprocessing_ok;
  return input_route_ok && preprocessing_ok;
}

// ---------------------------------------------------------------------------
// Stage 9 — provenance_check
//
// Reject vocabulary:
//   payload_checksum_mismatch, tensor_checksum_mismatch
//
// Provider-side checks (SHA-256 over embedded bytes) live in the package's
// bridge TU; at validation_core we ensure the manifest-declared checksums
// are consistent with the descriptor's compatibility view metadata and
// with the provenance record the caller passed.
// ---------------------------------------------------------------------------
bool stage_provenance_check(const hiko_u::PackageDescriptor& descriptor,
                            const PackageProvenance& provenance,
                            const universal::Span<const std::uint8_t> payload,
                            ValidationDiagnosticsBuffer& buffer,
                            std::size_t& diagnostic_count) noexcept {
  bool ok = true;
  if (provenance.payload_checksum.empty()) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::StorageChecksum,
               codes::kPayloadChecksumMismatch,
               "package provenance must declare a non-empty payload checksum");
  }
  if (provenance.checksum_algorithm.empty()) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::StorageChecksum,
               codes::kPayloadChecksumMismatch,
               "package provenance must declare a non-empty checksum algorithm");
  }
  if (payload.size == 0) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::StorageChecksum,
               codes::kPayloadChecksumMismatch,
               "package provenance payload span must be non-empty");
  }
  if (descriptor.compatibility_views.weights.view != nullptr &&
      !descriptor.compatibility_views.weights.view->metadata.checksum.empty() &&
      !provenance.payload_checksum.empty() &&
      descriptor.compatibility_views.weights.view->metadata.checksum !=
          provenance.payload_checksum) {
    ok = false;
    emit_error(buffer, diagnostic_count,
               hiko_u::PackageValidationStage::StorageChecksum,
               codes::kPayloadChecksumMismatch,
               "package compatibility view checksum does not match provenance payload checksum");
  }
  for (std::size_t index = 0; index < provenance.tensor_checksums.size;
       ++index) {
    const TensorChecksumView& entry =
        provenance.tensor_checksums.data[index];
    if (entry.tensor_name.empty() || entry.checksum.empty()) {
      ok = false;
      emit_error(buffer, diagnostic_count,
                 hiko_u::PackageValidationStage::StorageChecksum,
                 codes::kTensorChecksumMismatch,
                 "per-tensor provenance entry must declare a non-empty name and checksum");
    }
  }
  return ok;
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage 10 — validation_result
// Assemble the final PackageValidationReport POD.
// ---------------------------------------------------------------------------
universal::PackageValidationReport validate_package(
    const universal::PackageDescriptor& descriptor,
    const PackageProvenance& provenance,
    const universal::Span<const std::uint8_t> payload,
    ValidationDiagnosticsBuffer& buffer) noexcept {
  std::size_t diagnostic_count = 0;
  std::size_t warning_count = 0;
  std::uint64_t passed = 0;
  std::uint64_t failed = 0;

  // Stage 1 — schema_validation.
  const bool schema_ok =
      stage_schema_validation(descriptor, buffer, diagnostic_count);
  mark_pass(schema_ok, hiko_u::PackageValidationStage::SchemaVersion, passed,
            failed);

  // Stage 2 — architecture_binding.
  const RegisteredArchitectureRecord* architecture_record = nullptr;
  const bool architecture_ok = stage_architecture_binding(
      descriptor, architecture_record, buffer, diagnostic_count);
  mark_pass(architecture_ok,
            hiko_u::PackageValidationStage::ArchitectureRegistration, passed,
            failed);

  // Stage 3 — capability_handshake.
  const bool capability_ok = stage_capability_handshake(
      descriptor, buffer, diagnostic_count, warning_count);

  // Stage 4 — tensor_table_validation.
  const bool tensor_ok = stage_tensor_table_validation(
      descriptor, architecture_record, buffer, diagnostic_count);
  mark_pass(tensor_ok,
            hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes, passed,
            failed);

  // Stage 5 — prepared_state_plan_check.
  // The plan check itself is decided here, but its `PreparedStateBuild` flag
  // is finalized at the end so it reflects the full pipeline outcome.
  const bool prepared_state_local_ok = stage_prepared_state_plan_check(
      descriptor, architecture_record, buffer, diagnostic_count);

  // Stage 6 — scoring_binding.
  bool scoring_method_ok = true;
  bool scoring_semantics_ok = true;
  stage_scoring_binding(descriptor, buffer, diagnostic_count,
                        scoring_method_ok, scoring_semantics_ok);
  mark_pass(scoring_method_ok, hiko_u::PackageValidationStage::ScoringMethod,
            passed, failed);
  mark_pass(scoring_semantics_ok,
            hiko_u::PackageValidationStage::ScoreMatrixSemantics, passed, failed);

  // Stage 7 — alignment_binding.
  bool alignment_ok = true;
  bool gap_ok = true;
  stage_alignment_binding(descriptor, buffer, diagnostic_count, alignment_ok,
                          gap_ok);
  mark_pass(alignment_ok, hiko_u::PackageValidationStage::AlignmentAlgorithm,
            passed, failed);
  mark_pass(gap_ok, hiko_u::PackageValidationStage::GapModelDefaults, passed,
            failed);

  // Stage 8 — io_contract_binding.
  bool input_route_ok = true;
  bool preprocessing_ok = true;
  stage_io_contract_binding(descriptor, architecture_record, buffer,
                            diagnostic_count, input_route_ok, preprocessing_ok);
  mark_pass(input_route_ok, hiko_u::PackageValidationStage::InputRoute, passed,
            failed);
  mark_pass(preprocessing_ok,
            hiko_u::PackageValidationStage::PreprocessingCapabilities, passed,
            failed);
  // Capability handshake folds into WorkflowCompatibility; OR with output-kind
  // failures that stage_io_contract_binding may have flipped.
  mark_pass(capability_ok && input_route_ok,
            hiko_u::PackageValidationStage::WorkflowCompatibility, passed, failed);

  // Stage 9 — provenance_check.
  const bool provenance_ok = stage_provenance_check(
      descriptor, provenance, payload, buffer, diagnostic_count);
  mark_pass(provenance_ok, hiko_u::PackageValidationStage::StorageChecksum,
            passed, failed);

  // Stage 5 finalization: prepared-state build succeeds only when the local
  // plan check passed AND every other stage that contributes to plan
  // validity also passed.
  const bool composed_prepared_state_ok =
      prepared_state_local_ok && schema_ok && architecture_ok && tensor_ok &&
      provenance_ok;
  mark_pass(composed_prepared_state_ok,
            hiko_u::PackageValidationStage::PreparedStateBuild, passed, failed);

  const bool overall_ok = failed == 0;
  return {
      overall_ok ? universal::PackageHandle{nullptr, &descriptor}
                 : universal::PackageHandle{nullptr, nullptr},
      {buffer.diagnostics, diagnostic_count},
      {buffer.warnings, warning_count},
      passed,
      failed,
      overall_ok,
  };
}

}  // namespace hikoboshi::dispatch::registry

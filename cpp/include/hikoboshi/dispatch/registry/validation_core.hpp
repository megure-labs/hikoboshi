#ifndef HIKOBOSHI_DISPATCH_REGISTRY_VALIDATION_CORE_HPP
#define HIKOBOSHI_DISPATCH_REGISTRY_VALIDATION_CORE_HPP

/// @file
/// Shared ten-stage package validator used by all Hikoboshi package providers.
///
/// The model/scoring package decision report
/// (`docs/architecture/hikoboshi_model_package_execution_schema_claude.json`
/// at `capability_validation_recommendation.ten_stage_pipeline`) specs a
/// ten-stage validator that every compiled or externally loaded package
/// must walk through. This header exposes the entry point and the small
/// caller-owned diagnostic buffer those stages write into; the rejection
/// vocabulary lives in `hikoboshi/universal/package_validation_codes.hpp`.
///
/// The validator is allocation-free: it consults the closed static
/// registries (`architecture_registry()`, `scoring_registry()`,
/// `alignment_registry()`, plus the capability-axis registries) and writes
/// into the caller-owned `ValidationDiagnosticsBuffer`. The returned
/// `PackageValidationReport` references that buffer's populated prefix.
///
/// The ten conceptual stages map onto the existing
/// `PackageValidationStage` enum bits as follows:
///
/// | Conceptual stage           | `passed_stage_flags` bits set on pass     |
/// |----------------------------|-------------------------------------------|
/// | 1. schema_validation       | `SchemaVersion`                           |
/// | 2. architecture_binding    | `ArchitectureRegistration`                |
/// | 3. capability_handshake    | `WorkflowCompatibility`                   |
/// | 4. tensor_table_validation | `TensorTableRolesShapesDtypes`            |
/// | 5. prepared_state_plan     | `PreparedStateBuild`                      |
/// | 6. scoring_binding         | `ScoringMethod` + `ScoreMatrixSemantics`  |
/// | 7. alignment_binding       | `AlignmentAlgorithm` + `GapModelDefaults` |
/// | 8. io_contract_binding     | `InputRoute` + `PreprocessingCapabilities`|
/// | 9. provenance_check        | `StorageChecksum`                         |
/// | 10. validation_result      | terminal: produces the report POD         |

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::dispatch::registry {

/// Maximum diagnostics the staged validator may emit per call.
///
/// The capacity is sized so worst-case multi-stage failures from the
/// schema-valid-but-unsupported fixtures used by tests
/// (substitution-matrix descriptor, mutated MPNN-64 fixtures) fit without
/// truncation while every stage may emit both a canonical and any
/// provider-supplied legacy diagnostic.
inline constexpr std::size_t kValidationDiagnosticCapacity = 24;

/// Maximum warnings the staged validator may emit per call.
inline constexpr std::size_t kValidationWarningCapacity = 8;

/// Caller-owned diagnostic + warning buffer used by `validate_package`.
///
/// The buffer is plain inline storage; the validator writes into the
/// populated prefix and the returned report exposes spans into it. The
/// buffer must outlive the returned report.
struct ValidationDiagnosticsBuffer {
  universal::PackageValidationDiagnostic
      diagnostics[kValidationDiagnosticCapacity];
  universal::PackageWarning warnings[kValidationWarningCapacity];
};

/// Per-tensor checksum entry consumed by stage 9 (`provenance_check`).
struct TensorChecksumView {
  std::string_view tensor_name;
  std::string_view checksum;
};

/// Provenance facts the validator's stage 9 (`provenance_check`) compares
/// against the package descriptor and embedded blob metadata.
///
/// For compiled packages the values come from the package manifest; for
/// external packages they come from the loader-parsed manifest sidecar.
/// All strings reference storage owned by the caller and must outlive the
/// validation call.
struct PackageProvenance {
  std::string_view payload_checksum;
  std::string_view checksum_algorithm;
  std::string_view source_identifier;
  universal::Span<const TensorChecksumView> tensor_checksums;
};

/// Walk the ten-stage validation pipeline for a package descriptor.
///
/// The function consults the closed dispatch registries for architecture,
/// scoring, alignment, dtype, layout, backend, input-route, and
/// preprocessing-capability metadata, then accumulates pass/fail flags
/// and structured diagnostics into the caller-owned `buffer`. The
/// returned report's `accepted_handle` is `{nullptr, &descriptor}` on
/// successful validation and `{nullptr, nullptr}` when any error
/// diagnostic was emitted; callers compose a real opaque pointer when
/// they hand the accepted descriptor back to consumers.
///
/// The validator does not allocate. Each stage either reads static
/// registry tables or compares scalar fields; diagnostics and warnings
/// are written into the inline `buffer` storage.
universal::PackageValidationReport validate_package(
    const universal::PackageDescriptor& descriptor,
    const PackageProvenance& provenance,
    universal::Span<const std::uint8_t> payload,
    ValidationDiagnosticsBuffer& buffer) noexcept;

}  // namespace hikoboshi::dispatch::registry

#endif  // HIKOBOSHI_DISPATCH_REGISTRY_VALIDATION_CORE_HPP

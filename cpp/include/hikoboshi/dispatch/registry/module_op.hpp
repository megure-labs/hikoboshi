#ifndef HIKOBOSHI_DISPATCH_REGISTRY_MODULE_OP_HPP
#define HIKOBOSHI_DISPATCH_REGISTRY_MODULE_OP_HPP

/// @file
/// Closed registry of Hikoboshi compound module-ops.
///
/// One record per registered compound module that composes primitive ops into
/// a higher-level building block (encoder, similarity, layer, alignment
/// wrapper). Architectures register a tagged list of module-op ids they
/// require via `RegisteredArchitectureRecord::required_module_op_ids`; each
/// module-op declares the primitive-op ids it requires via
/// `required_primitive_op_ids`. Cross-reference validation walks the
/// required-primitive lists at first registry access and confirms every id
/// resolves in `primitive_op_registry()`.
///
/// `ParityMode` is shared with the primitive-op registry; module-ops inherit
/// the same numeric-contract axis so a future graph-IR consumer can ask the
/// same `Strict` / `Fast` question of a compound module that it asks of a
/// primitive op.

#include <cstdint>
#include <string_view>

#include <hikoboshi/dispatch/registry/primitive_op.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor_role.hpp>

namespace hikoboshi::dispatch::registry {

/// Family that a compound module-op belongs to.
///
/// Used by validators and graph-IR consumers to filter the module-op registry
/// by role (e.g. "list available encoders" or "list available alignment
/// wrappers").
enum class ModuleOpFamily : std::uint8_t {
  Encoder = 0,
  Similarity = 1,
  Layer = 2,
  AlignmentWrapper = 3,
};

/// Stable identity for a registered module-op.
///
/// `op_id` is the canonical dotted name (for example
/// `hikoboshi.mpnn.v1.encoder`); non-additive changes bump the version suffix.
/// `op_family` is the textual spelling of `ModuleOpFamily` for diagnostics.
/// `op_version` echoes the version suffix carried in `op_id`.
struct ModuleOpIdentity {
  std::string_view op_id;
  std::string_view op_family;
  std::string_view op_version;
};

/// Signature describing tensor inputs/outputs and the closed set of primitive
/// op ids a module-op requires.
///
/// All spans reference static storage owned by the dispatch registry TU and
/// remain valid for the lifetime of the process. `required_primitive_op_ids`
/// is the cross-reference walked by `validate_module_op_records()`.
struct ModuleOpSignature {
  universal::Span<const universal::TensorRole> inputs;
  universal::Span<const universal::TensorRole> outputs;
  universal::Span<const std::string_view> required_primitive_op_ids;
};

/// Backend and parity capabilities a module-op exposes.
///
/// `supported_backends` lists the backend families a record's
/// `dispatch_entry` is wired for in this build; the value is a subset of
/// `PackageBackendRequirement`. `supported_parity_modes` lists the per-op
/// numerical contracts the registered backends honor; `default_parity_mode`
/// names the mode used when the caller does not select one.
struct ModuleOpCapabilities {
  universal::Span<const universal::PackageBackendRequirement> supported_backends;
  universal::Span<const ParityMode> supported_parity_modes;
  ParityMode default_parity_mode;
};

/// One row in the module-op registry.
///
/// `dispatch_entry` points at a function-pointer wrapper that invokes the
/// compound module under the linked-in backend (Hikoboshi 0.1.0 has scalar
/// only). Consumers cast it to the matching call signature based on the op
/// identity. The pointer has static storage duration.
struct RegisteredModuleOpRecord {
  ModuleOpIdentity identity;
  ModuleOpFamily family;
  ModuleOpSignature signature;
  ModuleOpCapabilities capabilities;
  const void* dispatch_entry;
};

/// Return the closed set of module-op records.
///
/// The first call walks the records and validates that every
/// `required_primitive_op_ids` entry resolves in `primitive_op_registry()`;
/// the validation report is cached and exposed through
/// `module_op_registry_validation()`. Records are stable across calls within
/// a process. Hikoboshi 0.1.0 reports the five module-ops listed in the closed
/// op-set charter.
universal::Span<const RegisteredModuleOpRecord>
module_op_registry() noexcept;

/// Resolve a registered module-op by canonical id.
///
/// Returns `nullptr` if no record matches. The case-sensitive comparison
/// matches the canonical `op_id` only.
const RegisteredModuleOpRecord* find_module_op(
    std::string_view op_id) noexcept;

/// Severity of a module-op cross-reference diagnostic.
enum class ModuleOpValidationSeverity : std::uint8_t {
  Info = 0,
  Warning = 1,
  Error = 2,
};

/// Structured diagnostic emitted when a module-op declares a
/// `required_primitive_op_ids` entry that does not resolve in
/// `primitive_op_registry()`.
///
/// `code` is a stable identifier consumers can match on
/// (`module_op_required_primitive_missing`); `module_op_id` is the offending
/// record; `missing_primitive_op_id` is the unresolved id; `message` is a
/// human-readable rendering of the diagnostic.
struct ModuleOpValidationDiagnostic {
  ModuleOpValidationSeverity severity;
  std::string_view code;
  std::string_view module_op_id;
  std::string_view missing_primitive_op_id;
  std::string_view message;
};

/// Module-op cross-reference validation outcome.
///
/// `diagnostics` references storage owned either by the cached live-registry
/// report (`module_op_registry_validation()`) or by the caller-owned buffer
/// passed to `validate_module_op_records()`. `ok` is true iff no
/// Error-severity diagnostics were emitted.
struct ModuleOpValidationReport {
  universal::Span<const ModuleOpValidationDiagnostic> diagnostics;
  bool ok;
};

/// Return the cached cross-reference validation report for the live module-op
/// registry against the live primitive-op registry.
///
/// The report is computed on first call (Meyers-singleton pattern) by
/// walking `module_op_registry()` and resolving each
/// `required_primitive_op_ids` entry against `primitive_op_registry()`. The
/// returned reference is stable across calls and the diagnostics span has
/// static lifetime.
const ModuleOpValidationReport& module_op_registry_validation() noexcept;

/// Test seam: validate an arbitrary set of module-op records against the
/// live primitive-op registry.
///
/// Diagnostics are written into the caller-owned `diagnostic_buffer` up to
/// `buffer_capacity` entries; the returned report's `diagnostics` span
/// references the buffer prefix actually filled. `ok` is true iff no
/// Error-severity diagnostics were emitted (note: a small buffer that
/// truncates extra diagnostics still reports `ok = false` if any Error
/// diagnostic was recorded). Pass a buffer at least as large as the total
/// `required_primitive_op_ids` count across the records to guarantee every
/// diagnostic is captured.
ModuleOpValidationReport validate_module_op_records(
    universal::Span<const RegisteredModuleOpRecord> records,
    ModuleOpValidationDiagnostic* diagnostic_buffer,
    std::size_t buffer_capacity) noexcept;

}  // namespace hikoboshi::dispatch::registry

#endif  // HIKOBOSHI_DISPATCH_REGISTRY_MODULE_OP_HPP

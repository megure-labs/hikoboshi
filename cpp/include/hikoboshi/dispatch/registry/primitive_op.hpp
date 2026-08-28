#ifndef HIKOBOSHI_DISPATCH_REGISTRY_PRIMITIVE_OP_HPP
#define HIKOBOSHI_DISPATCH_REGISTRY_PRIMITIVE_OP_HPP

/// @file
/// Closed registry of Hikoboshi primitive ops.
///
/// One record per registered primitive op. Each record carries identity,
/// signature (tensor + parameter roles), capability metadata (supported
/// backends and parity modes), and a `dispatch_entry` pointer for the
/// currently linked-in backend. The dispatch table built in
/// `cpp/dispatch/dispatch_table.cpp` populates its typed slots from this
/// registry so the registry is the single source of truth for the closed
/// op set.
///
/// `ParityMode` is a per-op axis declaring the numeric contract a backend
/// promises. Hikoboshi 0.1.0 ships only `Strict`; `gemm-dual-mode-strict-fast`
/// adds the `Fast` mode for the GEMM family.

#include <cstdint>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor_role.hpp>

namespace hikoboshi::dispatch::registry {

/// Numerical contract advertised by a primitive op backend.
///
/// `Strict` means the backend reproduces a chosen reference implementation
/// bit-identically (e.g. PyTorch cuBLAS reduction-tree order for GEMM).
/// `Fast` means the backend stays within the public tolerance contract
/// (1e-4 max abs for GEMM) but does not promise bit identity.
///
/// Ops whose output is integer-valued or otherwise has no float-rounding
/// freedom declare `Strict` only.
enum class ParityMode : std::uint8_t {
  Strict = 0,
  Fast = 1,
};

/// Family that a primitive op belongs to.
///
/// Used by validators and `hikoboshi info ops` style diagnostics to group ops.
enum class PrimitiveOpFamily : std::uint8_t {
  Compute = 0,
  Linalg = 1,
  Alignment = 2,
};

/// Stable identity for a registered primitive op.
///
/// `op_id` is the canonical dotted name (for example `hikoboshi.gemm.nt.v1`);
/// non-additive changes bump the version suffix. `op_family` is the textual
/// spelling of `PrimitiveOpFamily` for diagnostics. `op_version` echoes the
/// version suffix carried in `op_id`.
struct PrimitiveOpIdentity {
  std::string_view op_id;
  std::string_view op_family;
  std::string_view op_version;
};

/// Signature describing the tensors and scalar parameters an op consumes.
///
/// All spans reference static storage owned by the dispatch registry TU and
/// remain valid for the lifetime of the process.
struct PrimitiveOpSignature {
  universal::Span<const universal::TensorRole> inputs;
  universal::Span<const universal::TensorRole> outputs;
  universal::Span<const universal::ParameterRole> parameters;
};

/// Backend capabilities a primitive op exposes.
///
/// `supported_backends` lists the backend families that a record's
/// `dispatch_entry` is wired for in this build; the value is a subset of
/// `PackageBackendRequirement`. `supported_parity_modes` lists the per-op
/// numerical contracts the registered backends honor; `default_parity_mode`
/// names the mode used when the caller does not select one.
struct PrimitiveOpCapabilities {
  universal::Span<const universal::PackageBackendRequirement> supported_backends;
  universal::Span<const ParityMode> supported_parity_modes;
  ParityMode default_parity_mode;
};

/// One row in the primitive op registry.
///
/// `dispatch_entry` points at the entry-wrapper function pointer registered
/// for this op under the linked-in backend (Hikoboshi 0.1.0 has scalar only).
/// Consumers cast it to the matching `DispatchTable` typedef based on the
/// op identity. The pointer has static storage duration.
struct RegisteredPrimitiveOpRecord {
  PrimitiveOpIdentity identity;
  PrimitiveOpFamily family;
  PrimitiveOpSignature signature;
  PrimitiveOpCapabilities capabilities;
  const void* dispatch_entry;
};

/// Return the closed set of primitive op records.
///
/// The returned span references storage with static lifetime. Records are
/// stable across calls within a process. Hikoboshi 0.1.0 reports 16 records
/// covering the closed primitive op set listed in
/// `docs/charters/CLOSED_OP_SET_CHARTER.md`.
universal::Span<const RegisteredPrimitiveOpRecord>
primitive_op_registry() noexcept;

/// Resolve a registered primitive op by canonical id.
///
/// Returns `nullptr` if no record matches. The case-sensitive comparison
/// matches the canonical `op_id` only.
const RegisteredPrimitiveOpRecord* find_primitive_op(
    std::string_view op_id) noexcept;

}  // namespace hikoboshi::dispatch::registry

#endif  // HIKOBOSHI_DISPATCH_REGISTRY_PRIMITIVE_OP_HPP

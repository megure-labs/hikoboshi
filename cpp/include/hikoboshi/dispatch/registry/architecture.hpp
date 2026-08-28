#ifndef HIKOBOSHI_DISPATCH_REGISTRY_ARCHITECTURE_HPP
#define HIKOBOSHI_DISPATCH_REGISTRY_ARCHITECTURE_HPP

/// @file
/// Closed registry of Hikoboshi architectures.
///
/// One record per `architecture_kind`. The record binds the architecture id,
/// the module/scoring op tags it composes, the prepared-state kind it
/// produces, a builder hook, and pointers to the architecture's capability
/// descriptor and IO contract. Hikoboshi 0.1.0 registers `hikoboshi_mpnn_v1`;
/// future architectures (e.g. `hikoboshi_esm2_v1`) register additional records.

#include <cstdint>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::dispatch::registry {

/// Closed tag for Hikoboshi architectures.
///
/// Reserved values are descriptor vocabulary only; only registered records are
/// considered live. Adding a new architecture must register it through this
/// enum and `architecture_registry()`.
enum class ArchitectureKind : std::uint8_t {
  Mpnn64 = 0,
  Esm2_8m = 1,
};

/// Prepared-state builder hook for a registered architecture.
///
/// The builder constructs architecture-specific prepared state from a
/// `PackageHandle`. Hikoboshi 0.1.0 uses a null builder for `hikoboshi_mpnn_v1`
/// because compiled-package construction stays in the weights TU; later
/// packets wire real builders through this hook.
using ArchitectureBuilderFn = void (*)(const universal::PackageHandle& package);

/// One row in the architecture registry.
///
/// All pointer and span fields reference static storage owned by the dispatch
/// registry TU and remain valid for the lifetime of the process.
///
/// `required_module_op_ids` is the closed list of compound module-op ids the
/// architecture composes; each id must resolve in `module_op_registry()`.
/// Together with `RegisteredModuleOpRecord::required_primitive_op_ids` this
/// makes the architecture / module-op / primitive-op chain fully explicit at
/// the registry level: an architecture is a tagged list of module-ops, each
/// module-op is a tagged list of primitive-ops, each primitive-op is a
/// registered kernel with typed signature, backend, and parity coverage.
struct RegisteredArchitectureRecord {
  ArchitectureKind kind;
  std::string_view architecture_id;
  std::string_view module_op_kind;
  std::string_view scoring_op_kind;
  std::string_view prepared_state_kind;
  ArchitectureBuilderFn builder;
  const universal::PackageCapabilities* capability_descriptor;
  const universal::PackageInputs* io_contract;
  universal::Span<const std::string_view> required_module_op_ids;
};

/// Return the closed set of architecture records.
///
/// The returned span references storage with static lifetime. Records are
/// stable across calls within a process. Hikoboshi 0.1.0 reports the
/// compiled `hikoboshi_mpnn_v1` record plus the sequence-only
/// `hikoboshi_esm2_v1` record; the ESM2-8M builder is left null in this
/// packet (descriptor-only registration) and wired up by subsequent
/// builder-population packets.
universal::Span<const RegisteredArchitectureRecord>
architecture_registry() noexcept;

/// Resolve a registered architecture by canonical id.
///
/// Returns `nullptr` if no record matches. The case-sensitive comparison
/// matches the canonical `architecture_id` only; aliases live at the package
/// layer.
const RegisteredArchitectureRecord* find_architecture(
    std::string_view architecture_id) noexcept;

}  // namespace hikoboshi::dispatch::registry

#endif  // HIKOBOSHI_DISPATCH_REGISTRY_ARCHITECTURE_HPP

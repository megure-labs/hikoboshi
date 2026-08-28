#ifndef HIKOBOSHI_DISPATCH_REGISTRY_CAPABILITY_HPP
#define HIKOBOSHI_DISPATCH_REGISTRY_CAPABILITY_HPP

/// @file
/// Closed registries for capability descriptor axes.
///
/// One registry per capability axis: input kinds, output kinds, preprocessing
/// kinds, dtypes, layouts, devices, and backends. The `implemented_in_0_1_0`
/// flag distinguishes axis values that are wired in Hikoboshi 0.1.0 from
/// reserved-vocabulary values that capability descriptors may name but that
/// must fail validation when claimed.

#include <cstdint>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::dispatch::registry {

/// One row in a closed capability axis registry.
///
/// `value` is the enum value (or device-kind tag); `name` is a stable
/// canonical string spelling for diagnostics; `implemented_in_0_1_0` is true
/// only for axis values that Hikoboshi 0.1.0 actually executes.
template <typename T>
struct RegisteredCapabilityRecord {
  T value;
  std::string_view name;
  bool implemented_in_0_1_0;
};

/// Closed tag for execution devices.
///
/// Existing `PackageBackendRequirement` distinguishes backend families. The
/// device tag is a lower-resolution view that the capability axis enumerates
/// for descriptor diagnostics.
enum class DeviceKind : std::uint8_t {
  Cpu = 0,
  Gpu = 1,
};

/// Return the closed registry of package input kinds.
universal::Span<const RegisteredCapabilityRecord<universal::PackageInputKind>>
input_kinds_registry() noexcept;

/// Return the closed registry of package output kinds.
universal::Span<const RegisteredCapabilityRecord<universal::PackageOutputKind>>
output_kinds_registry() noexcept;

/// Return the closed registry of preprocessing capabilities.
universal::Span<
    const RegisteredCapabilityRecord<universal::PackagePreprocessingCapability>>
preprocessing_kinds_registry() noexcept;

/// Return the closed registry of tensor dtypes.
universal::Span<const RegisteredCapabilityRecord<universal::DataType>>
dtypes_registry() noexcept;

/// Return the closed registry of tensor layouts.
universal::Span<
    const RegisteredCapabilityRecord<universal::PackageTensorLayout>>
layouts_registry() noexcept;

/// Return the closed registry of execution devices.
universal::Span<const RegisteredCapabilityRecord<DeviceKind>>
devices_registry() noexcept;

/// Return the closed registry of backend requirements.
universal::Span<
    const RegisteredCapabilityRecord<universal::PackageBackendRequirement>>
backends_registry() noexcept;

}  // namespace hikoboshi::dispatch::registry

#endif  // HIKOBOSHI_DISPATCH_REGISTRY_CAPABILITY_HPP

#ifndef HIKOBOSHI_WEIGHTS_PROVIDER_REGISTER_HPP
#define HIKOBOSHI_WEIGHTS_PROVIDER_REGISTER_HPP

/// @file
/// Small helpers for declaring compiled package registry records.

#include <string_view>

#include <hikoboshi/weights/provider.hpp>

namespace hikoboshi::weights {

/// Build a registry record from a package handle and availability metadata.
inline constexpr PackageRegistryRecord package_registry_record(
    const PackageHandle package,
    const bool compiled,
    const bool runtime_available,
    const std::string_view reason) noexcept {
  return {package, package.descriptor, compiled, runtime_available, reason};
}

/// Build an available compiled-package registry record.
inline constexpr PackageRegistryRecord available_compiled_package_record(
    const PackageHandle package) noexcept {
  return package_registry_record(package, true, true, std::string_view{});
}

}  // namespace hikoboshi::weights

#endif  // HIKOBOSHI_WEIGHTS_PROVIDER_REGISTER_HPP

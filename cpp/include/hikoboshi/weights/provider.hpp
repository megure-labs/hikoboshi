#ifndef HIKOBOSHI_WEIGHTS_PROVIDER_HPP
#define HIKOBOSHI_WEIGHTS_PROVIDER_HPP

/// @file
/// Public provider functions for compiled Hikoboshi model/scoring packages.

#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/manifest.hpp>

namespace hikoboshi::weights {

/// Provider-facing alias for public package handles.
using PackageHandle = universal::PackageHandle;
/// Provider-facing alias for package validation diagnostics.
using PackageValidationDiagnostic = universal::PackageValidationDiagnostic;
/// Provider-facing alias for package validation reports.
using PackageValidationReport = universal::PackageValidationReport;
/// Provider-facing alias for package warnings.
using PackageWarning = universal::PackageWarning;

/// Maximum diagnostics written into a package validation buffer.
inline constexpr std::size_t kPackageValidationDiagnosticCapacity = 16;
/// Maximum warnings written into a package validation buffer.
inline constexpr std::size_t kPackageValidationWarningCapacity = 4;

/// One compiled-package registry entry.
///
/// `compiled` reports whether the package is present in this build.
/// `runtime_available` reports whether it can be used in the current runtime
/// environment. `reason` is empty when the package is available.
struct PackageRegistryRecord {
  PackageHandle package;
  const universal::PackageDescriptor* descriptor;
  bool compiled;
  bool runtime_available;
  std::string_view reason;
};

/// Caller-owned storage for allocation-free package validation.
///
/// Provider validation writes diagnostics and warnings into this fixed buffer
/// and returns spans that reference the populated prefix.
struct PackageValidationBuffer {
  PackageValidationDiagnostic
      diagnostics[kPackageValidationDiagnosticCapacity];
  PackageWarning warnings[kPackageValidationWarningCapacity];
};

/// Return all compiled package registry records.
universal::Span<const PackageRegistryRecord> compiled_packages() noexcept;

/// Resolve a compiled package by canonical id or supported alias.
[[nodiscard]] universal::Result<PackageHandle> default_package(
    std::string_view package_id) noexcept;

/// Validate that a handle describes the compiled hikoboshi-mpnn-d64 package.
universal::PackageValidationReport validate_mpnn_d64_package(
    PackageHandle package,
    PackageValidationBuffer& buffer) noexcept;

/// Backward-compatible validator for the compiled hikoboshi-mpnn-d64 package.
universal::PackageValidationReport validate_mpnn64_package(
    PackageHandle package,
    PackageValidationBuffer& buffer) noexcept;

/// Return the compiled hikoboshi-mpnn-d64 package handle.
[[nodiscard]] universal::Result<PackageHandle> default_mpnn_d64_package() noexcept;

/// Backward-compatible package accessor for the compiled hikoboshi-mpnn-d64 package.
[[nodiscard]] universal::Result<PackageHandle> default_mpnn64_package() noexcept;

/// Return a tensor view for the compiled hikoboshi-mpnn-d64 package.
[[nodiscard]] universal::Result<universal::WeightsHandle> default_mpnn_d64() noexcept;

/// Backward-compatible tensor-view accessor for the compiled hikoboshi-mpnn-d64 package.
[[nodiscard]] universal::Result<universal::WeightsHandle> default_mpnn64() noexcept;

/// Validate that a handle describes the compiled hikoboshi-esm2-8m package.
///
/// The check threads the descriptor through the canonical staged
/// validator and the package-local tensor table verifier; on success
/// `report.accepted_handle` mirrors the input handle.
universal::PackageValidationReport validate_esm2_8m_package(
    PackageHandle package,
    PackageValidationBuffer& buffer) noexcept;

/// Return the compiled hikoboshi-esm2-8m package handle.
[[nodiscard]] universal::Result<PackageHandle> default_esm2_8m_package() noexcept;

/// Return a compatibility tensor view for the compiled hikoboshi-esm2-8m
/// package.
[[nodiscard]] universal::Result<universal::WeightsHandle> default_esm2_8m() noexcept;

}  // namespace hikoboshi::weights

#endif  // HIKOBOSHI_WEIGHTS_PROVIDER_HPP

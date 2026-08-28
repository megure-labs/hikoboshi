#ifndef HIKOBOSHI_API_VERSION_HPP
#define HIKOBOSHI_API_VERSION_HPP

/// @file
/// Public version and capability reports.

#include <string_view>

#include <hikoboshi/universal/backend.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/version.hpp>

namespace hikoboshi::api {

/// Product version report returned by the public API.
struct VersionInfo {
  std::string_view product_name = "Hikoboshi";
  universal::VersionView version = universal::kVersion;
};

/// Compile-time and runtime availability for one backend family.
struct BackendAvailability {
  bool compiled = false;
  bool runtime_available = false;
  std::string_view reason = "";
};

/// CPU backend availability report.
struct CpuBackendCapabilities {
  BackendAvailability scalar{};
  BackendAvailability sse4{};
  BackendAvailability avx2{};
  BackendAvailability avx512{};
  BackendAvailability neon{};
  BackendAvailability sve{};
};

/// GPU/backend-device availability report.
struct GpuBackendAvailability {
  BackendAvailability availability{};
  universal::Span<const std::string_view> devices{nullptr, 0};
};

/// Reserved GPU/backend families and their runtime availability.
struct GpuBackendCapabilities {
  GpuBackendAvailability cuda{};
  GpuBackendAvailability hip{};
  GpuBackendAvailability metal{};
  GpuBackendAvailability vulkan{};
  GpuBackendAvailability opencl{};
};

/// High-level workflow capabilities exposed by this build.
struct PipelineCapabilities {
  bool pairwise_alignment = false;
  bool symmetric_all_vs_all = false;
  bool structure_inputs = false;
  bool embedding_inputs = false;
};

/// Combined backend and workflow capability report.
///
/// Hikoboshi 0.1.0 reports scalar CPU as the default backend. Reserved backend
/// families may appear here as unavailable with explanatory reasons.
struct BackendCapabilities {
  CpuBackendCapabilities cpu{};
  GpuBackendCapabilities gpu{};
  PipelineCapabilities pipeline{};
  universal::Backend default_backend = universal::Backend::Scalar;
};

/// Return version metadata for the linked Hikoboshi library.
VersionInfo version_info() noexcept;
/// Return backend and workflow capabilities for the linked Hikoboshi library.
BackendCapabilities backend_capabilities() noexcept;

}  // namespace hikoboshi::api

#endif  // HIKOBOSHI_API_VERSION_HPP

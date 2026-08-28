#include <hikoboshi/api/version.hpp>

namespace hikoboshi::api {
namespace {

constexpr std::string_view kReservedBackendReason =
    "reserved backend is not compiled in Hikoboshi 0.1.0 scalar-only builds";

constexpr BackendAvailability available_backend() noexcept {
  return {true, true, ""};
}

constexpr BackendAvailability reserved_backend() noexcept {
  return {false, false, kReservedBackendReason};
}

constexpr GpuBackendAvailability reserved_gpu_backend() noexcept {
  return {reserved_backend(), {nullptr, 0}};
}

}  // namespace

VersionInfo version_info() noexcept {
  VersionInfo info{};
  info.product_name = "Hikoboshi";
  info.version = universal::kVersion;
  return info;
}

BackendCapabilities backend_capabilities() noexcept {
  BackendCapabilities capabilities{};
  capabilities.cpu.scalar = available_backend();
  capabilities.cpu.sse4 = reserved_backend();
  capabilities.cpu.avx2 = reserved_backend();
  capabilities.cpu.avx512 = reserved_backend();
  capabilities.cpu.neon = reserved_backend();
  capabilities.cpu.sve = reserved_backend();

  capabilities.gpu.cuda = reserved_gpu_backend();
  capabilities.gpu.hip = reserved_gpu_backend();
  capabilities.gpu.metal = reserved_gpu_backend();
  capabilities.gpu.vulkan = reserved_gpu_backend();
  capabilities.gpu.opencl = reserved_gpu_backend();

  capabilities.pipeline.pairwise_alignment = true;
  capabilities.pipeline.symmetric_all_vs_all = true;
  capabilities.pipeline.structure_inputs = true;
  capabilities.pipeline.embedding_inputs = true;
  capabilities.default_backend = universal::Backend::Scalar;
  return capabilities;
}

}  // namespace hikoboshi::api

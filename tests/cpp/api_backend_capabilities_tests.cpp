#include <hikoboshi/api/engine.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "api_backend_capabilities_tests: %s\n", message);
  std::exit(1);
}

constexpr bool version_is_0_1_0() noexcept {
  return hiko_u::kVersionMajor == 0 && hiko_u::kVersionMinor == 1 &&
         hiko_u::kVersionPatch == 0 && hiko_u::kVersionLabel.empty();
}

void require_0_1_0_release_lock() {
  if (!version_is_0_1_0()) {
    fail("backend capability expectations must be updated for this release");
  }
}

hiko_u::EmbeddingView one_residue_embedding(const std::vector<float>& values) {
  return {1, 1, {values.data(), values.size()}, {nullptr, 0}, {nullptr, 0}};
}

void require_embedding_pairwise_ok(hiko_u::Backend backend) {
  const std::vector<float> query = {1.0F};
  const std::vector<float> target = {1.0F};
  const hiko::Engine engine({{nullptr, nullptr}, {backend, 0}});
  const auto result = engine.pairwise(hiko::PairwiseEmbeddingRequest{
      one_residue_embedding(query),
      one_residue_embedding(target),
  });
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("Backend::Auto and Backend::Scalar must be accepted");
  }
}

void test_version_info_is_universal_version() {
  const hiko::VersionInfo version = hiko::version_info();
  if (version.product_name != "Hikoboshi" ||
      version.version.major != hiko_u::kVersionMajor ||
      version.version.minor != hiko_u::kVersionMinor ||
      version.version.patch != hiko_u::kVersionPatch ||
      version.version.label != hiko_u::kVersionLabel) {
    fail("version_info must report universal version constants");
  }
}

void test_backend_capability_flags() {
  require_0_1_0_release_lock();
  const hiko::BackendCapabilities capabilities = hiko::backend_capabilities();
  if (!capabilities.pipeline.pairwise_alignment ||
      !capabilities.pipeline.symmetric_all_vs_all ||
      !capabilities.pipeline.structure_inputs ||
      !capabilities.pipeline.embedding_inputs ||
      capabilities.default_backend != hiko_u::Backend::Scalar) {
    fail("backend_capabilities must advertise scalar-only Hikoboshi 0.1.0 support");
  }
}

void require_available_backend(
    const hiko::BackendAvailability& availability,
    const std::string_view name) {
  if (!availability.compiled || !availability.runtime_available ||
      !availability.reason.empty()) {
    std::fprintf(stderr,
                 "api_backend_capabilities_tests: %.*s must be compiled and "
                 "runtime-available without a reason\n",
                 static_cast<int>(name.size()), name.data());
    std::exit(1);
  }
}

void require_reserved_backend(
    const hiko::BackendAvailability& availability,
    const std::string_view name) {
  if (availability.compiled || availability.runtime_available ||
      availability.reason.empty()) {
    std::fprintf(stderr,
                 "api_backend_capabilities_tests: %.*s must be unavailable "
                 "with a non-empty reserved-backend reason\n",
                 static_cast<int>(name.size()), name.data());
    std::exit(1);
  }
}

void require_reserved_gpu_backend(
    const hiko::GpuBackendAvailability& capability,
    const std::string_view name) {
  require_reserved_backend(capability.availability, name);
  if (capability.devices.data != nullptr || capability.devices.size != 0U) {
    fail("reserved 0.1.0 GPU backend device lists must be empty");
  }
}

void test_backend_availability_matrix() {
  require_0_1_0_release_lock();
  const hiko::BackendCapabilities capabilities = hiko::backend_capabilities();
  require_available_backend(capabilities.cpu.scalar, "cpu.scalar");

  struct CpuExpectation {
    std::string_view name;
    hiko::BackendAvailability availability;
  };
  const CpuExpectation reserved_cpu_backends[] = {
      {"cpu.sse4", capabilities.cpu.sse4},
      {"cpu.avx2", capabilities.cpu.avx2},
      {"cpu.avx512", capabilities.cpu.avx512},
      {"cpu.neon", capabilities.cpu.neon},
      {"cpu.sve", capabilities.cpu.sve},
  };
  for (const CpuExpectation& backend : reserved_cpu_backends) {
    require_reserved_backend(backend.availability, backend.name);
  }

  struct GpuExpectation {
    std::string_view name;
    hiko::GpuBackendAvailability availability;
  };
  const GpuExpectation reserved_gpu_backends[] = {
      {"gpu.cuda", capabilities.gpu.cuda},
      {"gpu.hip", capabilities.gpu.hip},
      {"gpu.metal", capabilities.gpu.metal},
      {"gpu.vulkan", capabilities.gpu.vulkan},
      {"gpu.opencl", capabilities.gpu.opencl},
  };
  for (const GpuExpectation& backend : reserved_gpu_backends) {
    require_reserved_gpu_backend(backend.availability, backend.name);
  }
}

void test_backend_acceptance_and_rejection() {
  require_embedding_pairwise_ok(hiko_u::Backend::Auto);
  require_embedding_pairwise_ok(hiko_u::Backend::Scalar);

  const std::vector<float> values = {1.0F};
  const hiko::Engine invalid_engine(
      {{nullptr, nullptr}, {static_cast<hiko_u::Backend>(255), 0}});
  const auto invalid_result = invalid_engine.pairwise(
      hiko::PairwiseEmbeddingRequest{one_residue_embedding(values),
                                    one_residue_embedding(values)});
  if (invalid_result.status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("future backend tags must be rejected deterministically");
  }
}

}  // namespace

int main() {
  test_version_info_is_universal_version();
  test_backend_capability_flags();
  test_backend_availability_matrix();
  test_backend_acceptance_and_rejection();
  return 0;
}

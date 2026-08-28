#include <hikoboshi/dispatch/cpu_features.hpp>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace hiko_d = hikoboshi::dispatch;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "cpu_features_smoke: %s\n", message);
  std::exit(1);
}

void require(bool condition, const char* message) {
  if (!condition) {
    fail(message);
  }
}

bool any_feature_enabled(const hiko_d::CpuFeatures& features) {
  return features.sse4_1 || features.sse4_2 || features.avx || features.avx2 ||
         features.avx512f || features.avx512bw || features.avx512vl ||
         features.neon || features.sve;
}

void test_cached_reference() {
  const hiko_d::CpuFeatures& first = hiko_d::detected_cpu_features();
  const hiko_d::CpuFeatures& second = hiko_d::detected_cpu_features();
  require(&first == &second, "detected_cpu_features must return cached storage");
}

void test_pod_shape() {
  require(std::is_standard_layout<hiko_d::CpuFeatures>::value,
          "CpuFeatures must remain standard-layout");
  require(std::is_trivial<hiko_d::CpuFeatures>::value,
          "CpuFeatures must remain trivial");
}

void test_arch_consistency() {
  const hiko_d::CpuFeatures& features = hiko_d::detected_cpu_features();

  switch (features.arch) {
    case hiko_d::CpuArch::Unknown:
      require(!any_feature_enabled(features),
              "unknown CPU architecture must not report architecture features");
      break;
    case hiko_d::CpuArch::X86_64:
      require(!features.neon && !features.sve,
              "x86_64 detection must not report Arm features");
      require(!features.avx2 || features.avx,
              "usable AVX2 requires usable AVX state");
      break;
    case hiko_d::CpuArch::AArch64:
      require(!features.sse4_1 && !features.sse4_2 && !features.avx &&
                  !features.avx2 && !features.avx512f &&
                  !features.avx512bw && !features.avx512vl,
              "AArch64 detection must not report x86 features");
      break;
    default:
      fail("detected CPU architecture enum is outside the known range");
  }

#if defined(__x86_64__) || defined(_M_X64)
  require(features.arch == hiko_d::CpuArch::X86_64,
          "x86_64 builds must report CpuArch::X86_64");
#elif defined(__aarch64__) || defined(_M_ARM64)
  require(features.arch == hiko_d::CpuArch::AArch64,
          "AArch64 builds must report CpuArch::AArch64");
#endif
}

}  // namespace

int main() {
  test_pod_shape();
  test_cached_reference();
  test_arch_consistency();
  return 0;
}

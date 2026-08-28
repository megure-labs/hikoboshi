#include <hikoboshi/dispatch/cpu_features.hpp>

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#define HIKOBOSHI_DISPATCH_X86_64 1
#else
#define HIKOBOSHI_DISPATCH_X86_64 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define HIKOBOSHI_DISPATCH_AARCH64 1
#else
#define HIKOBOSHI_DISPATCH_AARCH64 0
#endif

#if defined(__has_include)
#define HIKOBOSHI_DISPATCH_HAS_INCLUDE(header) __has_include(header)
#else
#define HIKOBOSHI_DISPATCH_HAS_INCLUDE(header) 0
#endif

#if HIKOBOSHI_DISPATCH_X86_64
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

#if HIKOBOSHI_DISPATCH_AARCH64 && defined(__linux__) && \
    HIKOBOSHI_DISPATCH_HAS_INCLUDE(<sys/auxv.h>)
#define HIKOBOSHI_DISPATCH_HAVE_LINUX_AUXV 1
#include <sys/auxv.h>
#else
#define HIKOBOSHI_DISPATCH_HAVE_LINUX_AUXV 0
#endif

#if HIKOBOSHI_DISPATCH_AARCH64 && defined(__linux__) && \
    HIKOBOSHI_DISPATCH_HAS_INCLUDE(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif

namespace hikoboshi::dispatch {
namespace {

constexpr std::uint32_t cpuid_bit(unsigned bit) noexcept {
  return std::uint32_t{1} << bit;
}

#if HIKOBOSHI_DISPATCH_X86_64
bool read_cpuid(std::uint32_t leaf,
                std::uint32_t subleaf,
                std::uint32_t& eax,
                std::uint32_t& ebx,
                std::uint32_t& ecx,
                std::uint32_t& edx) noexcept {
#if defined(_MSC_VER)
  int registers[4] = {};
  __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
  eax = static_cast<std::uint32_t>(registers[0]);
  ebx = static_cast<std::uint32_t>(registers[1]);
  ecx = static_cast<std::uint32_t>(registers[2]);
  edx = static_cast<std::uint32_t>(registers[3]);
  return true;
#elif defined(__GNUC__) || defined(__clang__)
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  return true;
#else
  (void)leaf;
  (void)subleaf;
  eax = 0;
  ebx = 0;
  ecx = 0;
  edx = 0;
  return false;
#endif
}

std::uint64_t read_xcr0() noexcept {
#if defined(_MSC_VER)
  return static_cast<std::uint64_t>(_xgetbv(0));
#elif defined(__GNUC__) || defined(__clang__)
  std::uint32_t eax = 0;
  std::uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
  return 0;
#endif
}

void detect_x86_64(CpuFeatures& features) noexcept {
  features.arch = CpuArch::X86_64;

  std::uint32_t eax = 0;
  std::uint32_t ebx = 0;
  std::uint32_t ecx = 0;
  std::uint32_t edx = 0;
  if (!read_cpuid(0, 0, eax, ebx, ecx, edx)) {
    return;
  }
  const std::uint32_t max_basic_leaf = eax;

  if (max_basic_leaf >= 1 && read_cpuid(1, 0, eax, ebx, ecx, edx)) {
    features.sse4_1 = (ecx & cpuid_bit(19)) != 0;
    features.sse4_2 = (ecx & cpuid_bit(20)) != 0;

    const bool cpu_avx = (ecx & cpuid_bit(28)) != 0;
    const bool cpu_xsave = (ecx & cpuid_bit(26)) != 0;
    const bool os_xsave = (ecx & cpuid_bit(27)) != 0;
    if (cpu_xsave && os_xsave) {
      const std::uint64_t xcr0 = read_xcr0();
      const std::uint64_t xmm_ymm_state = (std::uint64_t{1} << 1) |
                                          (std::uint64_t{1} << 2);
      features.avx = cpu_avx && ((xcr0 & xmm_ymm_state) == xmm_ymm_state);

      const std::uint64_t avx512_state = xmm_ymm_state |
                                         (std::uint64_t{1} << 5) |
                                         (std::uint64_t{1} << 6) |
                                         (std::uint64_t{1} << 7);
      const bool os_avx512 = (xcr0 & avx512_state) == avx512_state;

      if (max_basic_leaf >= 7 && read_cpuid(7, 0, eax, ebx, ecx, edx)) {
        features.avx2 = features.avx && ((ebx & cpuid_bit(5)) != 0);
        features.avx512f = os_avx512 && ((ebx & cpuid_bit(16)) != 0);
        features.avx512bw = os_avx512 && ((ebx & cpuid_bit(30)) != 0);
        features.avx512vl = os_avx512 && ((ebx & cpuid_bit(31)) != 0);
      }
    }
  }
}
#endif

#if HIKOBOSHI_DISPATCH_AARCH64
void detect_aarch64(CpuFeatures& features) noexcept {
  features.arch = CpuArch::AArch64;

#if HIKOBOSHI_DISPATCH_HAVE_LINUX_AUXV && defined(AT_HWCAP)
  const unsigned long hwcap = getauxval(AT_HWCAP);
#if defined(HWCAP_ASIMD)
  features.neon = (hwcap & HWCAP_ASIMD) != 0;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  features.neon = true;
#endif
#if defined(HWCAP_SVE)
  features.sve = (hwcap & HWCAP_SVE) != 0;
#elif defined(__ARM_FEATURE_SVE)
  features.sve = true;
#endif
#else
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  features.neon = true;
#endif
#if defined(__ARM_FEATURE_SVE)
  features.sve = true;
#endif
#endif
}
#endif

CpuFeatures detect_cpu_features_uncached() noexcept {
  CpuFeatures features{};

#if HIKOBOSHI_DISPATCH_X86_64
  detect_x86_64(features);
#elif HIKOBOSHI_DISPATCH_AARCH64
  detect_aarch64(features);
#endif

  return features;
}

}  // namespace

const CpuFeatures& detected_cpu_features() noexcept {
  static const CpuFeatures features = detect_cpu_features_uncached();
  return features;
}

}  // namespace hikoboshi::dispatch

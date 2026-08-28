#ifndef HIKOBOSHI_DISPATCH_CPU_FEATURES_HPP
#define HIKOBOSHI_DISPATCH_CPU_FEATURES_HPP

namespace hikoboshi::dispatch {

enum class CpuArch {
  Unknown = 0,
  X86_64,
  AArch64,
};

struct CpuFeatures {
  CpuArch arch;
  bool sse4_1;
  bool sse4_2;
  bool avx;
  bool avx2;
  bool avx512f;
  bool avx512bw;
  bool avx512vl;
  bool neon;
  bool sve;
};

const CpuFeatures& detected_cpu_features() noexcept;

}  // namespace hikoboshi::dispatch

#endif  // HIKOBOSHI_DISPATCH_CPU_FEATURES_HPP

#ifndef HIKOBOSHI_DISPATCH_SIMD_DISPATCH_SELECT_HPP
#define HIKOBOSHI_DISPATCH_SIMD_DISPATCH_SELECT_HPP

#include <hikoboshi/dispatch/cpu_features.hpp>
#include <hikoboshi/dispatch/dispatch_table.hpp>

namespace hikoboshi::dispatch {

inline bool avx2_dispatch_available(const CpuFeatures& features) noexcept {
  return features.avx2;
}

inline bool avx512_dispatch_available(const CpuFeatures& features) noexcept {
  return features.avx512f;
}

inline const DispatchTable& avx2_dispatch_table() noexcept {
  return scalar_dispatch_table();
}

inline const DispatchTable& avx512_dispatch_table() noexcept {
  return scalar_dispatch_table();
}

}  // namespace hikoboshi::dispatch

#endif  // HIKOBOSHI_DISPATCH_SIMD_DISPATCH_SELECT_HPP

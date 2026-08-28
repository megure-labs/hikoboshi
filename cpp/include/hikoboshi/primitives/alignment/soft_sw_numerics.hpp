#ifndef HIKOBOSHI_PRIMITIVES_ALIGNMENT_SOFT_SW_NUMERICS_HPP
#define HIKOBOSHI_PRIMITIVES_ALIGNMENT_SOFT_SW_NUMERICS_HPP

#include <algorithm>
#include <cmath>

namespace hikoboshi::primitives::alignment {

inline constexpr float kSoftSwNegInf = -1e30F;
inline constexpr float kSoftSwExpClampMin = -88.0F;
inline constexpr float kSoftSwExpClampMax = 88.0F;

inline float safe_exp(float x) noexcept {
  if (x < kSoftSwExpClampMin) {
    return 0.0F;
  }
  if (x > kSoftSwExpClampMax) {
    x = kSoftSwExpClampMax;
  }
  return std::exp(x);
}

inline float safe_log(float x) noexcept {
  return x > 0.0F ? std::log(x) : kSoftSwNegInf;
}

struct KahanSumFloat {
  float sum = 0.0F;
  float compensation = 0.0F;

  inline void add(float value) noexcept {
    const float y = value - compensation;
    const float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }

  inline float result() const noexcept { return sum; }

  inline void reset() noexcept {
    sum = 0.0F;
    compensation = 0.0F;
  }
};

inline float logsumexp2(float a, float b, float temperature) noexcept {
  const float max_v = std::max(a, b);
  if (max_v <= kSoftSwNegInf) {
    return kSoftSwNegInf;
  }
  KahanSumFloat sum;
  sum.add(safe_exp((a - max_v) / temperature));
  sum.add(safe_exp((b - max_v) / temperature));
  return max_v + temperature * std::log(sum.result());
}

inline float logsumexp3(float a, float b, float c, float temperature) noexcept {
  const float max_v = std::max({a, b, c});
  if (max_v <= kSoftSwNegInf) {
    return kSoftSwNegInf;
  }
  KahanSumFloat sum;
  sum.add(safe_exp((a - max_v) / temperature));
  sum.add(safe_exp((b - max_v) / temperature));
  sum.add(safe_exp((c - max_v) / temperature));
  return max_v + temperature * std::log(sum.result());
}

inline float logsumexp4(float a, float b, float c, float d,
                        float temperature) noexcept {
  const float max_v = std::max({a, b, c, d});
  if (max_v <= kSoftSwNegInf) {
    return kSoftSwNegInf;
  }
  KahanSumFloat sum;
  sum.add(safe_exp((a - max_v) / temperature));
  sum.add(safe_exp((b - max_v) / temperature));
  sum.add(safe_exp((c - max_v) / temperature));
  sum.add(safe_exp((d - max_v) / temperature));
  return max_v + temperature * std::log(sum.result());
}

}  // namespace hikoboshi::primitives::alignment

#endif  // HIKOBOSHI_PRIMITIVES_ALIGNMENT_SOFT_SW_NUMERICS_HPP

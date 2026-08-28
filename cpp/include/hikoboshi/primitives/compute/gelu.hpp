#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_GELU_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_GELU_HPP

#include <cstddef>

#include <hikoboshi/primitives/detail/fast_exp_neg.hpp>

namespace hikoboshi::primitives::compute {

enum class GeluPolicy {
  Exact,
};

struct GeluInplaceScalarRequest {
  const float* input;
  std::size_t count;
  GeluPolicy policy;
};

namespace detail {

inline float gelu_fast_erf(float value) noexcept {
  float sign = 1.0F;
  if (value < 0.0F) {
    sign = -1.0F;
    value = -value;
  }
  if (value > 4.0F) {
    return sign;
  }

  constexpr float kA1 = 0.254829592F;
  constexpr float kA2 = -0.284496736F;
  constexpr float kA3 = 1.421413741F;
  constexpr float kA4 = -1.453152027F;
  constexpr float kA5 = 1.061405429F;
  constexpr float kP = 0.3275911F;

  const float t = 1.0F / (1.0F + kP * value);
  const float polynomial =
      (((((kA5 * t + kA4) * t) + kA3) * t + kA2) * t + kA1) * t;
  return sign *
         (1.0F - polynomial *
                      hikoboshi::primitives::detail::fast_exp_neg_sq(value));
}

inline float gelu_exact_scalar(float value) noexcept {
  constexpr float kOneOverSqrtTwo = 0.7071067811865475F;
  return 0.5F * value *
         (1.0F + gelu_fast_erf(value * kOneOverSqrtTwo));
}

}  // namespace detail

inline void gelu_inplace_scalar(const GeluInplaceScalarRequest& request,
                                float* output) noexcept {
  if (request.input == nullptr || output == nullptr ||
      request.policy != GeluPolicy::Exact) {
    return;
  }

  for (std::size_t i = 0; i < request.count; ++i) {
    output[i] = detail::gelu_exact_scalar(request.input[i]);
  }
}

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_GELU_HPP

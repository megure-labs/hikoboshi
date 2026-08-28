#ifndef HIKOBOSHI_PRIMITIVES_DETAIL_FAST_EXP_NEG_HPP
#define HIKOBOSHI_PRIMITIVES_DETAIL_FAST_EXP_NEG_HPP

namespace hikoboshi::primitives::detail {

// Approximates exp(x) for x in [-16, 0] using table lookup + 6-term
// Horner Taylor. Max abs error is about 1e-5 across [-16, 0].
// Returns 0 for x < -16; clamps positive inputs to 1; preserves NaN.
inline float fast_exp_neg(float x) noexcept {
  static constexpr float kExpNegTable[17] = {
      1.0F,
      0.36787944117144232F,
      0.13533528323661270F,
      0.04978706836786394F,
      0.01831563888873418F,
      0.00673794699908547F,
      0.00247875217666636F,
      0.00091188196555452F,
      0.00033546262790251F,
      0.00012340980408668F,
      0.00004539992976248F,
      0.00001670170079024F,
      0.00000614421235333F,
      0.00000226032940698F,
      0.00000083152871910F,
      0.00000030590232050F,
      0.00000011253517471F,
  };
  if (x != x) {
    return x;
  }
  float neg_x = -x;
  if (neg_x > 16.0F) {
    return 0.0F;
  }
  if (neg_x < 0.0F) {
    neg_x = 0.0F;
  }
  int n = static_cast<int>(neg_x + 0.5F);
  if (n > 16) {
    n = 16;
  }
  const float r = -(neg_x - static_cast<float>(n));
  const float exp_r =
      1.0F +
      r * (1.0F +
           r * (0.5F +
                r * (1.0F / 6.0F +
                     r * (1.0F / 24.0F +
                          r * (1.0F / 120.0F + r * (1.0F / 720.0F))))));
  return kExpNegTable[n] * exp_r;
}

// Convenience: exp(-(x*x)).
inline float fast_exp_neg_sq(float x) noexcept {
  return fast_exp_neg(-(x * x));
}

}  // namespace hikoboshi::primitives::detail

#endif  // HIKOBOSHI_PRIMITIVES_DETAIL_FAST_EXP_NEG_HPP

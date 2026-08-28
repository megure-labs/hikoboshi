#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_SOFTMAX_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_SOFTMAX_HPP

/// @file
/// Row-wise softmax primitive for transformer attention and any other
/// caller that needs `softmax(x / temperature + mask)` across each row of
/// a row-major `[row_count, row_dimension]` matrix.
///
/// Numerics follow the standard max-shift form:
///   `m_r = max_d (input[r, d] / temperature + mask[r, d])`
///   `output[r, d] = exp(eff[r, d] - m_r) / sum_d exp(eff[r, d] - m_r)`
/// with `eff[r, d] = input[r, d] / temperature + mask[r, d]` and `mask`
/// defaulting to zero when null. The temperature divides the logits
/// (higher temperature -> softer distribution); attention callers that
/// have already baked `1 / sqrt(d_head)` into `input` should pass
/// `temperature = 1.0F`.
///
/// Fully-masked rows (every entry's effective logit at -inf or
/// `kSoftmaxNegInf`) produce all-zero output rather than NaN so
/// downstream layers do not propagate NaN through padding rows.

#include <cstddef>

namespace hikoboshi::primitives::compute {

/// Sentinel treated as "completely mask out" when applied additively.
///
/// Inputs whose post-mask effective logit is `<= kSoftmaxNegInf` are
/// treated as fully masked; rows where every entry is at or below this
/// sentinel produce all-zero output.
inline constexpr float kSoftmaxNegInf = -1e30F;

struct SoftmaxScalarRequest {
  const float* input;          // row-major [row_count, row_dimension]
  std::size_t row_count;
  std::size_t row_dimension;
  float temperature;           // logit divisor; 1.0 == no extra scaling
  const float* mask;           // optional [row_count, row_dimension]
                               // additive mask; <= kSoftmaxNegInf masks
                               // a position; nullptr means zero mask
};

struct SoftmaxScalarOutput {
  float* output;               // row-major [row_count, row_dimension]
};

void softmax_scalar(const SoftmaxScalarRequest& request,
                    const SoftmaxScalarOutput& output);

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_SOFTMAX_HPP

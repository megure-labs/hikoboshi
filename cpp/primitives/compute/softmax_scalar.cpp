#include <hikoboshi/primitives/compute/softmax.hpp>

#include <cmath>
#include <cstddef>

namespace hikoboshi::primitives::compute {

namespace {

inline float effective_logit(float input, float inv_temperature,
                             const float* mask_row,
                             std::size_t d) noexcept {
  const float scaled = input * inv_temperature;
  if (mask_row == nullptr) {
    return scaled;
  }
  return scaled + mask_row[d];
}

}  // namespace

void softmax_scalar(const SoftmaxScalarRequest& request,
                    const SoftmaxScalarOutput& output) {
  const std::size_t row_count = request.row_count;
  const std::size_t dim = request.row_dimension;
  if (row_count == 0 || dim == 0) {
    return;
  }

  const float inv_t = 1.0F / request.temperature;

  for (std::size_t r = 0; r < row_count; ++r) {
    const float* row_in = request.input + r * dim;
    const float* mask_row =
        request.mask != nullptr ? request.mask + r * dim : nullptr;
    float* row_out = output.output + r * dim;

    // First pass: row max after temperature scaling + additive mask.
    float row_max = -INFINITY;
    for (std::size_t d = 0; d < dim; ++d) {
      const float v = effective_logit(row_in[d], inv_t, mask_row, d);
      if (v > row_max) {
        row_max = v;
      }
    }

    // Fully-masked row: emit zeros instead of NaN so downstream layers
    // (typically a value-vector weighting in attention) do not propagate
    // NaN through padding rows.
    if (!(row_max > kSoftmaxNegInf)) {
      for (std::size_t d = 0; d < dim; ++d) {
        row_out[d] = 0.0F;
      }
      continue;
    }

    // Second pass: shifted exp into the output buffer, summing in row
    // order so the reduction tree is deterministic across runs.
    float row_sum = 0.0F;
    for (std::size_t d = 0; d < dim; ++d) {
      const float v = effective_logit(row_in[d], inv_t, mask_row, d);
      const float shifted = v - row_max;
      const float e = std::exp(shifted);
      row_out[d] = e;
      row_sum += e;
    }

    // Third pass: normalize. The row_sum is strictly positive because
    // at least one entry exceeded kSoftmaxNegInf and its exp(0) = 1
    // contribution survives.
    const float inv_sum = 1.0F / row_sum;
    for (std::size_t d = 0; d < dim; ++d) {
      row_out[d] *= inv_sum;
    }
  }
}

}  // namespace hikoboshi::primitives::compute

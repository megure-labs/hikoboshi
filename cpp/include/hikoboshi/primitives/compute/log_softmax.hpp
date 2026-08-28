#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_LOG_SOFTMAX_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_LOG_SOFTMAX_HPP

/// @file
/// Row-wise scalar log-softmax primitive declaration.

#include <cstddef>

namespace hikoboshi::primitives::compute {

struct LogSoftmaxScalarRequest {
  const float* input;  // row-major [row_count, row_dimension]
  std::size_t row_count;
  std::size_t row_dimension;
};

struct LogSoftmaxScalarOutput {
  float* output;  // row-major [row_count, row_dimension]
};

/// Numerically stable `x - logsumexp(x)` over each row's last dimension.
void log_softmax_scalar(const LogSoftmaxScalarRequest& request,
                        const LogSoftmaxScalarOutput& output);

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_LOG_SOFTMAX_HPP

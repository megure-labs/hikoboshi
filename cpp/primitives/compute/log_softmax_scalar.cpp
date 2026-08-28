#include <hikoboshi/primitives/compute/log_softmax.hpp>

#include <cmath>
#include <cstddef>

namespace hikoboshi::primitives::compute {

void log_softmax_scalar(const LogSoftmaxScalarRequest& request,
                        const LogSoftmaxScalarOutput& output) {
  const std::size_t row_count = request.row_count;
  const std::size_t dim = request.row_dimension;
  if (row_count == 0 || dim == 0) {
    return;
  }

  for (std::size_t r = 0; r < row_count; ++r) {
    const float* row_in = request.input + r * dim;
    float* row_out = output.output + r * dim;

    float row_max = -INFINITY;
    for (std::size_t d = 0; d < dim; ++d) {
      if (row_in[d] > row_max) {
        row_max = row_in[d];
      }
    }

    float row_sum = 0.0F;
    for (std::size_t d = 0; d < dim; ++d) {
      row_sum += std::exp(row_in[d] - row_max);
    }

    const float log_row_sum = std::log(row_sum);
    for (std::size_t d = 0; d < dim; ++d) {
      row_out[d] = (row_in[d] - row_max) - log_row_sum;
    }
  }
}

}  // namespace hikoboshi::primitives::compute

#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_BIAS_ADD_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_BIAS_ADD_HPP

#include <cstddef>

namespace hikoboshi::primitives::compute {

struct BiasAddScalarRequest {
  const float* input;
  const float* bias;
  std::size_t row_count;
  std::size_t row_dimension;
};

inline void bias_add_scalar(const BiasAddScalarRequest& request,
                            float* output) noexcept {
  if (request.input == nullptr || request.bias == nullptr ||
      output == nullptr || request.row_dimension == 0U) {
    return;
  }

  const std::size_t dim = request.row_dimension;
  for (std::size_t row = 0; row < request.row_count; ++row) {
    const float* row_in = request.input + row * dim;
    float* row_out = output + row * dim;
    for (std::size_t d = 0; d < dim; ++d) {
      row_out[d] = row_in[d] + request.bias[d];
    }
  }
}

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_BIAS_ADD_HPP

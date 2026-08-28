#include <hikoboshi/primitives/compute/layer_norm.hpp>

#include <cmath>
#include <cstddef>

namespace hikoboshi::primitives::compute {

void layer_norm_scalar(const LayerNormScalarRequest& request, float* output) {
  const std::size_t dim = request.row_dimension;
  if (dim == 0) {
    return;
  }

  for (std::size_t r = 0; r < request.row_count; ++r) {
    const float* row_in = request.input + r * dim;
    float* row_out = output + r * dim;

    // Welford one-pass mean and variance.
    float mean = 0.0F;
    float m2 = 0.0F;
    for (std::size_t d = 0; d < dim; ++d) {
      const float x = row_in[d];
      const float count = static_cast<float>(d + 1);
      const float delta = x - mean;
      mean += delta / count;
      const float delta2 = x - mean;
      m2 += delta * delta2;
    }
    const float variance = m2 / static_cast<float>(dim);
    const float inv_std = 1.0F / std::sqrt(variance + request.epsilon);

    for (std::size_t d = 0; d < dim; ++d) {
      const float normalized = (row_in[d] - mean) * inv_std;
      const float scaled = request.gamma != nullptr
                               ? normalized * request.gamma[d]
                               : normalized;
      row_out[d] = request.beta != nullptr ? scaled + request.beta[d] : scaled;
    }
  }
}

}  // namespace hikoboshi::primitives::compute

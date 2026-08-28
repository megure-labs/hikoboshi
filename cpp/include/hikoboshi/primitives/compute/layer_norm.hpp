#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_LAYER_NORM_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_LAYER_NORM_HPP

#include <cstddef>

namespace hikoboshi::primitives::compute {

struct LayerNormScalarRequest {
  const float* input;
  const float* gamma;
  const float* beta;
  std::size_t row_count;
  std::size_t row_dimension;
  float epsilon;
};

void layer_norm_scalar(const LayerNormScalarRequest& request, float* output);

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_LAYER_NORM_HPP

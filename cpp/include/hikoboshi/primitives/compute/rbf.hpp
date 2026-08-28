#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_RBF_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_RBF_HPP

#include <cstddef>

namespace hikoboshi::primitives::compute {

struct RbfScalarRequest {
  const float* squared_distances;
  std::size_t value_count;
  std::size_t feature_count;
  float center_min;
  float center_max;
  float sigma;
};

void rbf_scalar(const RbfScalarRequest& request, float* output);

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_RBF_HPP

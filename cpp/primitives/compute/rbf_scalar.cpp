#include <hikoboshi/primitives/compute/rbf.hpp>

#include <hikoboshi/primitives/detail/fast_exp_neg.hpp>

#include <cmath>
#include <cstddef>

namespace hikoboshi::primitives::compute {

void rbf_scalar(const RbfScalarRequest& request, float* output) {
  if (request.feature_count == 0 || request.value_count == 0) {
    return;
  }

  const std::size_t feature_count = request.feature_count;
  const float center_min = request.center_min;
  const float center_max = request.center_max;
  const float center_step = feature_count > 1
                                ? (center_max - center_min) /
                                      static_cast<float>(feature_count - 1)
                                : 0.0F;
  const float sigma = request.sigma;
  const float two_sigma_squared = 2.0F * sigma * sigma;

  for (std::size_t i = 0; i < request.value_count; ++i) {
    const float distance = std::sqrt(request.squared_distances[i]);
    float* row = output + i * feature_count;
    for (std::size_t f = 0; f < feature_count; ++f) {
      const float center = center_min + static_cast<float>(f) * center_step;
      const float delta = distance - center;
      row[f] = hikoboshi::primitives::detail::fast_exp_neg(
          -(delta * delta) / two_sigma_squared);
    }
  }
}

}  // namespace hikoboshi::primitives::compute

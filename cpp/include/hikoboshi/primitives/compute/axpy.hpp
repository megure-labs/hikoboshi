#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_AXPY_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_AXPY_HPP

#include <cstddef>

namespace hikoboshi::primitives::compute {

struct AxpyScalarRequest {
  const float* x;
  const float* y;
  float alpha;
  std::size_t count;
};

inline void axpy_scalar(const AxpyScalarRequest& request,
                        float* output) noexcept {
  if (request.x == nullptr || request.y == nullptr || output == nullptr) {
    return;
  }

  for (std::size_t i = 0; i < request.count; ++i) {
    output[i] = request.alpha * request.x[i] + request.y[i];
  }
}

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_AXPY_HPP

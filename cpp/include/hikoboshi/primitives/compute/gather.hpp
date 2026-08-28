#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_GATHER_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_GATHER_HPP

#include <cstddef>
#include <cstdint>

namespace hikoboshi::primitives::compute {

struct GatherScalarRequest {
  const float* source;
  const std::int32_t* indices;
  std::size_t source_row_count;
  std::size_t row_dimension;
  std::size_t index_count;
};

void gather_scalar(const GatherScalarRequest& request, float* output);

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_GATHER_HPP

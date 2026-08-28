#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_REDUCE_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_REDUCE_HPP

#include <cstddef>

namespace hikoboshi::primitives::compute {

struct ReduceRowsScalarRequest {
  const float* input;
  std::size_t row_count;
  std::size_t row_dimension;
};

void reduce_sum_rows_scalar(const ReduceRowsScalarRequest& request, float* output);
void reduce_mean_rows_scalar(const ReduceRowsScalarRequest& request, float* output);

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_REDUCE_HPP

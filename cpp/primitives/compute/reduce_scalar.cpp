#include <hikoboshi/primitives/compute/reduce.hpp>

#include <cstddef>

namespace hikoboshi::primitives::compute {

void reduce_sum_rows_scalar(const ReduceRowsScalarRequest& request,
                            float* output) {
  for (std::size_t r = 0; r < request.row_count; ++r) {
    const float* row = request.input + r * request.row_dimension;
    float acc = 0.0F;
    // Sequential row-major accumulation: deterministic across threads because
    // the loop order is fixed and no associativity reordering happens here.
    for (std::size_t d = 0; d < request.row_dimension; ++d) {
      acc += row[d];
    }
    output[r] = acc;
  }
}

void reduce_mean_rows_scalar(const ReduceRowsScalarRequest& request,
                             float* output) {
  reduce_sum_rows_scalar(request, output);
  if (request.row_dimension == 0) {
    return;
  }
  const float inv = 1.0F / static_cast<float>(request.row_dimension);
  for (std::size_t r = 0; r < request.row_count; ++r) {
    output[r] *= inv;
  }
}

}  // namespace hikoboshi::primitives::compute

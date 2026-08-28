#include <hikoboshi/primitives/compute/gather.hpp>

#include <cstddef>
#include <cstdint>

namespace hikoboshi::primitives::compute {

void gather_scalar(const GatherScalarRequest& request, float* output) {
  const std::size_t row_dim = request.row_dimension;
  const std::int32_t source_count =
      static_cast<std::int32_t>(request.source_row_count);

  for (std::size_t i = 0; i < request.index_count; ++i) {
    float* dst = output + i * row_dim;
    const std::int32_t idx = request.indices[i];
    if (idx < 0 || idx >= source_count) {
      // Invalid-index policy: zero-fill so downstream masked math observes
      // the gather miss as a clean zero contribution.
      for (std::size_t d = 0; d < row_dim; ++d) {
        dst[d] = 0.0F;
      }
      continue;
    }
    const float* src = request.source + static_cast<std::size_t>(idx) * row_dim;
    for (std::size_t d = 0; d < row_dim; ++d) {
      dst[d] = src[d];
    }
  }
}

}  // namespace hikoboshi::primitives::compute

#include <hikoboshi/modules/similarity.hpp>

#include <hikoboshi/modules/detail/similarity_inline.hpp>

namespace hikoboshi::modules::detail {

void similarity_scalar_unchecked(const SimilarityScalarRequest& request,
                                 const SimilarityScalarOutput& output) noexcept {
  similarity_scalar_unchecked_inline(request, output);
}

}  // namespace hikoboshi::modules::detail

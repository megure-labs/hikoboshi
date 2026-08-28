#include <hikoboshi/modules/similarity.hpp>

namespace hikoboshi::modules {
namespace {

universal::Status invalid(const char* detail) noexcept {
  return {universal::StatusCode::InvalidArgument, detail};
}

}  // namespace

namespace detail {
void similarity_scalar_unchecked(const SimilarityScalarRequest& request,
                                 const SimilarityScalarOutput& output) noexcept;
}  // namespace detail

universal::Status similarity_scalar(
    const SimilarityScalarRequest& request,
    const SimilarityScalarOutput& output) noexcept {
  if (request.query_embedding.values.data == nullptr ||
      request.target_embedding.values.data == nullptr) {
    return invalid("similarity embedding values pointer is null");
  }
  if (output.similarity_matrix == nullptr) {
    return invalid("similarity output matrix pointer is null");
  }
  if (request.query_embedding.dimension == 0 ||
      request.target_embedding.dimension == 0) {
    return invalid("similarity embedding dimensions must be non-zero");
  }
  if (request.query_embedding.dimension != request.target_embedding.dimension) {
    return invalid("similarity embedding dimensions differ");
  }
  if (output.query_count < request.query_embedding.residue_count ||
      output.target_count < request.target_embedding.residue_count) {
    return invalid("similarity output shape is smaller than embedding inputs");
  }
  detail::similarity_scalar_unchecked(request, output);
  return {universal::StatusCode::Ok, ""};
}

}  // namespace hikoboshi::modules

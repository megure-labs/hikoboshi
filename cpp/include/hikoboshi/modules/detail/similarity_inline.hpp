#ifndef HIKOBOSHI_MODULES_DETAIL_SIMILARITY_INLINE_HPP
#define HIKOBOSHI_MODULES_DETAIL_SIMILARITY_INLINE_HPP

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/similarity.hpp>
#include <hikoboshi/universal/inline.hpp>

namespace hikoboshi::modules::detail {

HIKOBOSHI_FORCE_INLINE void similarity_scalar_unchecked_inline(
    const SimilarityScalarRequest& request,
    const SimilarityScalarOutput& output) noexcept {
  hikoboshi::primitives::linalg::GemmScalarRequest gemm{};
  gemm.lhs = request.query_embedding.values.data;
  gemm.rhs = request.target_embedding.values.data;
  gemm.m = request.query_embedding.residue_count;
  gemm.n = request.target_embedding.residue_count;
  gemm.k = request.query_embedding.dimension;
  hikoboshi::dispatch::gemm_nt_forward(hikoboshi::dispatch::ScalarTag{}, gemm,
                                     output.similarity_matrix);
}

}  // namespace hikoboshi::modules::detail

#endif  // HIKOBOSHI_MODULES_DETAIL_SIMILARITY_INLINE_HPP

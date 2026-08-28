#ifndef HIKOBOSHI_MODULES_SIMILARITY_HPP
#define HIKOBOSHI_MODULES_SIMILARITY_HPP

#include <cstddef>

#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::modules {

struct SimilarityScalarRequest {
  hikoboshi::universal::EmbeddingView query_embedding;
  hikoboshi::universal::EmbeddingView target_embedding;
};

struct SimilarityScalarOutput {
  float* similarity_matrix;  // row-major [Lq, Lt]
  std::size_t query_count;
  std::size_t target_count;
};

inline constexpr hikoboshi::universal::ScoreSemantics
    kSimilarityScalarScoreSemantics =
        hikoboshi::universal::kRawDotV1ScoreSemantics;

inline constexpr hikoboshi::universal::ScoreMatrixView score_matrix_view(
    const SimilarityScalarOutput& output) noexcept {
  return {output.similarity_matrix,
          output.query_count,
          output.target_count,
          output.target_count};
}

hikoboshi::universal::Status similarity_scalar(
    const SimilarityScalarRequest& request,
    const SimilarityScalarOutput& output) noexcept;

}  // namespace hikoboshi::modules

#endif  // HIKOBOSHI_MODULES_SIMILARITY_HPP

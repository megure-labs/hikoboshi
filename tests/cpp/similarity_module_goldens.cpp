#include <hikoboshi/modules/similarity.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

namespace hiko_m = hikoboshi::modules;
namespace hiko_u = hikoboshi::universal;

namespace {

bool nearly_equal(float a, float b, float tolerance = 1.0e-5F) {
  return std::fabs(a - b) <= tolerance;
}

void fail(const char* detail) {
  std::fprintf(stderr, "similarity_module_goldens: %s\n", detail);
  std::exit(1);
}

void require_raw_dot_hard_sw_semantics(const hiko_u::ScoreSemantics& semantics) {
  if (semantics.dtype != hiko_u::DataType::Float32 ||
      semantics.layout != hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget ||
      !semantics.higher_is_better || !semantics.local_affine_additive ||
      semantics.normalization != hiko_u::ScoreNormalization::None ||
      semantics.scale_family != hiko_u::ScoreScaleFamily::RawDot ||
      semantics.method != hiko_u::ScoreMethod::RawDotV1) {
    fail("raw dot similarity must declare float32 row-major hard-SW semantics");
  }
}

void test_raw_dot_product_not_cosine() {
  const float query[2 * 3] = {
      1.0F, 2.0F, 3.0F,
      4.0F, 5.0F, 6.0F,
  };
  const float target[2 * 3] = {
      7.0F, 8.0F, 9.0F,
      10.0F, 11.0F, 12.0F,
  };
  const hiko_u::EmbeddingView query_view{2, 3, {query, 6}, {}, {}};
  const hiko_u::EmbeddingView target_view{2, 3, {target, 6}, {}, {}};
  float scores[2 * 2] = {0.0F, 0.0F, 0.0F, 0.0F};
  const hiko_m::SimilarityScalarOutput output{scores, 2, 2};

  const auto status = hiko_m::similarity_scalar({query_view, target_view}, output);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("similarity_scalar returned non-ok for valid embeddings");
  }
  require_raw_dot_hard_sw_semantics(hiko_m::kSimilarityScalarScoreSemantics);
  const hiko_u::ScoreMatrixView matrix = hiko_m::score_matrix_view(output);
  if (matrix.values != scores || matrix.query_length != 2 ||
      matrix.target_length != 2 || matrix.row_stride != 2) {
    fail("similarity output must expose a contiguous [Lq, Lt] score matrix");
  }
  if (!nearly_equal(scores[0], 50.0F) || !nearly_equal(scores[1], 68.0F) ||
      !nearly_equal(scores[2], 122.0F) || !nearly_equal(scores[3], 167.0F)) {
    fail("raw dot-product scores do not match gemm_nt semantics");
  }
  if (!nearly_equal(matrix.values[0 * matrix.row_stride + 1], 68.0F) ||
      !nearly_equal(matrix.values[1 * matrix.row_stride + 0], 122.0F)) {
    fail("score matrix must be row-major with query rows and target columns");
  }
  if (scores[0] <= 1.0F) {
    fail("similarity output looks cosine-normalized instead of raw dot product");
  }
}

void test_dimension_mismatch_is_rejected() {
  const float query[2] = {1.0F, 2.0F};
  const float target[3] = {1.0F, 2.0F, 3.0F};
  const hiko_u::EmbeddingView query_view{1, 2, {query, 2}, {}, {}};
  const hiko_u::EmbeddingView target_view{1, 3, {target, 3}, {}, {}};
  float score = 0.0F;

  const auto status = hiko_m::similarity_scalar({query_view, target_view},
                                             {&score, 1, 1});
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("dimension mismatch must return InvalidArgument");
  }
}

void test_mpnn64_raw_dot_path_matches_reference() {
  constexpr std::size_t query_count = 5;
  constexpr std::size_t target_count = 6;
  constexpr std::size_t dimension = 64;
  std::array<float, query_count * dimension> query{};
  std::array<float, target_count * dimension> target{};
  std::array<float, query_count * target_count> scores{};

  for (std::size_t i = 0; i < query_count; ++i) {
    for (std::size_t d = 0; d < dimension; ++d) {
      const int centered = static_cast<int>(d % 7) - 3;
      query[i * dimension + d] =
          static_cast<float>((i + 1) * centered) * 0.03125F;
      if (d == i) {
        query[i * dimension + d] += 0.5F;
      }
    }
  }
  for (std::size_t j = 0; j < target_count; ++j) {
    for (std::size_t d = 0; d < dimension; ++d) {
      const int centered = static_cast<int>(d % 5) - 2;
      target[j * dimension + d] =
          static_cast<float>((j + 2) * centered) * -0.046875F;
      if ((d + j) % 11 == 0) {
        target[j * dimension + d] += 0.25F;
      }
    }
  }

  const hiko_u::EmbeddingView query_view{
      query_count, dimension, {query.data(), query.size()}, {}, {}};
  const hiko_u::EmbeddingView target_view{
      target_count, dimension, {target.data(), target.size()}, {}, {}};
  const auto status = hiko_m::similarity_scalar(
      {query_view, target_view}, {scores.data(), query_count, target_count});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("similarity_scalar returned non-ok for MPNN-64 embeddings");
  }

  for (std::size_t i = 0; i < query_count; ++i) {
    for (std::size_t j = 0; j < target_count; ++j) {
      float expected = 0.0F;
      for (std::size_t d = 0; d < dimension; ++d) {
        expected += query[i * dimension + d] * target[j * dimension + d];
      }
      if (!nearly_equal(scores[i * target_count + j], expected, 1.0e-4F)) {
        fail("MPNN-64 raw dot scores differ from scalar reference");
      }
    }
  }
}

}  // namespace

int main() {
  test_raw_dot_product_not_cosine();
  test_mpnn64_raw_dot_path_matches_reference();
  test_dimension_mismatch_is_rejected();
  return 0;
}

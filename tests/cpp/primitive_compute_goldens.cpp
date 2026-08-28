#include <hikoboshi/primitives/compute/gather.hpp>
#include <hikoboshi/primitives/compute/knn.hpp>
#include <hikoboshi/primitives/compute/layer_norm.hpp>
#include <hikoboshi/primitives/compute/rbf.hpp>
#include <hikoboshi/primitives/compute/reduce.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace hiko_p = hikoboshi::primitives::compute;

namespace {

bool nearly_equal(float a, float b, float tolerance = 1e-5F) {
  return std::fabs(a - b) <= tolerance;
}

void fail(const char* tag) {
  std::fprintf(stderr, "primitive_compute_goldens: %s\n", tag);
  std::exit(1);
}

void test_knn_tie_by_index() {
  // Five candidates equidistant from the query (squared distance 1.0); the
  // earliest target index must win each slot deterministically.
  const float query[3] = {0.0F, 0.0F, 0.0F};
  const float targets[5 * 3] = {
      1.0F, 0.0F, 0.0F,  // index 0
      0.0F, 1.0F, 0.0F,  // index 1
      -1.0F, 0.0F, 0.0F, // index 2
      0.0F, -1.0F, 0.0F, // index 3
      0.0F, 0.0F, 1.0F,  // index 4
  };
  std::int32_t indices[3] = {-9, -9, -9};
  float distances[3] = {-9.0F, -9.0F, -9.0F};
  hiko_p::KnnScalarRequest request{};
  request.query_coordinates = query;
  request.target_coordinates = targets;
  request.query_count = 1;
  request.target_count = 5;
  request.k = 3;
  request.include_self = true;
  hiko_p::KnnScalarOutput out{};
  out.neighbor_indices = indices;
  out.neighbor_squared_distances = distances;
  hiko_p::knn_scalar(request, out);
  if (indices[0] != 0 || indices[1] != 1 || indices[2] != 2) {
    fail("knn equidistant candidates must keep earliest indices");
  }
  for (std::size_t s = 0; s < 3; ++s) {
    if (!nearly_equal(distances[s], 1.0F)) {
      fail("knn squared distance for unit vectors should be 1");
    }
  }
}

void test_knn_self_neighbor_excluded_when_requested() {
  // Query == target coordinates and include_self=false; KNN must skip i==i.
  const float coords[3 * 3] = {
      0.0F, 0.0F, 0.0F,  // index 0
      1.0F, 0.0F, 0.0F,  // index 1
      0.0F, 2.0F, 0.0F,  // index 2
  };
  std::int32_t indices[2 * 2] = {-9, -9, -9, -9};
  float distances[2 * 2] = {-9.0F, -9.0F, -9.0F, -9.0F};

  hiko_p::KnnScalarRequest request{};
  request.query_coordinates = coords;
  request.target_coordinates = coords;
  request.query_count = 2;
  request.target_count = 3;
  request.k = 2;
  request.include_self = false;
  hiko_p::KnnScalarOutput out{};
  out.neighbor_indices = indices;
  out.neighbor_squared_distances = distances;
  hiko_p::knn_scalar(request, out);

  // Query 0 (origin) neighbors should be 1 and 2.
  if (indices[0] != 1 || indices[1] != 2) {
    fail("knn include_self=false must skip self for query 0");
  }
  if (!nearly_equal(distances[0], 1.0F) || !nearly_equal(distances[1], 4.0F)) {
    fail("knn squared distances for query 0 are wrong");
  }

  // Query 1 (1,0,0) neighbors: distances are 1 to index 0 and 5 to index 2.
  if (indices[2] != 0 || indices[3] != 2) {
    fail("knn include_self=false must order ascending for query 1");
  }
}

void test_rbf_formula() {
  const float squared_distances[1] = {4.0F};  // distance = 2.0
  const float center_min = 1.0F;
  const float center_max = 3.0F;
  const float sigma = 0.5F;
  float output[3] = {0.0F, 0.0F, 0.0F};
  hiko_p::RbfScalarRequest request{};
  request.squared_distances = squared_distances;
  request.value_count = 1;
  request.feature_count = 3;
  request.center_min = center_min;
  request.center_max = center_max;
  request.sigma = sigma;
  hiko_p::rbf_scalar(request, output);

  // Centers: 1.0, 2.0, 3.0. Distance 2.0 → exp(-(d - c)^2 / (2 sigma^2)).
  const float two_sigma_sq = 2.0F * sigma * sigma;
  const float expected0 = std::exp(-1.0F / two_sigma_sq);
  const float expected1 = 1.0F;
  const float expected2 = std::exp(-1.0F / two_sigma_sq);
  if (!nearly_equal(output[0], expected0) || !nearly_equal(output[1], expected1) ||
      !nearly_equal(output[2], expected2)) {
    fail("rbf scalar formula mismatch at uniform centers");
  }
}

void test_gather_invalid_index_zero_fill() {
  const float source[2 * 4] = {
      1.0F, 2.0F, 3.0F, 4.0F,
      5.0F, 6.0F, 7.0F, 8.0F,
  };
  const std::int32_t indices[5] = {1, -1, 0, 7, -42};
  float output[5 * 4] = {0};
  hiko_p::GatherScalarRequest request{};
  request.source = source;
  request.indices = indices;
  request.source_row_count = 2;
  request.row_dimension = 4;
  request.index_count = 5;
  hiko_p::gather_scalar(request, output);

  if (!nearly_equal(output[0], 5.0F) || !nearly_equal(output[3], 8.0F)) {
    fail("gather valid index 1 must copy source row 1");
  }
  for (std::size_t d = 0; d < 4; ++d) {
    if (!nearly_equal(output[1 * 4 + d], 0.0F)) {
      fail("gather invalid -1 must zero-fill row");
    }
  }
  if (!nearly_equal(output[2 * 4 + 0], 1.0F) || !nearly_equal(output[2 * 4 + 3], 4.0F)) {
    fail("gather valid index 0 must copy source row 0");
  }
  for (std::size_t d = 0; d < 4; ++d) {
    if (!nearly_equal(output[3 * 4 + d], 0.0F)) {
      fail("gather out-of-range positive index must zero-fill");
    }
    if (!nearly_equal(output[4 * 4 + d], 0.0F)) {
      fail("gather large negative index must zero-fill");
    }
  }
}

void test_layer_norm_welford() {
  // Single row [1, 2, 3, 4]; mean = 2.5, variance = 1.25, std = sqrt(1.25).
  const float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  const float gamma[4] = {1.0F, 1.0F, 1.0F, 1.0F};
  const float beta[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  float output[4] = {0};
  hiko_p::LayerNormScalarRequest request{};
  request.input = input;
  request.gamma = gamma;
  request.beta = beta;
  request.row_count = 1;
  request.row_dimension = 4;
  request.epsilon = 0.0F;
  hiko_p::layer_norm_scalar(request, output);

  const float mean = 2.5F;
  const float variance = 1.25F;
  const float inv_std = 1.0F / std::sqrt(variance);
  for (std::size_t d = 0; d < 4; ++d) {
    const float expected = (input[d] - mean) * inv_std;
    if (!nearly_equal(output[d], expected, 1e-4F)) {
      fail("layer norm Welford pass produced wrong row");
    }
  }
}

void test_layer_norm_affine_shift() {
  // Row [-1, 1] with gamma=[2, 2], beta=[1, -1]: mean=0, var=1, output=[2*(-1)+1, 2*1-1]=[-1, 1].
  const float input[2] = {-1.0F, 1.0F};
  const float gamma[2] = {2.0F, 2.0F};
  const float beta[2] = {1.0F, -1.0F};
  float output[2] = {0};
  hiko_p::LayerNormScalarRequest request{};
  request.input = input;
  request.gamma = gamma;
  request.beta = beta;
  request.row_count = 1;
  request.row_dimension = 2;
  request.epsilon = 0.0F;
  hiko_p::layer_norm_scalar(request, output);
  if (!nearly_equal(output[0], -1.0F, 1e-4F) ||
      !nearly_equal(output[1], 1.0F, 1e-4F)) {
    fail("layer norm affine scale/shift produced wrong values");
  }
}

void test_reduce_sum_and_mean_deterministic() {
  const float input[2 * 3] = {
      1.0F, 2.0F, 3.0F,
      4.0F, 5.0F, 6.0F,
  };
  hiko_p::ReduceRowsScalarRequest request{};
  request.input = input;
  request.row_count = 2;
  request.row_dimension = 3;

  float sum_out[2] = {0};
  hiko_p::reduce_sum_rows_scalar(request, sum_out);
  if (!nearly_equal(sum_out[0], 6.0F) || !nearly_equal(sum_out[1], 15.0F)) {
    fail("reduce_sum_rows produced wrong row totals");
  }

  float mean_out[2] = {0};
  hiko_p::reduce_mean_rows_scalar(request, mean_out);
  if (!nearly_equal(mean_out[0], 2.0F) || !nearly_equal(mean_out[1], 5.0F)) {
    fail("reduce_mean_rows produced wrong row means");
  }
}

}  // namespace

int main() {
  test_knn_tie_by_index();
  test_knn_self_neighbor_excluded_when_requested();
  test_rbf_formula();
  test_gather_invalid_index_zero_fill();
  test_layer_norm_welford();
  test_layer_norm_affine_shift();
  test_reduce_sum_and_mean_deterministic();
  return 0;
}

#include <hikoboshi/primitives/linalg/gemm.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hiko_l = hikoboshi::primitives::linalg;

namespace {

bool nearly_equal(float a, float b, float tolerance = 1e-4F) {
  return std::fabs(a - b) <= tolerance;
}

void fail(const char* tag) {
  std::fprintf(stderr, "primitive_linalg_goldens: %s\n", tag);
  std::exit(1);
}

void test_gemm_nn_identity_returns_lhs() {
  // 3x2 lhs, 2x2 identity rhs: output should equal lhs.
  const float lhs[3 * 2] = {
      1.0F, 2.0F,
      3.0F, 4.0F,
      5.0F, 6.0F,
  };
  const float rhs[2 * 2] = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  float output[3 * 2] = {0};
  hiko_l::GemmScalarRequest request{};
  request.lhs = lhs;
  request.rhs = rhs;
  request.m = 3;
  request.n = 2;
  request.k = 2;
  hiko_l::gemm_nn_scalar(request, output);
  for (std::size_t i = 0; i < 3 * 2; ++i) {
    if (!nearly_equal(output[i], lhs[i])) {
      fail("gemm_nn against identity must reproduce lhs");
    }
  }
}

void test_gemm_nn_small_matrix_correctness() {
  // [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
  const float lhs[2 * 3] = {
      1.0F, 2.0F, 3.0F,
      4.0F, 5.0F, 6.0F,
  };
  const float rhs[3 * 2] = {
      7.0F, 8.0F,
      9.0F, 10.0F,
      11.0F, 12.0F,
  };
  float output[2 * 2] = {0};
  hiko_l::GemmScalarRequest request{};
  request.lhs = lhs;
  request.rhs = rhs;
  request.m = 2;
  request.n = 2;
  request.k = 3;
  hiko_l::gemm_nn_scalar(request, output);
  if (!nearly_equal(output[0], 58.0F) || !nearly_equal(output[1], 64.0F) ||
      !nearly_equal(output[2], 139.0F) || !nearly_equal(output[3], 154.0F)) {
    fail("gemm_nn 2x3 by 3x2 produced wrong product");
  }
}

void test_gemm_nt_raw_dot_product() {
  // gemm_nt: C[i,j] = sum_k lhs[i,k] * rhs[j,k] (raw dot product over rows).
  const float lhs[2 * 3] = {
      1.0F, 2.0F, 3.0F,
      4.0F, 5.0F, 6.0F,
  };
  const float rhs[2 * 3] = {
      7.0F, 8.0F, 9.0F,
      10.0F, 11.0F, 12.0F,
  };
  float output[2 * 2] = {0};
  hiko_l::GemmScalarRequest request{};
  request.lhs = lhs;
  request.rhs = rhs;
  request.m = 2;
  request.n = 2;
  request.k = 3;
  hiko_l::gemm_nt_scalar(request, output);
  // C[0,0] = 1*7+2*8+3*9 = 50; C[0,1] = 1*10+2*11+3*12 = 68;
  // C[1,0] = 4*7+5*8+6*9 = 122; C[1,1] = 4*10+5*11+6*12 = 167.
  if (!nearly_equal(output[0], 50.0F) || !nearly_equal(output[1], 68.0F) ||
      !nearly_equal(output[2], 122.0F) || !nearly_equal(output[3], 167.0F)) {
    fail("gemm_nt 2x3 with 2x3 rhs produced wrong dot products");
  }
}

void test_gemm_nn_wider_than_block() {
  // Force the inner-block path by going wider than kBlockedGemmInnerBlock.
  constexpr std::size_t m = 5;
  constexpr std::size_t n = 5;
  constexpr std::size_t k = 64;
  float lhs[m * k];
  float rhs[k * n];
  for (std::size_t i = 0; i < m * k; ++i) {
    lhs[i] = static_cast<float>((i % 7) + 1);
  }
  for (std::size_t i = 0; i < k * n; ++i) {
    rhs[i] = static_cast<float>((i % 5) + 1);
  }
  float blocked_output[m * n] = {0};
  hiko_l::GemmScalarRequest request{};
  request.lhs = lhs;
  request.rhs = rhs;
  request.m = m;
  request.n = n;
  request.k = k;
  hiko_l::gemm_nn_scalar(request, blocked_output);

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float reference = 0.0F;
      for (std::size_t p = 0; p < k; ++p) {
        reference += lhs[i * k + p] * rhs[p * n + j];
      }
      if (!nearly_equal(blocked_output[i * n + j], reference, 1e-3F)) {
        fail("gemm_nn cache-blocked path disagrees with naive reference");
      }
    }
  }
}

}  // namespace

int main() {
  test_gemm_nn_identity_returns_lhs();
  test_gemm_nn_small_matrix_correctness();
  test_gemm_nt_raw_dot_product();
  test_gemm_nn_wider_than_block();
  return 0;
}

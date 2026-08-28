#include <hikoboshi/primitives/linalg/gemm.hpp>

#include <hikoboshi/primitives/detail/blocked_gemm.hpp>

#include <algorithm>
#include <cstddef>

namespace hikoboshi::primitives {

namespace detail {

namespace {

#if defined(__GNUC__) || defined(__clang__)
#define HIKOBOSHI_FAST_GEMM_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define HIKOBOSHI_FAST_GEMM_ALWAYS_INLINE inline
#endif

HIKOBOSHI_FAST_GEMM_ALWAYS_INLINE void zero_matrix(float* output,
                                                 std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = 0.0F;
  }
}

// BLIS-style 4x4 NN microkernel: streaming outer product.
// Reads 4 A-columns and 4 B-row entries per K-step; 16 register
// accumulators. `first_block` toggles between fresh-write and
// read-modify-write so the macro driver can K-block freely.
HIKOBOSHI_FAST_GEMM_ALWAYS_INLINE void fast_nn_4x4(const float* lhs,
                                                 const float* rhs,
                                                 float* output,
                                                 std::size_t lda,
                                                 std::size_t ldb,
                                                 std::size_t ldc,
                                                 std::size_t k_block,
                                                 bool first_block) noexcept {
  float c00 = first_block ? 0.0F : output[0 * ldc + 0];
  float c01 = first_block ? 0.0F : output[0 * ldc + 1];
  float c02 = first_block ? 0.0F : output[0 * ldc + 2];
  float c03 = first_block ? 0.0F : output[0 * ldc + 3];
  float c10 = first_block ? 0.0F : output[1 * ldc + 0];
  float c11 = first_block ? 0.0F : output[1 * ldc + 1];
  float c12 = first_block ? 0.0F : output[1 * ldc + 2];
  float c13 = first_block ? 0.0F : output[1 * ldc + 3];
  float c20 = first_block ? 0.0F : output[2 * ldc + 0];
  float c21 = first_block ? 0.0F : output[2 * ldc + 1];
  float c22 = first_block ? 0.0F : output[2 * ldc + 2];
  float c23 = first_block ? 0.0F : output[2 * ldc + 3];
  float c30 = first_block ? 0.0F : output[3 * ldc + 0];
  float c31 = first_block ? 0.0F : output[3 * ldc + 1];
  float c32 = first_block ? 0.0F : output[3 * ldc + 2];
  float c33 = first_block ? 0.0F : output[3 * ldc + 3];

  for (std::size_t p = 0; p < k_block; ++p) {
    const float a0 = lhs[0 * lda + p];
    const float a1 = lhs[1 * lda + p];
    const float a2 = lhs[2 * lda + p];
    const float a3 = lhs[3 * lda + p];
    const float* rhs_row = rhs + p * ldb;
    const float b0 = rhs_row[0];
    const float b1 = rhs_row[1];
    const float b2 = rhs_row[2];
    const float b3 = rhs_row[3];

    c00 += a0 * b0;
    c01 += a0 * b1;
    c02 += a0 * b2;
    c03 += a0 * b3;
    c10 += a1 * b0;
    c11 += a1 * b1;
    c12 += a1 * b2;
    c13 += a1 * b3;
    c20 += a2 * b0;
    c21 += a2 * b1;
    c22 += a2 * b2;
    c23 += a2 * b3;
    c30 += a3 * b0;
    c31 += a3 * b1;
    c32 += a3 * b2;
    c33 += a3 * b3;
  }

  output[0 * ldc + 0] = c00;
  output[0 * ldc + 1] = c01;
  output[0 * ldc + 2] = c02;
  output[0 * ldc + 3] = c03;
  output[1 * ldc + 0] = c10;
  output[1 * ldc + 1] = c11;
  output[1 * ldc + 2] = c12;
  output[1 * ldc + 3] = c13;
  output[2 * ldc + 0] = c20;
  output[2 * ldc + 1] = c21;
  output[2 * ldc + 2] = c22;
  output[2 * ldc + 3] = c23;
  output[3 * ldc + 0] = c30;
  output[3 * ldc + 1] = c31;
  output[3 * ldc + 2] = c32;
  output[3 * ldc + 3] = c33;
}

// BLIS-style 4x4 NT microkernel. B is laid out as [N, K] row-major;
// reading `rhs[j * ldb + p]` walks one B-row contiguous in memory per
// j-stride, matching the same streaming-outer-product shape as the NN
// kernel.
HIKOBOSHI_FAST_GEMM_ALWAYS_INLINE void fast_nt_4x4(const float* lhs,
                                                 const float* rhs,
                                                 float* output,
                                                 std::size_t lda,
                                                 std::size_t ldb,
                                                 std::size_t ldc,
                                                 std::size_t k_block,
                                                 bool first_block) noexcept {
  float c00 = first_block ? 0.0F : output[0 * ldc + 0];
  float c01 = first_block ? 0.0F : output[0 * ldc + 1];
  float c02 = first_block ? 0.0F : output[0 * ldc + 2];
  float c03 = first_block ? 0.0F : output[0 * ldc + 3];
  float c10 = first_block ? 0.0F : output[1 * ldc + 0];
  float c11 = first_block ? 0.0F : output[1 * ldc + 1];
  float c12 = first_block ? 0.0F : output[1 * ldc + 2];
  float c13 = first_block ? 0.0F : output[1 * ldc + 3];
  float c20 = first_block ? 0.0F : output[2 * ldc + 0];
  float c21 = first_block ? 0.0F : output[2 * ldc + 1];
  float c22 = first_block ? 0.0F : output[2 * ldc + 2];
  float c23 = first_block ? 0.0F : output[2 * ldc + 3];
  float c30 = first_block ? 0.0F : output[3 * ldc + 0];
  float c31 = first_block ? 0.0F : output[3 * ldc + 1];
  float c32 = first_block ? 0.0F : output[3 * ldc + 2];
  float c33 = first_block ? 0.0F : output[3 * ldc + 3];

  for (std::size_t p = 0; p < k_block; ++p) {
    const float a0 = lhs[0 * lda + p];
    const float a1 = lhs[1 * lda + p];
    const float a2 = lhs[2 * lda + p];
    const float a3 = lhs[3 * lda + p];
    const float b0 = rhs[0 * ldb + p];
    const float b1 = rhs[1 * ldb + p];
    const float b2 = rhs[2 * ldb + p];
    const float b3 = rhs[3 * ldb + p];

    c00 += a0 * b0;
    c01 += a0 * b1;
    c02 += a0 * b2;
    c03 += a0 * b3;
    c10 += a1 * b0;
    c11 += a1 * b1;
    c12 += a1 * b2;
    c13 += a1 * b3;
    c20 += a2 * b0;
    c21 += a2 * b1;
    c22 += a2 * b2;
    c23 += a2 * b3;
    c30 += a3 * b0;
    c31 += a3 * b1;
    c32 += a3 * b2;
    c33 += a3 * b3;
  }

  output[0 * ldc + 0] = c00;
  output[0 * ldc + 1] = c01;
  output[0 * ldc + 2] = c02;
  output[0 * ldc + 3] = c03;
  output[1 * ldc + 0] = c10;
  output[1 * ldc + 1] = c11;
  output[1 * ldc + 2] = c12;
  output[1 * ldc + 3] = c13;
  output[2 * ldc + 0] = c20;
  output[2 * ldc + 1] = c21;
  output[2 * ldc + 2] = c22;
  output[2 * ldc + 3] = c23;
  output[3 * ldc + 0] = c30;
  output[3 * ldc + 1] = c31;
  output[3 * ldc + 2] = c32;
  output[3 * ldc + 3] = c33;
}

#undef HIKOBOSHI_FAST_GEMM_ALWAYS_INLINE

}  // namespace

void blocked_gemm_nn_scalar_fast(const float* lhs,
                                 const float* rhs,
                                 float* output,
                                 std::size_t m,
                                 std::size_t n,
                                 std::size_t k) noexcept {
  if (m == 0 || n == 0) {
    return;
  }
  if (k == 0) {
    zero_matrix(output, m * n);
    return;
  }

  const std::size_t lda = k;
  const std::size_t ldb = n;
  const std::size_t ldc = n;

  for (std::size_t jj = 0; jj < n; jj += kFastGemmNc) {
    const std::size_t j_block = std::min(kFastGemmNc, n - jj);
    for (std::size_t kk = 0; kk < k; kk += kFastGemmKc) {
      const std::size_t k_block = std::min(kFastGemmKc, k - kk);
      const bool first_block = kk == 0;
      for (std::size_t ii = 0; ii < m; ii += kFastGemmMc) {
        const std::size_t i_block = std::min(kFastGemmMc, m - ii);
        const float* a_block = lhs + ii * lda + kk;
        const float* b_block = rhs + kk * ldb + jj;
        float* c_block = output + ii * ldc + jj;

        std::size_t i = 0;
        for (; i + kFastGemmMr <= i_block; i += kFastGemmMr) {
          std::size_t j = 0;
          for (; j + kFastGemmNr <= j_block; j += kFastGemmNr) {
            fast_nn_4x4(a_block + i * lda,
                        b_block + j,
                        c_block + i * ldc + j,
                        lda,
                        ldb,
                        ldc,
                        k_block,
                        first_block);
          }
          for (; j < j_block; ++j) {
            for (std::size_t i_tail = 0; i_tail < kFastGemmMr; ++i_tail) {
              const std::size_t row = i + i_tail;
              float acc = first_block ? 0.0F : c_block[row * ldc + j];
              for (std::size_t p = 0; p < k_block; ++p) {
                acc += a_block[row * lda + p] * b_block[p * ldb + j];
              }
              c_block[row * ldc + j] = acc;
            }
          }
        }
        for (; i < i_block; ++i) {
          for (std::size_t j = 0; j < j_block; ++j) {
            float acc = first_block ? 0.0F : c_block[i * ldc + j];
            for (std::size_t p = 0; p < k_block; ++p) {
              acc += a_block[i * lda + p] * b_block[p * ldb + j];
            }
            c_block[i * ldc + j] = acc;
          }
        }
      }
    }
  }
}

void blocked_gemm_nt_scalar_fast(const float* lhs,
                                 const float* rhs,
                                 float* output,
                                 std::size_t m,
                                 std::size_t n,
                                 std::size_t k) noexcept {
  if (m == 0 || n == 0) {
    return;
  }
  if (k == 0) {
    zero_matrix(output, m * n);
    return;
  }

  const std::size_t lda = k;
  const std::size_t ldb = k;
  const std::size_t ldc = n;

  for (std::size_t jj = 0; jj < n; jj += kFastGemmNc) {
    const std::size_t j_block = std::min(kFastGemmNc, n - jj);
    for (std::size_t kk = 0; kk < k; kk += kFastGemmKc) {
      const std::size_t k_block = std::min(kFastGemmKc, k - kk);
      const bool first_block = kk == 0;
      for (std::size_t ii = 0; ii < m; ii += kFastGemmMc) {
        const std::size_t i_block = std::min(kFastGemmMc, m - ii);
        const float* a_block = lhs + ii * lda + kk;
        const float* b_block = rhs + jj * ldb + kk;
        float* c_block = output + ii * ldc + jj;

        std::size_t i = 0;
        for (; i + kFastGemmMr <= i_block; i += kFastGemmMr) {
          std::size_t j = 0;
          for (; j + kFastGemmNr <= j_block; j += kFastGemmNr) {
            fast_nt_4x4(a_block + i * lda,
                        b_block + j * ldb,
                        c_block + i * ldc + j,
                        lda,
                        ldb,
                        ldc,
                        k_block,
                        first_block);
          }
          for (; j < j_block; ++j) {
            for (std::size_t i_tail = 0; i_tail < kFastGemmMr; ++i_tail) {
              const std::size_t row = i + i_tail;
              float acc = first_block ? 0.0F : c_block[row * ldc + j];
              for (std::size_t p = 0; p < k_block; ++p) {
                acc += a_block[row * lda + p] * b_block[j * ldb + p];
              }
              c_block[row * ldc + j] = acc;
            }
          }
        }
        for (; i < i_block; ++i) {
          for (std::size_t j = 0; j < j_block; ++j) {
            float acc = first_block ? 0.0F : c_block[i * ldc + j];
            for (std::size_t p = 0; p < k_block; ++p) {
              acc += a_block[i * lda + p] * b_block[j * ldb + p];
            }
            c_block[i * ldc + j] = acc;
          }
        }
      }
    }
  }
}

}  // namespace detail

namespace linalg {

void gemm_nn_scalar_fast(const GemmScalarRequest& request, float* output) {
  detail::blocked_gemm_nn_scalar_fast(request.lhs,
                                      request.rhs,
                                      output,
                                      request.m,
                                      request.n,
                                      request.k);
}

void gemm_nt_scalar_fast(const GemmScalarRequest& request, float* output) {
  detail::blocked_gemm_nt_scalar_fast(request.lhs,
                                      request.rhs,
                                      output,
                                      request.m,
                                      request.n,
                                      request.k);
}

}  // namespace linalg

}  // namespace hikoboshi::primitives

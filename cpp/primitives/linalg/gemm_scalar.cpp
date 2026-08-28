#include <hikoboshi/primitives/linalg/gemm.hpp>

#include <hikoboshi/primitives/detail/blocked_gemm.hpp>

#include <algorithm>
#include <cstddef>

namespace hikoboshi::primitives {

namespace detail {

namespace {

#if defined(__GNUC__) || defined(__clang__)
#define HIKOBOSHI_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define HIKOBOSHI_ALWAYS_INLINE inline
#endif

HIKOBOSHI_ALWAYS_INLINE void zero_matrix(float* output,
                                       std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    output[i] = 0.0F;
  }
}

HIKOBOSHI_ALWAYS_INLINE void gemm_nn_4x4_kernel(const float* lhs,
                                              const float* rhs,
                                              float* output,
                                              std::size_t n,
                                              std::size_t k,
                                              std::size_t p_begin,
                                              std::size_t p_end,
                                              bool first_block) noexcept {
  float c00 = first_block ? 0.0F : output[0 * n + 0];
  float c01 = first_block ? 0.0F : output[0 * n + 1];
  float c02 = first_block ? 0.0F : output[0 * n + 2];
  float c03 = first_block ? 0.0F : output[0 * n + 3];
  float c10 = first_block ? 0.0F : output[1 * n + 0];
  float c11 = first_block ? 0.0F : output[1 * n + 1];
  float c12 = first_block ? 0.0F : output[1 * n + 2];
  float c13 = first_block ? 0.0F : output[1 * n + 3];
  float c20 = first_block ? 0.0F : output[2 * n + 0];
  float c21 = first_block ? 0.0F : output[2 * n + 1];
  float c22 = first_block ? 0.0F : output[2 * n + 2];
  float c23 = first_block ? 0.0F : output[2 * n + 3];
  float c30 = first_block ? 0.0F : output[3 * n + 0];
  float c31 = first_block ? 0.0F : output[3 * n + 1];
  float c32 = first_block ? 0.0F : output[3 * n + 2];
  float c33 = first_block ? 0.0F : output[3 * n + 3];

  for (std::size_t p = p_begin; p < p_end; ++p) {
    const float a0 = lhs[0 * k + p];
    const float a1 = lhs[1 * k + p];
    const float a2 = lhs[2 * k + p];
    const float a3 = lhs[3 * k + p];
    const float* rhs_row = rhs + p * n;
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

  output[0 * n + 0] = c00;
  output[0 * n + 1] = c01;
  output[0 * n + 2] = c02;
  output[0 * n + 3] = c03;
  output[1 * n + 0] = c10;
  output[1 * n + 1] = c11;
  output[1 * n + 2] = c12;
  output[1 * n + 3] = c13;
  output[2 * n + 0] = c20;
  output[2 * n + 1] = c21;
  output[2 * n + 2] = c22;
  output[2 * n + 3] = c23;
  output[3 * n + 0] = c30;
  output[3 * n + 1] = c31;
  output[3 * n + 2] = c32;
  output[3 * n + 3] = c33;
}

HIKOBOSHI_ALWAYS_INLINE void gemm_nt_4x4_kernel(const float* lhs,
                                              const float* rhs,
                                              float* output,
                                              std::size_t n,
                                              std::size_t k,
                                              std::size_t p_begin,
                                              std::size_t p_end,
                                              bool first_block) noexcept {
  float c00 = first_block ? 0.0F : output[0 * n + 0];
  float c01 = first_block ? 0.0F : output[0 * n + 1];
  float c02 = first_block ? 0.0F : output[0 * n + 2];
  float c03 = first_block ? 0.0F : output[0 * n + 3];
  float c10 = first_block ? 0.0F : output[1 * n + 0];
  float c11 = first_block ? 0.0F : output[1 * n + 1];
  float c12 = first_block ? 0.0F : output[1 * n + 2];
  float c13 = first_block ? 0.0F : output[1 * n + 3];
  float c20 = first_block ? 0.0F : output[2 * n + 0];
  float c21 = first_block ? 0.0F : output[2 * n + 1];
  float c22 = first_block ? 0.0F : output[2 * n + 2];
  float c23 = first_block ? 0.0F : output[2 * n + 3];
  float c30 = first_block ? 0.0F : output[3 * n + 0];
  float c31 = first_block ? 0.0F : output[3 * n + 1];
  float c32 = first_block ? 0.0F : output[3 * n + 2];
  float c33 = first_block ? 0.0F : output[3 * n + 3];

  for (std::size_t p = p_begin; p < p_end; ++p) {
    const float a0 = lhs[0 * k + p];
    const float a1 = lhs[1 * k + p];
    const float a2 = lhs[2 * k + p];
    const float a3 = lhs[3 * k + p];
    const float b0 = rhs[0 * k + p];
    const float b1 = rhs[1 * k + p];
    const float b2 = rhs[2 * k + p];
    const float b3 = rhs[3 * k + p];

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

  output[0 * n + 0] = c00;
  output[0 * n + 1] = c01;
  output[0 * n + 2] = c02;
  output[0 * n + 3] = c03;
  output[1 * n + 0] = c10;
  output[1 * n + 1] = c11;
  output[1 * n + 2] = c12;
  output[1 * n + 3] = c13;
  output[2 * n + 0] = c20;
  output[2 * n + 1] = c21;
  output[2 * n + 2] = c22;
  output[2 * n + 3] = c23;
  output[3 * n + 0] = c30;
  output[3 * n + 1] = c31;
  output[3 * n + 2] = c32;
  output[3 * n + 3] = c33;
}

HIKOBOSHI_ALWAYS_INLINE void gemm_nn_1x4_kernel(const float* lhs,
                                              const float* rhs,
                                              float* output,
                                              std::size_t n,
                                              std::size_t p_begin,
                                              std::size_t p_end,
                                              bool first_block) noexcept {
  float c0 = first_block ? 0.0F : output[0];
  float c1 = first_block ? 0.0F : output[1];
  float c2 = first_block ? 0.0F : output[2];
  float c3 = first_block ? 0.0F : output[3];
  for (std::size_t p = p_begin; p < p_end; ++p) {
    const float a = lhs[p];
    const float* rhs_row = rhs + p * n;
    c0 += a * rhs_row[0];
    c1 += a * rhs_row[1];
    c2 += a * rhs_row[2];
    c3 += a * rhs_row[3];
  }
  output[0] = c0;
  output[1] = c1;
  output[2] = c2;
  output[3] = c3;
}

HIKOBOSHI_ALWAYS_INLINE void gemm_nt_1x4_kernel(const float* lhs,
                                              const float* rhs,
                                              float* output,
                                              std::size_t k,
                                              std::size_t p_begin,
                                              std::size_t p_end,
                                              bool first_block) noexcept {
  float c0 = first_block ? 0.0F : output[0];
  float c1 = first_block ? 0.0F : output[1];
  float c2 = first_block ? 0.0F : output[2];
  float c3 = first_block ? 0.0F : output[3];
  for (std::size_t p = p_begin; p < p_end; ++p) {
    const float a = lhs[p];
    c0 += a * rhs[0 * k + p];
    c1 += a * rhs[1 * k + p];
    c2 += a * rhs[2 * k + p];
    c3 += a * rhs[3 * k + p];
  }
  output[0] = c0;
  output[1] = c1;
  output[2] = c2;
  output[3] = c3;
}

void gemm_nt_k64_scalar(const float* lhs,
                        const float* rhs,
                        float* output,
                        std::size_t m,
                        std::size_t n) noexcept {
  constexpr std::size_t kHiddenDimension = 64;
  constexpr std::size_t kRowBlock = 64;
  constexpr std::size_t kColBlock = 256;

  for (std::size_t i0 = 0; i0 < m; i0 += kRowBlock) {
    const std::size_t i_end = std::min(i0 + kRowBlock, m);
    for (std::size_t j0 = 0; j0 < n; j0 += kColBlock) {
      const std::size_t j_end = std::min(j0 + kColBlock, n);
      std::size_t i = i0;
      for (; i + 4 <= i_end; i += 4) {
        std::size_t j = j0;
        for (; j + 4 <= j_end; j += 4) {
          gemm_nt_4x4_kernel(lhs + i * kHiddenDimension,
                             rhs + j * kHiddenDimension,
                             output + i * n + j,
                             n,
                             kHiddenDimension,
                             0,
                             kHiddenDimension,
                             true);
        }
        for (; j < j_end; ++j) {
          for (std::size_t i_tail = i; i_tail < i + 4; ++i_tail) {
            float acc = 0.0F;
            for (std::size_t p = 0; p < kHiddenDimension; ++p) {
              acc += lhs[i_tail * kHiddenDimension + p] *
                     rhs[j * kHiddenDimension + p];
            }
            output[i_tail * n + j] = acc;
          }
        }
      }
      for (; i < i_end; ++i) {
        std::size_t j = j0;
        for (; j + 4 <= j_end; j += 4) {
          gemm_nt_1x4_kernel(lhs + i * kHiddenDimension,
                             rhs + j * kHiddenDimension,
                             output + i * n + j,
                             kHiddenDimension,
                             0,
                             kHiddenDimension,
                             true);
        }
        for (; j < j_end; ++j) {
          float acc = 0.0F;
          for (std::size_t p = 0; p < kHiddenDimension; ++p) {
            acc += lhs[i * kHiddenDimension + p] *
                   rhs[j * kHiddenDimension + p];
          }
          output[i * n + j] = acc;
        }
      }
    }
  }
}

bool is_mpnn64_nt_shape(std::size_t n, std::size_t k) noexcept {
  return n == 64 && (k == 192 || k == 256 || k == 416);
}

void gemm_nt_n64_full_k_scalar(const float* lhs,
                               const float* rhs,
                               float* output,
                               std::size_t m,
                               std::size_t k) noexcept {
  constexpr std::size_t kOutputDimension = 64;
  constexpr std::size_t kRowBlock = 64;

  for (std::size_t i0 = 0; i0 < m; i0 += kRowBlock) {
    const std::size_t i_end = std::min(i0 + kRowBlock, m);
    std::size_t i = i0;
    for (; i + 4 <= i_end; i += 4) {
      for (std::size_t j = 0; j < kOutputDimension; j += 4) {
        gemm_nt_4x4_kernel(lhs + i * k,
                           rhs + j * k,
                           output + i * kOutputDimension + j,
                           kOutputDimension,
                           k,
                           0,
                           k,
                           true);
      }
    }
    for (; i < i_end; ++i) {
      for (std::size_t j = 0; j < kOutputDimension; j += 4) {
        gemm_nt_1x4_kernel(lhs + i * k,
                           rhs + j * k,
                           output + i * kOutputDimension + j,
                           k,
                           0,
                           k,
                           true);
      }
    }
  }
}

#undef HIKOBOSHI_ALWAYS_INLINE

}  // namespace

void blocked_gemm_nn_scalar(const float* lhs,
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
  for (std::size_t i0 = 0; i0 < m; i0 += kBlockedGemmRowBlock) {
    const std::size_t i_end = std::min(i0 + kBlockedGemmRowBlock, m);
    for (std::size_t j0 = 0; j0 < n; j0 += kBlockedGemmColBlock) {
      const std::size_t j_end = std::min(j0 + kBlockedGemmColBlock, n);
      for (std::size_t p0 = 0; p0 < k; p0 += kBlockedGemmInnerBlock) {
        const std::size_t p_end = std::min(p0 + kBlockedGemmInnerBlock, k);
        const bool first_block = p0 == 0;
        std::size_t i = i0;
        for (; i + 4 <= i_end; i += 4) {
          std::size_t j = j0;
          for (; j + 4 <= j_end; j += 4) {
            gemm_nn_4x4_kernel(lhs + i * k,
                               rhs + j,
                               output + i * n + j,
                               n,
                               k,
                               p0,
                               p_end,
                               first_block);
          }
          for (; j < j_end; ++j) {
            for (std::size_t i_tail = i; i_tail < i + 4; ++i_tail) {
              float acc = first_block ? 0.0F : output[i_tail * n + j];
              for (std::size_t p = p0; p < p_end; ++p) {
                acc += lhs[i_tail * k + p] * rhs[p * n + j];
              }
              output[i_tail * n + j] = acc;
            }
          }
        }
        for (; i < i_end; ++i) {
          std::size_t j = j0;
          for (; j + 4 <= j_end; j += 4) {
            gemm_nn_1x4_kernel(lhs + i * k,
                               rhs + j,
                               output + i * n + j,
                               n,
                               p0,
                               p_end,
                               first_block);
          }
          for (; j < j_end; ++j) {
            float acc = first_block ? 0.0F : output[i * n + j];
            for (std::size_t p = p0; p < p_end; ++p) {
              acc += lhs[i * k + p] * rhs[p * n + j];
            }
            output[i * n + j] = acc;
          }
        }
      }
    }
  }
}

void blocked_gemm_nt_scalar(const float* lhs,
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
  if (k == 64) {
    gemm_nt_k64_scalar(lhs, rhs, output, m, n);
    return;
  }
  if (is_mpnn64_nt_shape(n, k)) {
    gemm_nt_n64_full_k_scalar(lhs, rhs, output, m, k);
    return;
  }
  for (std::size_t i0 = 0; i0 < m; i0 += kBlockedGemmRowBlock) {
    const std::size_t i_end = std::min(i0 + kBlockedGemmRowBlock, m);
    for (std::size_t j0 = 0; j0 < n; j0 += kBlockedGemmColBlock) {
      const std::size_t j_end = std::min(j0 + kBlockedGemmColBlock, n);
      for (std::size_t p0 = 0; p0 < k; p0 += kBlockedGemmInnerBlock) {
        const std::size_t p_end = std::min(p0 + kBlockedGemmInnerBlock, k);
        const bool first_block = p0 == 0;
        std::size_t i = i0;
        for (; i + 4 <= i_end; i += 4) {
          std::size_t j = j0;
          for (; j + 4 <= j_end; j += 4) {
            gemm_nt_4x4_kernel(lhs + i * k,
                               rhs + j * k,
                               output + i * n + j,
                               n,
                               k,
                               p0,
                               p_end,
                               first_block);
          }
          for (; j < j_end; ++j) {
            for (std::size_t i_tail = i; i_tail < i + 4; ++i_tail) {
              float acc = first_block ? 0.0F : output[i_tail * n + j];
              for (std::size_t p = p0; p < p_end; ++p) {
                acc += lhs[i_tail * k + p] * rhs[j * k + p];
              }
              output[i_tail * n + j] = acc;
            }
          }
        }
        for (; i < i_end; ++i) {
          std::size_t j = j0;
          for (; j + 4 <= j_end; j += 4) {
            gemm_nt_1x4_kernel(lhs + i * k,
                               rhs + j * k,
                               output + i * n + j,
                               k,
                               p0,
                               p_end,
                               first_block);
          }
          for (; j < j_end; ++j) {
            float acc = first_block ? 0.0F : output[i * n + j];
            for (std::size_t p = p0; p < p_end; ++p) {
              acc += lhs[i * k + p] * rhs[j * k + p];
            }
            output[i * n + j] = acc;
          }
        }
      }
    }
  }
}

}  // namespace detail

namespace linalg {

void gemm_nn_scalar(const GemmScalarRequest& request, float* output) {
  detail::blocked_gemm_nn_scalar(request.lhs, request.rhs, output, request.m,
                                 request.n, request.k);
}

void gemm_nt_scalar(const GemmScalarRequest& request, float* output) {
  detail::blocked_gemm_nt_scalar(request.lhs, request.rhs, output, request.m,
                                 request.n, request.k);
}

}  // namespace linalg

}  // namespace hikoboshi::primitives

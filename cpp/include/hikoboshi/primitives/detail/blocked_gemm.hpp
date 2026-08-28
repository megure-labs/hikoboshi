#ifndef HIKOBOSHI_PRIMITIVES_DETAIL_BLOCKED_GEMM_HPP
#define HIKOBOSHI_PRIMITIVES_DETAIL_BLOCKED_GEMM_HPP

#include <cstddef>

namespace hikoboshi::primitives::detail {

// Strict-parity scalar GEMM tile constants. Used by `blocked_gemm_*_scalar`
// to match the existing Hikoboshi-MPNN-64 reduction tree.
inline constexpr std::size_t kBlockedGemmRowBlock = 16;
inline constexpr std::size_t kBlockedGemmColBlock = 16;
inline constexpr std::size_t kBlockedGemmInnerBlock = 32;

// Fast-parity scalar GEMM tile constants. Used by
// `blocked_gemm_*_scalar_fast` to drive the BLIS-style streaming-
// outer-product traversal ported from the archive implementation.
inline constexpr std::size_t kFastGemmMc = 64;
inline constexpr std::size_t kFastGemmNc = 256;
inline constexpr std::size_t kFastGemmKc = 64;
inline constexpr std::size_t kFastGemmMr = 4;
inline constexpr std::size_t kFastGemmNr = 4;

void blocked_gemm_nn_scalar(const float* lhs,
                            const float* rhs,
                            float* output,
                            std::size_t m,
                            std::size_t n,
                            std::size_t k) noexcept;

void blocked_gemm_nt_scalar(const float* lhs,
                            const float* rhs,
                            float* output,
                            std::size_t m,
                            std::size_t n,
                            std::size_t k) noexcept;

void blocked_gemm_nn_scalar_fast(const float* lhs,
                                 const float* rhs,
                                 float* output,
                                 std::size_t m,
                                 std::size_t n,
                                 std::size_t k) noexcept;

void blocked_gemm_nt_scalar_fast(const float* lhs,
                                 const float* rhs,
                                 float* output,
                                 std::size_t m,
                                 std::size_t n,
                                 std::size_t k) noexcept;

}  // namespace hikoboshi::primitives::detail

#endif  // HIKOBOSHI_PRIMITIVES_DETAIL_BLOCKED_GEMM_HPP

#ifndef HIKOBOSHI_PRIMITIVES_LINALG_GEMM_HPP
#define HIKOBOSHI_PRIMITIVES_LINALG_GEMM_HPP

#include <cstddef>

namespace hikoboshi::primitives::linalg {

struct GemmScalarRequest {
  const float* lhs;
  const float* rhs;
  std::size_t m;
  std::size_t n;
  std::size_t k;
};

// Strict-parity scalar GEMM. Bit-identical to the current Hikoboshi-MPNN-64
// reduction tree and the existing primitive parity goldens. K-fixed and
// N-fixed short-circuits for MPNN-64 shapes are documented in
// `docs/charters/GEMM_SPECIALIZATION_CHARTER.md`.
void gemm_nn_scalar(const GemmScalarRequest& request, float* output);
void gemm_nt_scalar(const GemmScalarRequest& request, float* output);

// Fast-parity scalar GEMM. BLIS-style streaming-outer-product blocked over
// MC=64, NC=256, KC=64 with the same 4x4 register-tile microkernel. Stays
// within the public 1e-4 max-abs tolerance contract but does not promise
// bit identity to the strict reduction tree. Ported from the archive
// implementation at
// `hikoboshi-archive/src/core/hikoboshi/primitives/linalg/gemm/gemm_scalar.cpp`.
void gemm_nn_scalar_fast(const GemmScalarRequest& request, float* output);
void gemm_nt_scalar_fast(const GemmScalarRequest& request, float* output);

}  // namespace hikoboshi::primitives::linalg

#endif  // HIKOBOSHI_PRIMITIVES_LINALG_GEMM_HPP

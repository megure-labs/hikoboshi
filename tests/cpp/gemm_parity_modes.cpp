// Verifies the StrictParity / FastParity dispatch axis on the scalar
// backend. Strict-mode entries must be bit-identical to the existing
// `gemm_*_scalar` primitive; fast-mode entries must stay within the
// public 1e-4 max-abs tolerance contract against a naive reference at
// the encoder, similarity, and FFN GEMM shapes.

#include <hikoboshi/dispatch/backend_tag.hpp>
#include <hikoboshi/dispatch/dispatch_table.hpp>
#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/primitives/linalg/gemm.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

namespace hiko_d = hikoboshi::dispatch;
namespace hiko_l = hikoboshi::primitives::linalg;

namespace {

constexpr float kFastTolerance = 1.0e-4F;

void fail(const char* tag) {
  std::fprintf(stderr, "gemm_parity_modes: %s\n", tag);
  std::exit(1);
}

float fill_value(std::size_t i, std::size_t seed) {
  // Deterministic non-trivial fill: values in [-1, 1] with phase from seed.
  const std::uint64_t mixed = static_cast<std::uint64_t>(i) * 2654435761ull +
                              static_cast<std::uint64_t>(seed) * 40503ull;
  const float normalized =
      static_cast<float>(mixed % 65537ull) / 32768.0F - 1.0F;
  return normalized;
}

void naive_gemm_nt(const float* lhs,
                   const float* rhs,
                   float* output,
                   std::size_t m,
                   std::size_t n,
                   std::size_t k) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0F;
      for (std::size_t p = 0; p < k; ++p) {
        acc += lhs[i * k + p] * rhs[j * k + p];
      }
      output[i * n + j] = acc;
    }
  }
}

void naive_gemm_nn(const float* lhs,
                   const float* rhs,
                   float* output,
                   std::size_t m,
                   std::size_t n,
                   std::size_t k) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0F;
      for (std::size_t p = 0; p < k; ++p) {
        acc += lhs[i * k + p] * rhs[p * n + j];
      }
      output[i * n + j] = acc;
    }
  }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > worst) {
      worst = d;
    }
  }
  return worst;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> make_matrix(std::size_t rows,
                               std::size_t cols,
                               std::size_t seed) {
  std::vector<float> result(rows * cols);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = fill_value(i, seed);
  }
  return result;
}

void test_strict_mode_matches_primitive(std::size_t m,
                                        std::size_t n,
                                        std::size_t k,
                                        const char* tag) {
  const std::vector<float> lhs = make_matrix(m, k, 1);
  const std::vector<float> rhs_nt = make_matrix(n, k, 2);
  const std::vector<float> rhs_nn = make_matrix(k, n, 3);

  hiko_l::GemmScalarRequest request_nt{};
  request_nt.lhs = lhs.data();
  request_nt.rhs = rhs_nt.data();
  request_nt.m = m;
  request_nt.n = n;
  request_nt.k = k;

  std::vector<float> nt_primitive(m * n);
  hiko_l::gemm_nt_scalar(request_nt, nt_primitive.data());

  std::vector<float> nt_strict(m * n);
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{},
                       hiko_d::StrictParityTag{},
                       request_nt,
                       nt_strict.data());
  if (!bit_equal(nt_primitive, nt_strict)) {
    std::fprintf(stderr,
                 "gemm_parity_modes: %s nt strict-tag must be bit-identical to "
                 "gemm_nt_scalar (m=%zu n=%zu k=%zu)\n",
                 tag, m, n, k);
    std::exit(1);
  }

  hiko_l::GemmScalarRequest request_nn{};
  request_nn.lhs = lhs.data();
  request_nn.rhs = rhs_nn.data();
  request_nn.m = m;
  request_nn.n = n;
  request_nn.k = k;

  std::vector<float> nn_primitive(m * n);
  hiko_l::gemm_nn_scalar(request_nn, nn_primitive.data());

  std::vector<float> nn_strict(m * n);
  hiko_d::gemm_nn_forward(hiko_d::ScalarTag{},
                       hiko_d::StrictParityTag{},
                       request_nn,
                       nn_strict.data());
  if (!bit_equal(nn_primitive, nn_strict)) {
    std::fprintf(stderr,
                 "gemm_parity_modes: %s nn strict-tag must be bit-identical to "
                 "gemm_nn_scalar (m=%zu n=%zu k=%zu)\n",
                 tag, m, n, k);
    std::exit(1);
  }
}

void test_fast_mode_within_tolerance(std::size_t m,
                                     std::size_t n,
                                     std::size_t k,
                                     const char* tag) {
  const std::vector<float> lhs = make_matrix(m, k, 1);
  const std::vector<float> rhs_nt = make_matrix(n, k, 2);
  const std::vector<float> rhs_nn = make_matrix(k, n, 3);

  std::vector<float> reference_nt(m * n);
  naive_gemm_nt(lhs.data(), rhs_nt.data(), reference_nt.data(), m, n, k);

  hiko_l::GemmScalarRequest request_nt{};
  request_nt.lhs = lhs.data();
  request_nt.rhs = rhs_nt.data();
  request_nt.m = m;
  request_nt.n = n;
  request_nt.k = k;

  std::vector<float> nt_fast(m * n);
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{},
                       hiko_d::FastParityTag{},
                       request_nt,
                       nt_fast.data());
  const float nt_max = max_abs_diff(nt_fast, reference_nt);
  if (!(nt_max <= kFastTolerance)) {
    std::fprintf(stderr,
                 "gemm_parity_modes: %s nt fast-tag max abs %.3e exceeds "
                 "1e-4 tolerance (m=%zu n=%zu k=%zu)\n",
                 tag, static_cast<double>(nt_max), m, n, k);
    std::exit(1);
  }

  std::vector<float> reference_nn(m * n);
  naive_gemm_nn(lhs.data(), rhs_nn.data(), reference_nn.data(), m, n, k);

  hiko_l::GemmScalarRequest request_nn{};
  request_nn.lhs = lhs.data();
  request_nn.rhs = rhs_nn.data();
  request_nn.m = m;
  request_nn.n = n;
  request_nn.k = k;

  std::vector<float> nn_fast(m * n);
  hiko_d::gemm_nn_forward(hiko_d::ScalarTag{},
                       hiko_d::FastParityTag{},
                       request_nn,
                       nn_fast.data());
  const float nn_max = max_abs_diff(nn_fast, reference_nn);
  if (!(nn_max <= kFastTolerance)) {
    std::fprintf(stderr,
                 "gemm_parity_modes: %s nn fast-tag max abs %.3e exceeds "
                 "1e-4 tolerance (m=%zu n=%zu k=%zu)\n",
                 tag, static_cast<double>(nn_max), m, n, k);
    std::exit(1);
  }
}

void test_runtime_selector_via_env() {
  constexpr std::size_t m = 17;
  constexpr std::size_t n = 32;
  constexpr std::size_t k = 64;
  const std::vector<float> lhs = make_matrix(m, k, 11);
  const std::vector<float> rhs = make_matrix(n, k, 13);

  hiko_l::GemmScalarRequest request{};
  request.lhs = lhs.data();
  request.rhs = rhs.data();
  request.m = m;
  request.n = n;
  request.k = k;

  std::vector<float> strict_explicit(m * n);
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{},
                       hiko_d::StrictParityTag{},
                       request,
                       strict_explicit.data());
  std::vector<float> fast_explicit(m * n);
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{},
                       hiko_d::FastParityTag{},
                       request,
                       fast_explicit.data());

  std::vector<float> auto_selected(m * n);
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{}, request, auto_selected.data());

  const hiko_d::GemmParityMode mode = hiko_d::active_gemm_parity_mode();
  const std::vector<float>& expected =
      mode == hiko_d::GemmParityMode::Fast ? fast_explicit : strict_explicit;
  if (!bit_equal(expected, auto_selected)) {
    fail("active_gemm_parity_mode must drive the unqualified dispatch");
  }
}

void test_dispatch_table_pointers_match_explicit_overloads() {
  constexpr std::size_t m = 13;
  constexpr std::size_t n = 64;
  constexpr std::size_t k = 64;
  const std::vector<float> lhs = make_matrix(m, k, 21);
  const std::vector<float> rhs = make_matrix(n, k, 23);

  hiko_l::GemmScalarRequest request{};
  request.lhs = lhs.data();
  request.rhs = rhs.data();
  request.m = m;
  request.n = n;
  request.k = k;

  const hiko_d::DispatchTable& table = hiko_d::selected_dispatch_table();
  if (table.gemm_nt_strict == nullptr || table.gemm_nt_fast == nullptr) {
    fail("dispatch table must carry both strict and fast GEMM nt pointers");
  }
  if (table.gemm_nn_strict == nullptr || table.gemm_nn_fast == nullptr) {
    fail("dispatch table must carry both strict and fast GEMM nn pointers");
  }

  std::vector<float> table_strict(m * n);
  table.gemm_nt_strict(request, table_strict.data());
  std::vector<float> overload_strict(m * n);
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{},
                       hiko_d::StrictParityTag{},
                       request,
                       overload_strict.data());
  if (!bit_equal(table_strict, overload_strict)) {
    fail("dispatch_table.gemm_nt_strict must reproduce StrictParityTag overload");
  }

  std::vector<float> table_fast(m * n);
  table.gemm_nt_fast(request, table_fast.data());
  std::vector<float> overload_fast(m * n);
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{},
                       hiko_d::FastParityTag{},
                       request,
                       overload_fast.data());
  if (!bit_equal(table_fast, overload_fast)) {
    fail("dispatch_table.gemm_nt_fast must reproduce FastParityTag overload");
  }
}

}  // namespace

int main() {
  // MPNN-64 hot shapes: encoder W_e style (large M, K=N=64),
  // 192->64 / 416->64 / 256->64 message + FFN W_out (n=64, k>=192),
  // FFN W_in (n=256, k=64), and similarity (Lq x 64 x Lt) for small L.
  test_strict_mode_matches_primitive(64, 64, 64, "encoder_we");
  test_strict_mode_matches_primitive(64, 64, 192, "message_w3");
  test_strict_mode_matches_primitive(64, 64, 256, "ffn_w_out");
  test_strict_mode_matches_primitive(64, 64, 416, "edge_update");
  test_strict_mode_matches_primitive(80, 256, 64, "ffn_w_in");
  test_strict_mode_matches_primitive(179, 179, 64, "similarity_l179");
  // Generic non-MPNN-64 shape to exercise the strict K-blocked fallback.
  test_strict_mode_matches_primitive(17, 23, 100, "generic_fallback");

  test_fast_mode_within_tolerance(64, 64, 64, "encoder_we");
  test_fast_mode_within_tolerance(64, 64, 192, "message_w3");
  test_fast_mode_within_tolerance(64, 64, 256, "ffn_w_out");
  test_fast_mode_within_tolerance(64, 64, 416, "edge_update");
  test_fast_mode_within_tolerance(80, 256, 64, "ffn_w_in");
  test_fast_mode_within_tolerance(179, 179, 64, "similarity_l179");
  test_fast_mode_within_tolerance(17, 23, 100, "generic_fallback");

  test_runtime_selector_via_env();
  test_dispatch_table_pointers_match_explicit_overloads();
  return 0;
}

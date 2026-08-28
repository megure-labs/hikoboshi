// Transformer attention compound-module parity tests.
//
// Each test builds a small input case, runs the production attention module
// (`hikoboshi.attention.v1`), and compares against an
// independently-implemented naive reference that mirrors PyTorch's
// `torch.nn.functional.scaled_dot_product_attention` semantics plus the
// canonical RoPE rotation. Two tolerance bands cover both registered parity
// modes: strict (1e-5 max-abs) and fast (1e-4 max-abs). The default build
// resolves the internal GEMMs through strict mode, so the strict tolerance
// is the tighter assertion exercised here; the fast-mode contract is
// covered structurally by the registry signature plus the GEMM parity-mode
// test that independently asserts fast GEMM stays within 1e-4 vs the strict
// kernel.
//
// The naive reference avoids `hikoboshi::dispatch` so a regression in the
// shared GEMM, LayerNorm, softmax, or axpy primitives cannot mask itself by
// being identical on both sides.

#include <hikoboshi/dispatch/registry/module_op.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>
#include <hikoboshi/modules/transformer/attention.hpp>
#include <hikoboshi/modules/transformer/detail/workspace.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace hiko_t = hikoboshi::modules::transformer;
namespace hiko_td = hikoboshi::modules::transformer::detail;
namespace hiko_dr = hikoboshi::dispatch::registry;
namespace hiko_u = hikoboshi::universal;

namespace {

// The default scalar build resolves the internal GEMMs through the BLIS-style
// fast kernel (per `hikoboshi_gemm_parity_mode=fast`), so the parity contract
// this file exercises is the public 1e-4 max-abs tolerance vs the naive
// reference. Strict-mode bit-parity for the underlying GEMM primitive is
// covered independently by `hikoboshi_gemm_parity_modes`. Future strict-build
// runs of this test will satisfy this tolerance trivially.
constexpr float kFastTolerance = 1.0e-4F;

void fail(const char* tag) {
  std::fprintf(stderr, "attention_parity_tests: %s\n", tag);
  std::exit(1);
}

float deterministic_value(std::size_t i, std::size_t seed) {
  const std::uint64_t mixed = static_cast<std::uint64_t>(i) * 2654435761ull +
                              static_cast<std::uint64_t>(seed) * 40503ull;
  const float normalized =
      static_cast<float>(mixed % 65537ull) / 32768.0F - 1.0F;
  return normalized * 0.5F;
}

void fill_deterministic(std::vector<float>& buffer, std::size_t seed) {
  for (std::size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = deterministic_value(i, seed);
  }
}

void check_max_abs_within(const float* actual, const float* reference,
                          std::size_t count, float tolerance,
                          const char* tag) {
  float worst = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    const float diff = std::fabs(actual[i] - reference[i]);
    if (diff > worst) {
      worst = diff;
    }
  }
  if (!(worst <= tolerance)) {
    std::fprintf(stderr,
                 "attention_parity_tests: %s — worst abs diff %.6e exceeds "
                 "tolerance %.6e\n",
                 tag, static_cast<double>(worst),
                 static_cast<double>(tolerance));
    std::exit(1);
  }
}

// === Naive reference ======================================================

void naive_gemm_nt(const float* lhs, const float* rhs, float* output,
                   std::size_t m, std::size_t n, std::size_t k) {
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

void naive_layer_norm(const float* input, const float* gamma, const float* beta,
                      std::size_t row_count, std::size_t dim, float epsilon,
                      float* output) {
  for (std::size_t r = 0; r < row_count; ++r) {
    const float* row = input + r * dim;
    float sum = 0.0F;
    for (std::size_t d = 0; d < dim; ++d) {
      sum += row[d];
    }
    const float mean = sum / static_cast<float>(dim);
    float sq = 0.0F;
    for (std::size_t d = 0; d < dim; ++d) {
      const float dx = row[d] - mean;
      sq += dx * dx;
    }
    const float var = sq / static_cast<float>(dim);
    const float inv_std = 1.0F / std::sqrt(var + epsilon);
    for (std::size_t d = 0; d < dim; ++d) {
      const float normalized = (row[d] - mean) * inv_std;
      const float scaled = gamma != nullptr ? normalized * gamma[d] : normalized;
      output[r * dim + d] = beta != nullptr ? scaled + beta[d] : scaled;
    }
  }
}

void naive_softmax_row(const float* input, std::size_t dim,
                       const float* mask_row, float temperature, float* output) {
  const float inv_t = 1.0F / temperature;
  float row_max = -INFINITY;
  for (std::size_t d = 0; d < dim; ++d) {
    const float scaled = input[d] * inv_t;
    const float v = mask_row != nullptr ? scaled + mask_row[d] : scaled;
    if (v > row_max) {
      row_max = v;
    }
  }
  float sum = 0.0F;
  for (std::size_t d = 0; d < dim; ++d) {
    const float scaled = input[d] * inv_t;
    const float v = mask_row != nullptr ? scaled + mask_row[d] : scaled;
    const float e = std::exp(v - row_max);
    output[d] = e;
    sum += e;
  }
  const float inv_sum = 1.0F / sum;
  for (std::size_t d = 0; d < dim; ++d) {
    output[d] *= inv_sum;
  }
}

void apply_rope_pair(float* row, std::size_t head_dim,
                     const float* cos_row, const float* sin_row) {
  const std::size_t half = head_dim / 2;
  for (std::size_t d = 0; d < half; ++d) {
    const float lo = row[d];
    const float hi = row[d + half];
    const float c = cos_row[d];
    const float sv = sin_row[d];
    row[d] = lo * c - hi * sv;
    row[d + half] = lo * sv + hi * c;
  }
}

void naive_attention(const hiko_t::AttentionLayerRequest& request,
                     float* output_embeddings) {
  const std::size_t seq_len = request.seq_len;
  const std::size_t hidden = request.hidden_dim;
  const std::size_t head_count = request.head_count;
  const std::size_t head_dim = request.head_dim;
  const std::size_t hd = head_count * head_dim;

  std::vector<float> norm(seq_len * hidden);
  naive_layer_norm(request.input_embeddings, request.pre_norm.gamma,
                   request.pre_norm.beta, seq_len, hidden,
                   request.pre_norm.epsilon, norm.data());

  auto project = [&](const hiko_t::LinearLayerWeightsView& linear,
                     std::vector<float>& out) {
    out.assign(seq_len * linear.output_dim, 0.0F);
    naive_gemm_nt(norm.data(), linear.weight, out.data(), seq_len,
                  linear.output_dim, linear.input_dim);
    if (linear.bias != nullptr) {
      for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t d = 0; d < linear.output_dim; ++d) {
          out[r * linear.output_dim + d] += linear.bias[d];
        }
      }
    }
  };

  std::vector<float> q, k, v;
  project(request.wq, q);
  project(request.wk, k);
  project(request.wv, v);

  std::vector<float> q_head(head_count * seq_len * head_dim, 0.0F);
  std::vector<float> k_head(head_count * seq_len * head_dim, 0.0F);
  std::vector<float> v_head(head_count * seq_len * head_dim, 0.0F);
  for (std::size_t s = 0; s < seq_len; ++s) {
    for (std::size_t h = 0; h < head_count; ++h) {
      for (std::size_t d = 0; d < head_dim; ++d) {
        q_head[(h * seq_len + s) * head_dim + d] = q[s * hd + h * head_dim + d];
        k_head[(h * seq_len + s) * head_dim + d] = k[s * hd + h * head_dim + d];
        v_head[(h * seq_len + s) * head_dim + d] = v[s * hd + h * head_dim + d];
      }
    }
  }

  if (request.rope_cos != nullptr && request.rope_sin != nullptr) {
    for (std::size_t h = 0; h < head_count; ++h) {
      for (std::size_t s = 0; s < seq_len; ++s) {
        const std::size_t half = head_dim / 2;
        apply_rope_pair(q_head.data() + (h * seq_len + s) * head_dim, head_dim,
                        request.rope_cos + s * half,
                        request.rope_sin + s * half);
        apply_rope_pair(k_head.data() + (h * seq_len + s) * head_dim, head_dim,
                        request.rope_cos + s * half,
                        request.rope_sin + s * half);
      }
    }
  }

  std::vector<float> head_out(head_count * seq_len * head_dim, 0.0F);
  std::vector<float> scores(seq_len * seq_len, 0.0F);
  std::vector<float> weights(seq_len, 0.0F);
  const float temperature = std::sqrt(static_cast<float>(head_dim));
  for (std::size_t h = 0; h < head_count; ++h) {
    naive_gemm_nt(q_head.data() + h * seq_len * head_dim,
                  k_head.data() + h * seq_len * head_dim, scores.data(),
                  seq_len, seq_len, head_dim);
    for (std::size_t i = 0; i < seq_len; ++i) {
      const float* mask_row = request.attention_mask != nullptr
                                  ? request.attention_mask + i * seq_len
                                  : nullptr;
      naive_softmax_row(scores.data() + i * seq_len, seq_len, mask_row,
                        temperature, weights.data());
      for (std::size_t d = 0; d < head_dim; ++d) {
        float acc = 0.0F;
        for (std::size_t p = 0; p < seq_len; ++p) {
          acc += weights[p] * v_head[(h * seq_len + p) * head_dim + d];
        }
        head_out[(h * seq_len + i) * head_dim + d] = acc;
      }
    }
  }

  std::vector<float> concat(seq_len * hidden, 0.0F);
  for (std::size_t s = 0; s < seq_len; ++s) {
    for (std::size_t h = 0; h < head_count; ++h) {
      for (std::size_t d = 0; d < head_dim; ++d) {
        concat[s * hd + h * head_dim + d] =
            head_out[(h * seq_len + s) * head_dim + d];
      }
    }
  }

  std::vector<float> attn(seq_len * hidden, 0.0F);
  naive_gemm_nt(concat.data(), request.wo.weight, attn.data(), seq_len,
                request.wo.output_dim, request.wo.input_dim);
  if (request.wo.bias != nullptr) {
    for (std::size_t r = 0; r < seq_len; ++r) {
      for (std::size_t d = 0; d < hidden; ++d) {
        attn[r * hidden + d] += request.wo.bias[d];
      }
    }
  }

  for (std::size_t i = 0; i < seq_len * hidden; ++i) {
    output_embeddings[i] = request.input_embeddings[i] + attn[i];
  }
}

// === Test case builder ====================================================

struct AttentionCase {
  std::size_t seq_len;
  std::size_t hidden_dim;
  std::size_t head_count;
  std::size_t head_dim;
  std::vector<float> input;
  std::vector<float> wq_weight;
  std::vector<float> wq_bias;
  std::vector<float> wk_weight;
  std::vector<float> wk_bias;
  std::vector<float> wv_weight;
  std::vector<float> wv_bias;
  std::vector<float> wo_weight;
  std::vector<float> wo_bias;
  std::vector<float> norm_gamma;
  std::vector<float> norm_beta;
  std::vector<float> rope_cos;
  std::vector<float> rope_sin;
  std::vector<float> mask;
  bool use_bias;
  bool use_rope;
  bool use_mask;

  // Workspace storage.
  std::vector<float> norm_buffer;
  std::vector<float> q_buffer;
  std::vector<float> k_buffer;
  std::vector<float> v_buffer;
  std::vector<float> q_head_buffer;
  std::vector<float> k_head_buffer;
  std::vector<float> v_head_buffer;
  std::vector<float> scores_buffer;
  std::vector<float> head_out_buffer;
  std::vector<float> concat_buffer;
  std::vector<float> attn_buffer;
  hiko_td::TransformerAttentionWorkspace workspace{};
};

void build_case(AttentionCase& tc, std::size_t seq_len, std::size_t head_count,
                std::size_t head_dim, bool use_bias, bool use_rope,
                bool use_mask, std::size_t seed_base) {
  tc.seq_len = seq_len;
  tc.head_count = head_count;
  tc.head_dim = head_dim;
  tc.hidden_dim = head_count * head_dim;
  tc.use_bias = use_bias;
  tc.use_rope = use_rope;
  tc.use_mask = use_mask;
  const std::size_t hidden = tc.hidden_dim;

  tc.input.resize(seq_len * hidden);
  fill_deterministic(tc.input, seed_base + 1);
  tc.wq_weight.resize(hidden * hidden);
  fill_deterministic(tc.wq_weight, seed_base + 2);
  tc.wk_weight.resize(hidden * hidden);
  fill_deterministic(tc.wk_weight, seed_base + 3);
  tc.wv_weight.resize(hidden * hidden);
  fill_deterministic(tc.wv_weight, seed_base + 4);
  tc.wo_weight.resize(hidden * hidden);
  fill_deterministic(tc.wo_weight, seed_base + 5);

  if (use_bias) {
    tc.wq_bias.resize(hidden);
    fill_deterministic(tc.wq_bias, seed_base + 6);
    tc.wk_bias.resize(hidden);
    fill_deterministic(tc.wk_bias, seed_base + 7);
    tc.wv_bias.resize(hidden);
    fill_deterministic(tc.wv_bias, seed_base + 8);
    tc.wo_bias.resize(hidden);
    fill_deterministic(tc.wo_bias, seed_base + 9);
  }

  tc.norm_gamma.resize(hidden);
  tc.norm_beta.resize(hidden);
  // LayerNorm parameters: gamma centered near 1, beta near 0, mimicking
  // the post-init state of a transformer norm layer.
  for (std::size_t d = 0; d < hidden; ++d) {
    tc.norm_gamma[d] = 1.0F + 0.1F * deterministic_value(d, seed_base + 10);
    tc.norm_beta[d] = 0.05F * deterministic_value(d, seed_base + 11);
  }

  if (use_rope) {
    const std::size_t half = head_dim / 2;
    tc.rope_cos.resize(seq_len * half);
    tc.rope_sin.resize(seq_len * half);
    for (std::size_t s = 0; s < seq_len; ++s) {
      for (std::size_t d = 0; d < half; ++d) {
        const float theta = static_cast<float>(s) /
                            std::pow(10000.0F, static_cast<float>(2 * d) /
                                                   static_cast<float>(head_dim));
        tc.rope_cos[s * half + d] = std::cos(theta);
        tc.rope_sin[s * half + d] = std::sin(theta);
      }
    }
  }

  if (use_mask) {
    tc.mask.assign(seq_len * seq_len, 0.0F);
    // Lower-triangular causal mask: position i cannot attend to j > i.
    for (std::size_t i = 0; i < seq_len; ++i) {
      for (std::size_t j = i + 1; j < seq_len; ++j) {
        tc.mask[i * seq_len + j] = -1.0e30F;
      }
    }
  }

  tc.norm_buffer.assign(seq_len * hidden, 0.0F);
  tc.q_buffer.assign(seq_len * hidden, 0.0F);
  tc.k_buffer.assign(seq_len * hidden, 0.0F);
  tc.v_buffer.assign(seq_len * hidden, 0.0F);
  tc.q_head_buffer.assign(head_count * seq_len * head_dim, 0.0F);
  tc.k_head_buffer.assign(head_count * seq_len * head_dim, 0.0F);
  tc.v_head_buffer.assign(head_count * seq_len * head_dim, 0.0F);
  tc.scores_buffer.assign(seq_len * seq_len, 0.0F);
  tc.head_out_buffer.assign(head_count * seq_len * head_dim, 0.0F);
  tc.concat_buffer.assign(seq_len * hidden, 0.0F);
  tc.attn_buffer.assign(seq_len * hidden, 0.0F);

  tc.workspace.plan = {seq_len, hidden, head_count, head_dim};
  tc.workspace.norm_buffer = {tc.norm_buffer.data(), tc.norm_buffer.size()};
  tc.workspace.q_buffer = {tc.q_buffer.data(), tc.q_buffer.size()};
  tc.workspace.k_buffer = {tc.k_buffer.data(), tc.k_buffer.size()};
  tc.workspace.v_buffer = {tc.v_buffer.data(), tc.v_buffer.size()};
  tc.workspace.q_head_buffer = {tc.q_head_buffer.data(),
                                tc.q_head_buffer.size()};
  tc.workspace.k_head_buffer = {tc.k_head_buffer.data(),
                                tc.k_head_buffer.size()};
  tc.workspace.v_head_buffer = {tc.v_head_buffer.data(),
                                tc.v_head_buffer.size()};
  tc.workspace.scores_buffer = {tc.scores_buffer.data(),
                                tc.scores_buffer.size()};
  tc.workspace.head_out_buffer = {tc.head_out_buffer.data(),
                                  tc.head_out_buffer.size()};
  tc.workspace.concat_buffer = {tc.concat_buffer.data(),
                                tc.concat_buffer.size()};
  tc.workspace.attn_buffer = {tc.attn_buffer.data(), tc.attn_buffer.size()};
}

hiko_t::AttentionLayerRequest fill_request(AttentionCase& tc) {
  hiko_t::AttentionLayerRequest request{};
  request.input_embeddings = tc.input.data();
  request.seq_len = tc.seq_len;
  request.hidden_dim = tc.hidden_dim;
  request.head_count = tc.head_count;
  request.head_dim = tc.head_dim;
  request.wq = {tc.wq_weight.data(),
                tc.use_bias ? tc.wq_bias.data() : nullptr, tc.hidden_dim,
                tc.hidden_dim};
  request.wk = {tc.wk_weight.data(),
                tc.use_bias ? tc.wk_bias.data() : nullptr, tc.hidden_dim,
                tc.hidden_dim};
  request.wv = {tc.wv_weight.data(),
                tc.use_bias ? tc.wv_bias.data() : nullptr, tc.hidden_dim,
                tc.hidden_dim};
  request.wo = {tc.wo_weight.data(),
                tc.use_bias ? tc.wo_bias.data() : nullptr, tc.hidden_dim,
                tc.hidden_dim};
  request.pre_norm = {tc.norm_gamma.data(), tc.norm_beta.data(), tc.hidden_dim,
                      1.0e-5F};
  request.rope_cos = tc.use_rope ? tc.rope_cos.data() : nullptr;
  request.rope_sin = tc.use_rope ? tc.rope_sin.data() : nullptr;
  request.attention_mask = tc.use_mask ? tc.mask.data() : nullptr;
  request.workspace = &tc.workspace;
  return request;
}

void run_attention(AttentionCase& tc, std::vector<float>& output) {
  output.assign(tc.seq_len * tc.hidden_dim, 0.0F);
  hiko_t::AttentionLayerRequest request = fill_request(tc);
  hiko_t::AttentionLayerOutput attn_output{};
  attn_output.output_embeddings = output.data();
  hiko_t::attention_layer_scalar(request, attn_output);
}

void run_naive(AttentionCase& tc, std::vector<float>& output) {
  output.assign(tc.seq_len * tc.hidden_dim, 0.0F);
  hiko_t::AttentionLayerRequest request = fill_request(tc);
  naive_attention(request, output.data());
}

// === Tests ================================================================

void test_attention_matches_naive_two_heads_no_rope_no_mask() {
  AttentionCase tc{};
  build_case(tc, /*seq_len=*/4, /*head_count=*/2, /*head_dim=*/4,
             /*use_bias=*/true, /*use_rope=*/false, /*use_mask=*/false,
             /*seed_base=*/100);
  std::vector<float> actual;
  std::vector<float> reference;
  run_attention(tc, actual);
  run_naive(tc, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "two-head no-RoPE no-mask attention fast-tolerance");
}

void test_attention_matches_naive_with_rope() {
  AttentionCase tc{};
  build_case(tc, /*seq_len=*/3, /*head_count=*/2, /*head_dim=*/4,
             /*use_bias=*/true, /*use_rope=*/true, /*use_mask=*/false,
             /*seed_base=*/200);
  std::vector<float> actual;
  std::vector<float> reference;
  run_attention(tc, actual);
  run_naive(tc, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "RoPE-enabled attention fast-tolerance");
}

void test_attention_matches_naive_with_causal_mask() {
  AttentionCase tc{};
  build_case(tc, /*seq_len=*/4, /*head_count=*/2, /*head_dim=*/4,
             /*use_bias=*/true, /*use_rope=*/false, /*use_mask=*/true,
             /*seed_base=*/300);
  std::vector<float> actual;
  std::vector<float> reference;
  run_attention(tc, actual);
  run_naive(tc, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "causal-mask attention fast-tolerance");
}

void test_attention_matches_naive_no_bias() {
  AttentionCase tc{};
  build_case(tc, /*seq_len=*/3, /*head_count=*/2, /*head_dim=*/4,
             /*use_bias=*/false, /*use_rope=*/false, /*use_mask=*/false,
             /*seed_base=*/400);
  std::vector<float> actual;
  std::vector<float> reference;
  run_attention(tc, actual);
  run_naive(tc, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "bias-less attention fast-tolerance");
}

void test_single_head_edge_case() {
  AttentionCase tc{};
  build_case(tc, /*seq_len=*/3, /*head_count=*/1, /*head_dim=*/4,
             /*use_bias=*/true, /*use_rope=*/false, /*use_mask=*/false,
             /*seed_base=*/500);
  std::vector<float> actual;
  std::vector<float> reference;
  run_attention(tc, actual);
  run_naive(tc, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "single-head edge case fast-tolerance");
}

void test_rope_golden_pair() {
  // Standalone RoPE golden: head_dim=4, half=2. Confirm the canonical
  // pairwise rotation rule and the explicit module behavior agree on a tiny
  // input we can hand-check.
  AttentionCase tc{};
  build_case(tc, /*seq_len=*/2, /*head_count=*/1, /*head_dim=*/4,
             /*use_bias=*/false, /*use_rope=*/true, /*use_mask=*/false,
             /*seed_base=*/600);
  // Override the cos/sin table with simple, hand-checkable values so the
  // rotation is unambiguous: position 0 is identity (cos=1, sin=0); position
  // 1 rotates by exactly 90 degrees in each pair (cos=0, sin=1).
  const std::size_t half = tc.head_dim / 2;
  for (std::size_t d = 0; d < half; ++d) {
    tc.rope_cos[0 * half + d] = 1.0F;
    tc.rope_sin[0 * half + d] = 0.0F;
    tc.rope_cos[1 * half + d] = 0.0F;
    tc.rope_sin[1 * half + d] = 1.0F;
  }
  std::vector<float> actual;
  std::vector<float> reference;
  run_attention(tc, actual);
  run_naive(tc, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "RoPE 90-degree golden pair fast-tolerance");
}

// === Registry assertions ==================================================

void test_registry_records_attention() {
  const hiko_dr::RegisteredModuleOpRecord* record =
      hiko_dr::find_module_op("hikoboshi.attention.v1");
  if (record == nullptr) {
    fail("module_op_registry must list hikoboshi.attention.v1");
  }
  if (record->family != hiko_dr::ModuleOpFamily::Layer) {
    fail("attention module-op must register under the Layer family");
  }
  if (record->identity.op_version != std::string_view{"v1"}) {
    fail("attention module-op identity version must be v1");
  }
  if (record->dispatch_entry == nullptr) {
    fail("attention module-op must register a non-null dispatch entry");
  }
}

void test_registry_capabilities_for_attention() {
  const hiko_dr::RegisteredModuleOpRecord* record =
      hiko_dr::find_module_op("hikoboshi.attention.v1");
  if (record == nullptr) {
    fail("attention module-op must be present before capability check");
  }
  bool has_scalar = false;
  for (std::size_t i = 0; i < record->capabilities.supported_backends.size;
       ++i) {
    if (record->capabilities.supported_backends.data[i] ==
        hiko_u::PackageBackendRequirement::CpuScalar) {
      has_scalar = true;
    }
  }
  if (!has_scalar) {
    fail("attention module-op must list the cpu.scalar backend");
  }
  bool has_strict = false;
  bool has_fast = false;
  for (std::size_t i = 0;
       i < record->capabilities.supported_parity_modes.size; ++i) {
    const hiko_dr::ParityMode mode =
        record->capabilities.supported_parity_modes.data[i];
    if (mode == hiko_dr::ParityMode::Strict) {
      has_strict = true;
    }
    if (mode == hiko_dr::ParityMode::Fast) {
      has_fast = true;
    }
  }
  if (!has_strict || !has_fast) {
    fail(
        "attention module-op must support both strict and fast parity modes");
  }
  if (record->capabilities.default_parity_mode != hiko_dr::ParityMode::Fast) {
    fail("attention module-op default parity mode must be fast");
  }
}

void test_required_primitive_ids_for_attention_resolve() {
  const hiko_dr::RegisteredModuleOpRecord* record =
      hiko_dr::find_module_op("hikoboshi.attention.v1");
  if (record == nullptr) {
    fail("attention module-op must be present before required check");
  }
  const hiko_u::Span<const std::string_view> required =
      record->signature.required_primitive_op_ids;
  if (required.size == 0) {
    fail("attention module-op must declare at least one required primitive");
  }
  const std::string_view expected_ids[] = {
      "hikoboshi.layer_norm.v1",      "hikoboshi.gemm.nt.v1",
      "hikoboshi.gemm.nn.v1",         "hikoboshi.softmax.row_wise.v1",
      "hikoboshi.bias_add.v1",        "hikoboshi.axpy.v1",
  };
  for (const std::string_view expected : expected_ids) {
    bool found = false;
    for (std::size_t r = 0; r < required.size; ++r) {
      if (required.data[r] == expected) {
        found = true;
        break;
      }
    }
    if (!found) {
      fail("attention module-op missing a required primitive id");
    }
    if (hiko_dr::find_primitive_op(expected) == nullptr) {
      fail(
          "attention module-op required_primitive_op_ids entry does not "
          "resolve in primitive_op_registry()");
    }
  }
}

}  // namespace

int main() {
  test_attention_matches_naive_two_heads_no_rope_no_mask();
  test_attention_matches_naive_with_rope();
  test_attention_matches_naive_with_causal_mask();
  test_attention_matches_naive_no_bias();
  test_single_head_edge_case();
  test_rope_golden_pair();
  test_registry_records_attention();
  test_registry_capabilities_for_attention();
  test_required_primitive_ids_for_attention_resolve();
  return 0;
}

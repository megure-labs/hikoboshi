// ESM2-8M encoder forward-pass parity tests.
//
// The production `hikoboshi.esm2.v1.encoder` compound module is exercised
// against an independently-implemented naive reference that mirrors the
// ESM2/RoFormer block topology (pre-norm attention with RoPE on Q/K,
// pre-norm FFN with exact-erf GELU, residual sums, final LayerNorm).
// The reference avoids `hikoboshi::dispatch` so a regression in the shared
// GEMM, LayerNorm, softmax, GELU, or axpy primitives cannot mask itself
// by being identical on both sides.
//
// Hikoboshi 0.1.0's scalar build resolves the internal GEMMs through the
// BLIS-style fast kernel by default (`hikoboshi_gemm_parity_mode=fast`),
// so the parity contract this file exercises is the public 1e-4 max-abs
// tolerance against the naive reference. Strict-mode bit-parity for the
// underlying GEMM primitive is covered independently by
// `hikoboshi_gemm_parity_modes`. Bit-tight parity against real PyTorch
// ESM2-8M activations on Casey's fine-tuned checkpoint is enforced by
// `tests/cpp/esm2_pytorch_real_parity.cpp` (committed fixture under
// `tests/cpp/data/esm2_pytorch_goldens.npz`); the shape envelope
// (vocab=29 matching Casey's checkpoint, 6 layers, hidden=320,
// head_count=20, head_dim=16, ffn_hidden=1280) is exercised here at
// small dimensions so the failure surface is local to the modules layer.

#include <hikoboshi/dispatch/registry/architecture.hpp>
#include <hikoboshi/dispatch/registry/module_op.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>
#include <hikoboshi/modules/esm2.hpp>
#include <hikoboshi/modules/detail/esm2_layers.hpp>
#include <hikoboshi/modules/detail/esm2_workspace.hpp>
#include <hikoboshi/modules/transformer/detail/workspace.hpp>
#include <hikoboshi/modules/ffn/detail/workspace.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace hiko_m = hikoboshi::modules;
namespace hiko_md = hikoboshi::modules::detail;
namespace hiko_mc = hikoboshi::modules::common;
namespace hiko_mt = hikoboshi::modules::transformer::detail;
namespace hiko_mf = hikoboshi::modules::ffn::detail;
namespace hiko_dr = hikoboshi::dispatch::registry;
namespace hiko_u = hikoboshi::universal;

namespace {

constexpr float kFastTolerance = 1.0e-4F;

void fail(const char* tag) {
  std::fprintf(stderr, "esm2_forward_parity_tests: %s\n", tag);
  std::exit(1);
}

// === Deterministic value generation =======================================

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
  std::size_t worst_index = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const float diff = std::fabs(actual[i] - reference[i]);
    if (diff > worst) {
      worst = diff;
      worst_index = i;
    }
  }
  if (!(worst <= tolerance)) {
    std::fprintf(stderr,
                 "esm2_forward_parity_tests: %s — worst abs diff %.6e at "
                 "index %zu exceeds tolerance %.6e\n",
                 tag, static_cast<double>(worst), worst_index,
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

void naive_softmax_row(const float* input, std::size_t dim, float temperature,
                       float* output) {
  const float inv_t = 1.0F / temperature;
  float row_max = -INFINITY;
  for (std::size_t d = 0; d < dim; ++d) {
    const float v = input[d] * inv_t;
    if (v > row_max) {
      row_max = v;
    }
  }
  float sum = 0.0F;
  for (std::size_t d = 0; d < dim; ++d) {
    const float v = input[d] * inv_t;
    const float e = std::exp(v - row_max);
    output[d] = e;
    sum += e;
  }
  const float inv_sum = 1.0F / sum;
  for (std::size_t d = 0; d < dim; ++d) {
    output[d] *= inv_sum;
  }
}

void naive_apply_rope_pair(float* row, std::size_t head_dim,
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

float naive_gelu_exact(float x) {
  const double half = 0.5;
  const double inv_sqrt_two = 0.70710678118654752440;
  return static_cast<float>(static_cast<double>(x) * half *
                            (1.0 + std::erf(static_cast<double>(x) *
                                            inv_sqrt_two)));
}

void naive_attention_block(const hiko_md::Esm2LayerWeights& w, std::size_t seq_len,
                           std::size_t hidden, std::size_t head_count,
                           std::size_t head_dim, const float* rope_cos,
                           const float* rope_sin, const float* input,
                           float* output) {
  const std::size_t hd = hidden;

  std::vector<float> norm(seq_len * hidden);
  naive_layer_norm(input, w.attn_pre_norm.gamma, w.attn_pre_norm.beta, seq_len,
                   hidden, w.attn_pre_norm.epsilon, norm.data());

  auto project = [&](const hiko_mc::LinearLayerWeightsView& linear,
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
  project(w.wq, q);
  project(w.wk, k);
  project(w.wv, v);

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

  if (rope_cos != nullptr && rope_sin != nullptr) {
    const std::size_t half = head_dim / 2;
    for (std::size_t h = 0; h < head_count; ++h) {
      for (std::size_t s = 0; s < seq_len; ++s) {
        naive_apply_rope_pair(
            q_head.data() + (h * seq_len + s) * head_dim, head_dim,
            rope_cos + s * half, rope_sin + s * half);
        naive_apply_rope_pair(
            k_head.data() + (h * seq_len + s) * head_dim, head_dim,
            rope_cos + s * half, rope_sin + s * half);
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
      naive_softmax_row(scores.data() + i * seq_len, seq_len, temperature,
                        weights.data());
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
  naive_gemm_nt(concat.data(), w.wo.weight, attn.data(), seq_len,
                w.wo.output_dim, w.wo.input_dim);
  if (w.wo.bias != nullptr) {
    for (std::size_t r = 0; r < seq_len; ++r) {
      for (std::size_t d = 0; d < hidden; ++d) {
        attn[r * hidden + d] += w.wo.bias[d];
      }
    }
  }

  for (std::size_t i = 0; i < seq_len * hidden; ++i) {
    output[i] = input[i] + attn[i];
  }
}

void naive_ffn_block(const hiko_md::Esm2LayerWeights& w, std::size_t seq_len,
                     std::size_t hidden, std::size_t ffn_hidden,
                     const float* input, float* output) {
  std::vector<float> norm(seq_len * hidden);
  naive_layer_norm(input, w.ffn_pre_norm.gamma, w.ffn_pre_norm.beta, seq_len,
                   hidden, w.ffn_pre_norm.epsilon, norm.data());

  std::vector<float> intermediate(seq_len * ffn_hidden, 0.0F);
  naive_gemm_nt(norm.data(), w.ffn_in.weight, intermediate.data(), seq_len,
                ffn_hidden, hidden);
  if (w.ffn_in.bias != nullptr) {
    for (std::size_t r = 0; r < seq_len; ++r) {
      for (std::size_t d = 0; d < ffn_hidden; ++d) {
        intermediate[r * ffn_hidden + d] += w.ffn_in.bias[d];
      }
    }
  }
  for (float& value : intermediate) {
    value = naive_gelu_exact(value);
  }

  std::vector<float> projected(seq_len * hidden, 0.0F);
  naive_gemm_nt(intermediate.data(), w.ffn_out.weight, projected.data(),
                seq_len, hidden, ffn_hidden);
  if (w.ffn_out.bias != nullptr) {
    for (std::size_t r = 0; r < seq_len; ++r) {
      for (std::size_t d = 0; d < hidden; ++d) {
        projected[r * hidden + d] += w.ffn_out.bias[d];
      }
    }
  }

  for (std::size_t i = 0; i < seq_len * hidden; ++i) {
    output[i] = input[i] + projected[i];
  }
}

void naive_esm2_forward(const hiko_m::Esm2ForwardRequest& request,
                        float* output_embeddings) {
  const hiko_md::Esm2Weights& weights = *request.weights;
  const hiko_m::Esm2Descriptor& descriptor = request.descriptor;
  const std::size_t seq_len = request.seq_len;
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t head_count = descriptor.head_count;
  const std::size_t head_dim = descriptor.head_dim;
  const std::size_t ffn_hidden = descriptor.ffn_hidden_dimension;

  std::vector<float> hidden_state(seq_len * hidden, 0.0F);
  for (std::size_t s = 0; s < seq_len; ++s) {
    const std::int32_t raw = request.token_ids[s];
    std::size_t row = 0U;
    if (raw >= 0 && static_cast<std::size_t>(raw) < descriptor.vocab_size) {
      row = static_cast<std::size_t>(raw);
    }
    for (std::size_t d = 0; d < hidden; ++d) {
      hidden_state[s * hidden + d] = weights.embedding_table[row * hidden + d];
    }
  }

  std::vector<float> after_attn(seq_len * hidden, 0.0F);
  std::vector<float> after_ffn(seq_len * hidden, 0.0F);
  for (std::size_t layer = 0; layer < descriptor.layer_count; ++layer) {
    const hiko_md::Esm2LayerWeights& w = weights.layers[layer];
    naive_attention_block(w, seq_len, hidden, head_count, head_dim,
                          request.workspace->rope_cos.data,
                          request.workspace->rope_sin.data, hidden_state.data(),
                          after_attn.data());
    naive_ffn_block(w, seq_len, hidden, ffn_hidden, after_attn.data(),
                    after_ffn.data());
    for (std::size_t i = 0; i < seq_len * hidden; ++i) {
      hidden_state[i] = after_ffn[i];
    }
  }

  naive_layer_norm(hidden_state.data(), weights.final_norm.gamma,
                   weights.final_norm.beta, seq_len, hidden,
                   weights.final_norm.epsilon, output_embeddings);
}

// === Test fixture builder ================================================

struct Esm2Case {
  hiko_m::Esm2Descriptor descriptor;
  std::vector<std::int32_t> token_ids;
  std::vector<float> embedding_table;
  std::vector<float> final_norm_gamma;
  std::vector<float> final_norm_beta;

  std::vector<std::vector<float>> attn_pre_norm_gamma;
  std::vector<std::vector<float>> attn_pre_norm_beta;
  std::vector<std::vector<float>> ffn_pre_norm_gamma;
  std::vector<std::vector<float>> ffn_pre_norm_beta;
  std::vector<std::vector<float>> wq_weight;
  std::vector<std::vector<float>> wq_bias;
  std::vector<std::vector<float>> wk_weight;
  std::vector<std::vector<float>> wk_bias;
  std::vector<std::vector<float>> wv_weight;
  std::vector<std::vector<float>> wv_bias;
  std::vector<std::vector<float>> wo_weight;
  std::vector<std::vector<float>> wo_bias;
  std::vector<std::vector<float>> ffn_in_weight;
  std::vector<std::vector<float>> ffn_in_bias;
  std::vector<std::vector<float>> ffn_out_weight;
  std::vector<std::vector<float>> ffn_out_bias;

  std::vector<hiko_md::Esm2LayerWeights> layers;
  hiko_md::Esm2Weights weights{};

  // Workspace storage.
  std::vector<float> hidden_state_buffer;
  std::vector<float> hidden_state_post_attn_buffer;
  std::vector<float> ffn_norm_buffer;
  std::vector<float> ffn_residual_buffer;
  std::vector<float> rope_cos;
  std::vector<float> rope_sin;
  std::vector<float> attn_norm_buffer;
  std::vector<float> attn_q_buffer;
  std::vector<float> attn_k_buffer;
  std::vector<float> attn_v_buffer;
  std::vector<float> attn_q_head_buffer;
  std::vector<float> attn_k_head_buffer;
  std::vector<float> attn_v_head_buffer;
  std::vector<float> attn_scores_buffer;
  std::vector<float> attn_head_out_buffer;
  std::vector<float> attn_concat_buffer;
  std::vector<float> attn_attn_buffer;
  std::vector<float> ffn_intermediate_buffer;
  hiko_md::Esm2Workspace workspace{};
};

void build_case(Esm2Case& tc, std::size_t vocab_size, std::size_t hidden,
                std::size_t head_count, std::size_t head_dim,
                std::size_t ffn_hidden, std::size_t layer_count,
                std::size_t seq_len, std::size_t seed_base) {
  tc.descriptor.vocab_size = vocab_size;
  tc.descriptor.hidden_dimension = hidden;
  tc.descriptor.head_count = head_count;
  tc.descriptor.head_dim = head_dim;
  tc.descriptor.ffn_hidden_dimension = ffn_hidden;
  tc.descriptor.layer_count = layer_count;
  tc.descriptor.max_sequence_length = seq_len;

  // Token ids cycle through the vocab so the embedding-table lookup
  // exercises every row at least once when seq_len >= vocab_size.
  tc.token_ids.assign(seq_len, 0);
  for (std::size_t s = 0; s < seq_len; ++s) {
    tc.token_ids[s] = static_cast<std::int32_t>(s % vocab_size);
  }

  tc.embedding_table.resize(vocab_size * hidden);
  fill_deterministic(tc.embedding_table, seed_base + 1);

  tc.final_norm_gamma.resize(hidden);
  tc.final_norm_beta.resize(hidden);
  for (std::size_t d = 0; d < hidden; ++d) {
    tc.final_norm_gamma[d] = 1.0F + 0.05F * deterministic_value(d, seed_base + 2);
    tc.final_norm_beta[d] = 0.02F * deterministic_value(d, seed_base + 3);
  }

  tc.attn_pre_norm_gamma.assign(layer_count, std::vector<float>(hidden));
  tc.attn_pre_norm_beta.assign(layer_count, std::vector<float>(hidden));
  tc.ffn_pre_norm_gamma.assign(layer_count, std::vector<float>(hidden));
  tc.ffn_pre_norm_beta.assign(layer_count, std::vector<float>(hidden));
  tc.wq_weight.assign(layer_count, std::vector<float>(hidden * hidden));
  tc.wq_bias.assign(layer_count, std::vector<float>(hidden));
  tc.wk_weight.assign(layer_count, std::vector<float>(hidden * hidden));
  tc.wk_bias.assign(layer_count, std::vector<float>(hidden));
  tc.wv_weight.assign(layer_count, std::vector<float>(hidden * hidden));
  tc.wv_bias.assign(layer_count, std::vector<float>(hidden));
  tc.wo_weight.assign(layer_count, std::vector<float>(hidden * hidden));
  tc.wo_bias.assign(layer_count, std::vector<float>(hidden));
  tc.ffn_in_weight.assign(layer_count, std::vector<float>(ffn_hidden * hidden));
  tc.ffn_in_bias.assign(layer_count, std::vector<float>(ffn_hidden));
  tc.ffn_out_weight.assign(layer_count, std::vector<float>(hidden * ffn_hidden));
  tc.ffn_out_bias.assign(layer_count, std::vector<float>(hidden));

  for (std::size_t layer = 0; layer < layer_count; ++layer) {
    const std::size_t base = seed_base + 100U * layer + 10U;
    for (std::size_t d = 0; d < hidden; ++d) {
      tc.attn_pre_norm_gamma[layer][d] =
          1.0F + 0.05F * deterministic_value(d, base + 1);
      tc.attn_pre_norm_beta[layer][d] = 0.02F * deterministic_value(d, base + 2);
      tc.ffn_pre_norm_gamma[layer][d] =
          1.0F + 0.05F * deterministic_value(d, base + 3);
      tc.ffn_pre_norm_beta[layer][d] = 0.02F * deterministic_value(d, base + 4);
    }
    fill_deterministic(tc.wq_weight[layer], base + 5);
    fill_deterministic(tc.wq_bias[layer], base + 6);
    fill_deterministic(tc.wk_weight[layer], base + 7);
    fill_deterministic(tc.wk_bias[layer], base + 8);
    fill_deterministic(tc.wv_weight[layer], base + 9);
    fill_deterministic(tc.wv_bias[layer], base + 10);
    fill_deterministic(tc.wo_weight[layer], base + 11);
    fill_deterministic(tc.wo_bias[layer], base + 12);
    fill_deterministic(tc.ffn_in_weight[layer], base + 13);
    fill_deterministic(tc.ffn_in_bias[layer], base + 14);
    fill_deterministic(tc.ffn_out_weight[layer], base + 15);
    fill_deterministic(tc.ffn_out_bias[layer], base + 16);
  }

  tc.layers.assign(layer_count, hiko_md::Esm2LayerWeights{});
  for (std::size_t layer = 0; layer < layer_count; ++layer) {
    hiko_md::Esm2LayerWeights& w = tc.layers[layer];
    w.attn_pre_norm = {tc.attn_pre_norm_gamma[layer].data(),
                       tc.attn_pre_norm_beta[layer].data(), hidden,
                       hiko_md::kEsm2LayerNormEpsilon};
    w.wq = {tc.wq_weight[layer].data(), tc.wq_bias[layer].data(), hidden,
            hidden};
    w.wk = {tc.wk_weight[layer].data(), tc.wk_bias[layer].data(), hidden,
            hidden};
    w.wv = {tc.wv_weight[layer].data(), tc.wv_bias[layer].data(), hidden,
            hidden};
    w.wo = {tc.wo_weight[layer].data(), tc.wo_bias[layer].data(), hidden,
            hidden};
    w.ffn_pre_norm = {tc.ffn_pre_norm_gamma[layer].data(),
                      tc.ffn_pre_norm_beta[layer].data(), hidden,
                      hiko_md::kEsm2LayerNormEpsilon};
    w.ffn_in = {tc.ffn_in_weight[layer].data(), tc.ffn_in_bias[layer].data(),
                ffn_hidden, hidden};
    w.ffn_out = {tc.ffn_out_weight[layer].data(),
                 tc.ffn_out_bias[layer].data(), hidden, ffn_hidden};
  }

  tc.weights.embedding_table = tc.embedding_table.data();
  tc.weights.vocab_size = vocab_size;
  tc.weights.hidden_dimension = hidden;
  tc.weights.layer_count = layer_count;
  tc.weights.layers = tc.layers.data();
  tc.weights.final_norm = {tc.final_norm_gamma.data(),
                           tc.final_norm_beta.data(), hidden,
                           hiko_md::kEsm2LayerNormEpsilon};

  // Workspace allocation.
  const std::size_t hidden_count = seq_len * hidden;
  const std::size_t half = head_dim / 2U;
  const std::size_t rope_count = seq_len * half;
  tc.hidden_state_buffer.assign(hidden_count, 0.0F);
  tc.hidden_state_post_attn_buffer.assign(hidden_count, 0.0F);
  tc.ffn_norm_buffer.assign(hidden_count, 0.0F);
  tc.ffn_residual_buffer.assign(hidden_count, 0.0F);
  tc.rope_cos.assign(rope_count, 0.0F);
  tc.rope_sin.assign(rope_count, 0.0F);
  for (std::size_t s = 0; s < seq_len; ++s) {
    for (std::size_t d = 0; d < half; ++d) {
      const float theta = static_cast<float>(s) /
                          std::pow(10000.0F, static_cast<float>(2 * d) /
                                                 static_cast<float>(head_dim));
      tc.rope_cos[s * half + d] = std::cos(theta);
      tc.rope_sin[s * half + d] = std::sin(theta);
    }
  }
  tc.attn_norm_buffer.assign(hidden_count, 0.0F);
  tc.attn_q_buffer.assign(hidden_count, 0.0F);
  tc.attn_k_buffer.assign(hidden_count, 0.0F);
  tc.attn_v_buffer.assign(hidden_count, 0.0F);
  const std::size_t head_major_count = head_count * seq_len * head_dim;
  tc.attn_q_head_buffer.assign(head_major_count, 0.0F);
  tc.attn_k_head_buffer.assign(head_major_count, 0.0F);
  tc.attn_v_head_buffer.assign(head_major_count, 0.0F);
  tc.attn_scores_buffer.assign(seq_len * seq_len, 0.0F);
  tc.attn_head_out_buffer.assign(head_major_count, 0.0F);
  tc.attn_concat_buffer.assign(hidden_count, 0.0F);
  tc.attn_attn_buffer.assign(hidden_count, 0.0F);
  tc.ffn_intermediate_buffer.assign(seq_len * ffn_hidden, 0.0F);

  tc.workspace.plan = {seq_len, hidden, head_count, head_dim, ffn_hidden};
  tc.workspace.hidden_state = {tc.hidden_state_buffer.data(),
                               tc.hidden_state_buffer.size()};
  tc.workspace.hidden_state_post_attn = {tc.hidden_state_post_attn_buffer.data(),
                                         tc.hidden_state_post_attn_buffer.size()};
  tc.workspace.ffn_norm_buffer = {tc.ffn_norm_buffer.data(),
                                  tc.ffn_norm_buffer.size()};
  tc.workspace.ffn_residual_buffer = {tc.ffn_residual_buffer.data(),
                                      tc.ffn_residual_buffer.size()};
  tc.workspace.rope_cos = {tc.rope_cos.data(), tc.rope_cos.size()};
  tc.workspace.rope_sin = {tc.rope_sin.data(), tc.rope_sin.size()};
  tc.workspace.attention_workspace.plan = {seq_len, hidden, head_count,
                                           head_dim};
  tc.workspace.attention_workspace.norm_buffer = {tc.attn_norm_buffer.data(),
                                                  tc.attn_norm_buffer.size()};
  tc.workspace.attention_workspace.q_buffer = {tc.attn_q_buffer.data(),
                                               tc.attn_q_buffer.size()};
  tc.workspace.attention_workspace.k_buffer = {tc.attn_k_buffer.data(),
                                               tc.attn_k_buffer.size()};
  tc.workspace.attention_workspace.v_buffer = {tc.attn_v_buffer.data(),
                                               tc.attn_v_buffer.size()};
  tc.workspace.attention_workspace.q_head_buffer = {
      tc.attn_q_head_buffer.data(), tc.attn_q_head_buffer.size()};
  tc.workspace.attention_workspace.k_head_buffer = {
      tc.attn_k_head_buffer.data(), tc.attn_k_head_buffer.size()};
  tc.workspace.attention_workspace.v_head_buffer = {
      tc.attn_v_head_buffer.data(), tc.attn_v_head_buffer.size()};
  tc.workspace.attention_workspace.scores_buffer = {
      tc.attn_scores_buffer.data(), tc.attn_scores_buffer.size()};
  tc.workspace.attention_workspace.head_out_buffer = {
      tc.attn_head_out_buffer.data(), tc.attn_head_out_buffer.size()};
  tc.workspace.attention_workspace.concat_buffer = {
      tc.attn_concat_buffer.data(), tc.attn_concat_buffer.size()};
  tc.workspace.attention_workspace.attn_buffer = {tc.attn_attn_buffer.data(),
                                                  tc.attn_attn_buffer.size()};
  tc.workspace.ffn_workspace.intermediate_buffer =
      tc.ffn_intermediate_buffer.data();
  tc.workspace.ffn_workspace.intermediate_capacity =
      tc.ffn_intermediate_buffer.size();
}

hiko_m::Esm2ForwardRequest fill_request(Esm2Case& tc, std::size_t seq_len) {
  hiko_m::Esm2ForwardRequest request{};
  request.token_ids = tc.token_ids.data();
  request.seq_len = seq_len;
  request.descriptor = tc.descriptor;
  request.weights = &tc.weights;
  request.workspace = &tc.workspace;
  return request;
}

void run_module(Esm2Case& tc, std::size_t seq_len,
                std::vector<float>& output) {
  const std::size_t hidden = tc.descriptor.hidden_dimension;
  output.assign(seq_len * hidden, 0.0F);
  hiko_m::Esm2ForwardRequest request = fill_request(tc, seq_len);
  hiko_m::Esm2ForwardOutput out{};
  out.embeddings = output.data();
  out.seq_len = seq_len;
  out.hidden_dimension = hidden;
  const hiko_u::Status status = hiko_m::esm2_forward_scalar(request, out);
  if (!hiko_u::is_ok(status)) {
    fail("esm2_forward_scalar returned non-Ok status");
  }
}

void run_naive(Esm2Case& tc, std::size_t seq_len,
               std::vector<float>& output) {
  const std::size_t hidden = tc.descriptor.hidden_dimension;
  output.assign(seq_len * hidden, 0.0F);
  hiko_m::Esm2ForwardRequest request = fill_request(tc, seq_len);
  naive_esm2_forward(request, output.data());
}

// === Tests ================================================================

void test_single_layer_small_dimensions() {
  Esm2Case tc{};
  build_case(tc, /*vocab_size=*/8, /*hidden=*/8, /*head_count=*/2,
             /*head_dim=*/4, /*ffn_hidden=*/16, /*layer_count=*/1,
             /*seq_len=*/4, /*seed_base=*/1);
  std::vector<float> actual;
  std::vector<float> reference;
  run_module(tc, 4, actual);
  run_naive(tc, 4, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "single-layer small-dimensions fast-tolerance");
}

void test_six_layer_stack_matches_naive() {
  // Mirror the ESM2-8M layer count (6) at small per-dim sizes so the test
  // exercises the full encoder loop while staying CI-friendly.
  Esm2Case tc{};
  build_case(tc, /*vocab_size=*/12, /*hidden=*/16, /*head_count=*/4,
             /*head_dim=*/4, /*ffn_hidden=*/32, /*layer_count=*/6,
             /*seq_len=*/8, /*seed_base=*/2);
  std::vector<float> actual;
  std::vector<float> reference;
  run_module(tc, 8, actual);
  run_naive(tc, 8, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "six-layer stack fast-tolerance");
}

void test_compacted_vocab_29_lookup() {
  // Casey's checkpoint uses a compacted vocab of 29 entries; confirm the
  // module correctly handles a lookup table with non-power-of-two row count
  // and tokens that hit the last row.
  Esm2Case tc{};
  build_case(tc, /*vocab_size=*/29, /*hidden=*/8, /*head_count=*/2,
             /*head_dim=*/4, /*ffn_hidden=*/16, /*layer_count=*/1,
             /*seq_len=*/29, /*seed_base=*/3);
  std::vector<float> actual;
  std::vector<float> reference;
  run_module(tc, 29, actual);
  run_naive(tc, 29, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "compacted vocab lookup fast-tolerance");
}

void test_long_sequence_performance_sanity() {
  // 64-residue sequence with 2 layers — a sanity check that the encoder
  // runs at a length representative of small protein domains without
  // diverging from the naive reference.
  Esm2Case tc{};
  build_case(tc, /*vocab_size=*/16, /*hidden=*/16, /*head_count=*/4,
             /*head_dim=*/4, /*ffn_hidden=*/32, /*layer_count=*/2,
             /*seq_len=*/64, /*seed_base=*/4);
  std::vector<float> actual;
  std::vector<float> reference;
  run_module(tc, 64, actual);
  run_naive(tc, 64, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "64-residue sequence fast-tolerance");
}

void test_workspace_supports_shorter_seq() {
  // Build the case with capacity for seq_len=8 but only run 5 residues so
  // the encoder walks the workspace's "max_seq_len bigger than runtime"
  // path. The naive reference uses the same runtime length so the check
  // remains apples-to-apples.
  Esm2Case tc{};
  build_case(tc, /*vocab_size=*/8, /*hidden=*/8, /*head_count=*/2,
             /*head_dim=*/4, /*ffn_hidden=*/16, /*layer_count=*/2,
             /*seq_len=*/8, /*seed_base=*/5);
  std::vector<float> actual;
  std::vector<float> reference;
  run_module(tc, 5, actual);
  run_naive(tc, 5, reference);
  check_max_abs_within(actual.data(), reference.data(), actual.size(),
                       kFastTolerance,
                       "shorter-than-plan seq_len fast-tolerance");
}

// === Registry assertions =================================================

void test_module_op_registered() {
  const hiko_dr::RegisteredModuleOpRecord* record =
      hiko_dr::find_module_op("hikoboshi.esm2.v1.encoder");
  if (record == nullptr) {
    fail("module_op_registry must list hikoboshi.esm2.v1.encoder");
  }
  if (record->family != hiko_dr::ModuleOpFamily::Encoder) {
    fail("ESM2 encoder module-op must register under the Encoder family");
  }
  if (record->identity.op_version != std::string_view{"v1"}) {
    fail("ESM2 encoder identity version must be v1");
  }
  if (record->dispatch_entry == nullptr) {
    fail("ESM2 encoder must register a non-null dispatch entry");
  }
}

void test_required_primitive_ids_resolve() {
  const hiko_dr::RegisteredModuleOpRecord* record =
      hiko_dr::find_module_op("hikoboshi.esm2.v1.encoder");
  if (record == nullptr) {
    fail("ESM2 encoder must be present before required-primitive check");
  }
  const hiko_u::Span<const std::string_view> required =
      record->signature.required_primitive_op_ids;
  const std::string_view expected_ids[] = {
      "hikoboshi.layer_norm.v1",       "hikoboshi.gemm.nt.v1",
      "hikoboshi.gemm.nn.v1",          "hikoboshi.softmax.row_wise.v1",
      "hikoboshi.gelu.v1",             "hikoboshi.bias_add.v1",
      "hikoboshi.axpy.v1",
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
      std::fprintf(
          stderr,
          "esm2_forward_parity_tests: missing required primitive id %.*s\n",
          static_cast<int>(expected.size()), expected.data());
      std::exit(1);
    }
    if (hiko_dr::find_primitive_op(expected) == nullptr) {
      std::fprintf(stderr,
                   "esm2_forward_parity_tests: required primitive id %.*s does "
                   "not resolve in primitive_op_registry()\n",
                   static_cast<int>(expected.size()), expected.data());
      std::exit(1);
    }
  }
}

void test_architecture_record_has_builder() {
  const hiko_dr::RegisteredArchitectureRecord* record =
      hiko_dr::find_architecture("hikoboshi_esm2_v1");
  if (record == nullptr) {
    fail("architecture_registry must list hikoboshi_esm2_v1");
  }
  if (record->builder == nullptr) {
    fail(
        "hikoboshi_esm2_v1 architecture record must register a non-null "
        "builder once the forward-pass packet lands");
  }
}

void test_architecture_required_module_op_resolves() {
  const hiko_dr::RegisteredArchitectureRecord* record =
      hiko_dr::find_architecture("hikoboshi_esm2_v1");
  if (record == nullptr) {
    fail("architecture_registry must list hikoboshi_esm2_v1");
  }
  bool encoder_found = false;
  for (std::size_t i = 0; i < record->required_module_op_ids.size; ++i) {
    if (record->required_module_op_ids.data[i] ==
        std::string_view{"hikoboshi.esm2.v1.encoder"}) {
      encoder_found = true;
      break;
    }
  }
  if (!encoder_found) {
    fail(
        "hikoboshi_esm2_v1 architecture must declare hikoboshi.esm2.v1.encoder "
        "in its required_module_op_ids");
  }
}

void test_capabilities_strict_and_fast() {
  const hiko_dr::RegisteredModuleOpRecord* record =
      hiko_dr::find_module_op("hikoboshi.esm2.v1.encoder");
  if (record == nullptr) {
    fail("ESM2 encoder must be present before capability check");
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
    fail("ESM2 encoder must list the cpu.scalar backend");
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
    fail("ESM2 encoder must support both strict and fast parity modes");
  }
  if (record->capabilities.default_parity_mode != hiko_dr::ParityMode::Fast) {
    fail("ESM2 encoder default parity mode must be fast");
  }
}

}  // namespace

int main() {
  test_single_layer_small_dimensions();
  test_six_layer_stack_matches_naive();
  test_compacted_vocab_29_lookup();
  test_long_sequence_performance_sanity();
  test_workspace_supports_shorter_seq();
  test_module_op_registered();
  test_required_primitive_ids_resolve();
  test_architecture_record_has_builder();
  test_architecture_required_module_op_resolves();
  test_capabilities_strict_and_fast();
  return 0;
}

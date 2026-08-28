#ifndef HIKOBOSHI_MODULES_TRANSFORMER_DETAIL_ATTENTION_INLINE_HPP
#define HIKOBOSHI_MODULES_TRANSFORMER_DETAIL_ATTENTION_INLINE_HPP

/// @file
/// Header-inline body of the transformer attention compound module.
///
/// `attention_layer_scalar_inline` composes the registered scalar primitives
/// into one transformer attention block. The header-inline form follows the
/// `modules-header-inline` pattern: the public `.cpp` wrapper exists only to
/// emit a non-inline symbol for ABI consumers and the dispatch registry,
/// while every other call site (notably the future ESM2-8M forward pass)
/// includes this header directly so the optimizer can fold the
/// per-head loops into the surrounding architecture body.

#include <cmath>
#include <cstddef>
#include <cstring>

// `<hikoboshi/dispatch/scalar_forward.hpp>` reexports every primitive request /
// output type the body below uses. Modules layer is not allowed to include
// `<hikoboshi/primitives/...>` directly per `config/include_rules.json`;
// routing through dispatch matches the existing MPNN compound-module pattern
// (`cpp/include/hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp`).
#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/transformer/attention.hpp>
#include <hikoboshi/modules/transformer/detail/workspace.hpp>
#include <hikoboshi/universal/inline.hpp>

namespace hikoboshi::modules::transformer::detail {

namespace hiko_p = hikoboshi::primitives;
namespace hiko_d = hikoboshi::dispatch;

HIKOBOSHI_FORCE_INLINE void apply_linear_nt_inline(
    const float* input, const LinearLayerWeightsView& linear,
    std::size_t row_count, float* output) noexcept {
  hiko_p::linalg::GemmScalarRequest projection{};
  projection.lhs = input;
  projection.rhs = linear.weight;
  projection.m = row_count;
  projection.n = linear.output_dim;
  projection.k = linear.input_dim;
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{}, projection, output);

  if (linear.bias != nullptr) {
    hiko_p::compute::BiasAddScalarRequest bias{};
    bias.input = output;
    bias.bias = linear.bias;
    bias.row_count = row_count;
    bias.row_dimension = linear.output_dim;
    hiko_d::bias_add_forward(hiko_d::ScalarTag{}, bias, output);
  }
}

HIKOBOSHI_FORCE_INLINE void apply_pre_norm_inline(
    const float* input, const NormLayerWeightsView& norm,
    std::size_t row_count, float* output) noexcept {
  hiko_p::compute::LayerNormScalarRequest request{};
  request.input = input;
  request.gamma = norm.gamma;
  request.beta = norm.beta;
  request.row_count = row_count;
  request.row_dimension = norm.dim;
  request.epsilon = norm.epsilon;
  hiko_d::layer_norm_forward(hiko_d::ScalarTag{}, request, output);
}

/// Transpose `[seq_len, head_count, head_dim]` interleaved layout (the
/// shape `gemm_nt` produces for the Q/K/V projections) to the head-major
/// `[head_count, seq_len, head_dim]` layout each per-head matmul consumes.
HIKOBOSHI_FORCE_INLINE void transpose_to_head_major_inline(
    const float* source, std::size_t seq_len, std::size_t head_count,
    std::size_t head_dim, float* head_major) noexcept {
  for (std::size_t s = 0; s < seq_len; ++s) {
    const float* source_row = source + s * head_count * head_dim;
    for (std::size_t h = 0; h < head_count; ++h) {
      const float* source_head_slice = source_row + h * head_dim;
      float* head_major_row =
          head_major + (h * seq_len + s) * head_dim;
      for (std::size_t d = 0; d < head_dim; ++d) {
        head_major_row[d] = source_head_slice[d];
      }
    }
  }
}

/// Inverse of `transpose_to_head_major_inline`: takes per-head outputs in
/// `[head_count, seq_len, head_dim]` layout and emits `[seq_len, hidden_dim]`
/// row-major where each row is the concatenation of every head's output for
/// that token.
HIKOBOSHI_FORCE_INLINE void transpose_from_head_major_inline(
    const float* head_major, std::size_t seq_len, std::size_t head_count,
    std::size_t head_dim, float* destination) noexcept {
  for (std::size_t s = 0; s < seq_len; ++s) {
    float* destination_row = destination + s * head_count * head_dim;
    for (std::size_t h = 0; h < head_count; ++h) {
      const float* head_major_row =
          head_major + (h * seq_len + s) * head_dim;
      float* destination_head_slice = destination_row + h * head_dim;
      for (std::size_t d = 0; d < head_dim; ++d) {
        destination_head_slice[d] = head_major_row[d];
      }
    }
  }
}

/// Apply rotary position embedding in place to a head-major Q or K buffer.
///
/// Uses the canonical pairwise rotation: for each token `s` and head `h`,
/// pair dimension `d` in `[0, head_dim/2)` with dimension `d + head_dim/2`
/// and rotate by `(cos[s, d], sin[s, d])`:
///
///   q'[d]            = q[d] * cos[s, d] - q[d + head_dim/2] * sin[s, d]
///   q'[d + head_dim/2] = q[d] * sin[s, d] + q[d + head_dim/2] * cos[s, d]
///
/// `head_major` is `[head_count, seq_len, head_dim]`. `rope_cos` / `rope_sin`
/// are `[seq_len, head_dim/2]`. The same rotation table is applied across
/// every head, matching the LLaMA / RoFormer convention.
HIKOBOSHI_FORCE_INLINE void apply_rope_inplace_inline(
    float* head_major, std::size_t seq_len, std::size_t head_count,
    std::size_t head_dim, const float* rope_cos,
    const float* rope_sin) noexcept {
  const std::size_t half = head_dim / 2;
  if (half == 0) {
    return;
  }
  for (std::size_t h = 0; h < head_count; ++h) {
    for (std::size_t s = 0; s < seq_len; ++s) {
      float* row = head_major + (h * seq_len + s) * head_dim;
      const float* cos_row = rope_cos + s * half;
      const float* sin_row = rope_sin + s * half;
      for (std::size_t d = 0; d < half; ++d) {
        const float lo = row[d];
        const float hi = row[d + half];
        const float c = cos_row[d];
        const float sv = sin_row[d];
        row[d] = lo * c - hi * sv;
        row[d + half] = lo * sv + hi * c;
      }
    }
  }
}

/// Compute one head's attention output:
///   scores = Q_h @ K_h^T            (shape `[seq_len, seq_len]`)
///   attn_h = softmax(scores / sqrt(head_dim) + mask) @ V_h
///
/// The scaling is folded into the softmax temperature so the score matrix
/// does not need a separate elementwise pass.
HIKOBOSHI_FORCE_INLINE void apply_one_head_attention_inline(
    const float* q_head, const float* k_head, const float* v_head,
    std::size_t seq_len, std::size_t head_dim, const float* attention_mask,
    float* scores_scratch, float* attn_out) noexcept {
  hiko_p::linalg::GemmScalarRequest scores_request{};
  scores_request.lhs = q_head;
  scores_request.rhs = k_head;
  scores_request.m = seq_len;
  scores_request.n = seq_len;
  scores_request.k = head_dim;
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{}, scores_request, scores_scratch);

  hiko_p::compute::SoftmaxScalarRequest softmax_request{};
  softmax_request.input = scores_scratch;
  softmax_request.row_count = seq_len;
  softmax_request.row_dimension = seq_len;
  softmax_request.temperature =
      std::sqrt(static_cast<float>(head_dim));
  softmax_request.mask = attention_mask;
  hiko_p::compute::SoftmaxScalarOutput softmax_output{};
  softmax_output.output = scores_scratch;
  hiko_d::softmax_forward(hiko_d::ScalarTag{}, softmax_request, softmax_output);

  hiko_p::linalg::GemmScalarRequest attn_request{};
  attn_request.lhs = scores_scratch;
  attn_request.rhs = v_head;
  attn_request.m = seq_len;
  attn_request.n = head_dim;
  attn_request.k = seq_len;
  hiko_d::gemm_nn_forward(hiko_d::ScalarTag{}, attn_request, attn_out);
}

HIKOBOSHI_FORCE_INLINE void apply_residual_add_inline(
    const float* input, const float* attn, std::size_t count,
    float* output) noexcept {
  hiko_p::compute::AxpyScalarRequest request{};
  request.x = attn;
  request.y = input;
  request.alpha = 1.0F;
  request.count = count;
  hiko_d::axpy_forward(hiko_d::ScalarTag{}, request, output);
}

HIKOBOSHI_FORCE_INLINE void attention_layer_scalar_inline(
    const AttentionLayerRequest& request,
    const AttentionLayerOutput& output) noexcept {
  TransformerAttentionWorkspace& workspace = *request.workspace;
  const std::size_t seq_len = request.seq_len;
  const std::size_t hidden_dim = request.hidden_dim;
  const std::size_t head_count = request.head_count;
  const std::size_t head_dim = request.head_dim;

  apply_pre_norm_inline(request.input_embeddings, request.pre_norm, seq_len,
                        workspace.norm_buffer.data);

  apply_linear_nt_inline(workspace.norm_buffer.data, request.wq, seq_len,
                         workspace.q_buffer.data);
  apply_linear_nt_inline(workspace.norm_buffer.data, request.wk, seq_len,
                         workspace.k_buffer.data);
  apply_linear_nt_inline(workspace.norm_buffer.data, request.wv, seq_len,
                         workspace.v_buffer.data);

  transpose_to_head_major_inline(workspace.q_buffer.data, seq_len, head_count,
                                 head_dim, workspace.q_head_buffer.data);
  transpose_to_head_major_inline(workspace.k_buffer.data, seq_len, head_count,
                                 head_dim, workspace.k_head_buffer.data);
  transpose_to_head_major_inline(workspace.v_buffer.data, seq_len, head_count,
                                 head_dim, workspace.v_head_buffer.data);

  if (request.rope_cos != nullptr && request.rope_sin != nullptr) {
    apply_rope_inplace_inline(workspace.q_head_buffer.data, seq_len,
                              head_count, head_dim, request.rope_cos,
                              request.rope_sin);
    apply_rope_inplace_inline(workspace.k_head_buffer.data, seq_len,
                              head_count, head_dim, request.rope_cos,
                              request.rope_sin);
  }

  for (std::size_t h = 0; h < head_count; ++h) {
    const float* q_head = workspace.q_head_buffer.data + h * seq_len * head_dim;
    const float* k_head = workspace.k_head_buffer.data + h * seq_len * head_dim;
    const float* v_head = workspace.v_head_buffer.data + h * seq_len * head_dim;
    float* attn_head =
        workspace.head_out_buffer.data + h * seq_len * head_dim;
    apply_one_head_attention_inline(q_head, k_head, v_head, seq_len, head_dim,
                                    request.attention_mask,
                                    workspace.scores_buffer.data, attn_head);
  }

  transpose_from_head_major_inline(workspace.head_out_buffer.data, seq_len,
                                   head_count, head_dim,
                                   workspace.concat_buffer.data);

  apply_linear_nt_inline(workspace.concat_buffer.data, request.wo, seq_len,
                         workspace.attn_buffer.data);

  apply_residual_add_inline(request.input_embeddings,
                            workspace.attn_buffer.data, seq_len * hidden_dim,
                            output.output_embeddings);
}

}  // namespace hikoboshi::modules::transformer::detail

#endif  // HIKOBOSHI_MODULES_TRANSFORMER_DETAIL_ATTENTION_INLINE_HPP

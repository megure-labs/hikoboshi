#ifndef HIKOBOSHI_MODULES_TRANSFORMER_DETAIL_WORKSPACE_HPP
#define HIKOBOSHI_MODULES_TRANSFORMER_DETAIL_WORKSPACE_HPP

/// @file
/// Caller-owned workspace plan and buffer table for the transformer attention
/// compound module.
///
/// Hikoboshi workspaces are never allocated by the module: the caller supplies
/// `Span<float>` ranges sized from `*_count(plan)` helpers and the module
/// borrows them for the duration of one forward pass. The pattern mirrors
/// `hikoboshi::modules::detail::Mpnn64Workspace` so future architectures and
/// the eventual algorithms-layer pairwise driver can share allocation
/// strategies.

#include <cstddef>

#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::modules::transformer::detail {

/// Maximum-shape plan recorded alongside the workspace buffers.
///
/// `max_seq_len`, `hidden_dim`, `head_count`, and `head_dim` describe the
/// shape envelope the caller is willing to handle. A single forward call
/// must satisfy `request.seq_len <= plan.max_seq_len` and the rest of the
/// equalities; the inline body asserts this implicitly by reading buffer
/// extents from the plan rather than the request.
struct TransformerAttentionMemoryPlan {
  std::size_t max_seq_len;
  std::size_t hidden_dim;
  std::size_t head_count;
  std::size_t head_dim;
};

/// Caller-owned workspace buffers for one attention forward pass.
///
/// Buffer roles:
///
/// - `norm_buffer` — `[seq_len, hidden_dim]` pre-norm output.
/// - `q_buffer`, `k_buffer`, `v_buffer` — `[seq_len, hidden_dim]` Q/K/V
///   projection outputs in the original interleaved-head layout that
///   `gemm_nt` produces.
/// - `q_head_buffer`, `k_head_buffer`, `v_head_buffer` —
///   `[head_count, seq_len, head_dim]` head-major transpositions used by
///   the per-head matmuls. Required because the scalar GEMM consumes
///   row-major contiguous tiles only; a strided view would not match its
///   `lhs`/`rhs` layout assumption.
/// - `scores_buffer` — `[seq_len, seq_len]` per-head scratch holding the
///   `Q_h @ K_h^T` score matrix and the softmax output written in place.
/// - `head_out_buffer` — `[head_count, seq_len, head_dim]` accumulated
///   per-head attention outputs prior to head-concatenation.
/// - `concat_buffer` — `[seq_len, hidden_dim]` row-major concatenation of
///   the per-head outputs (inverse of the Q/K/V transposition).
/// - `attn_buffer` — `[seq_len, hidden_dim]` output projection result
///   prior to the residual add.
///
/// Spans hold borrowed pointers; the caller keeps the underlying storage
/// alive for the duration of the forward call.
struct TransformerAttentionWorkspace {
  TransformerAttentionMemoryPlan plan;
  hikoboshi::universal::Span<float> norm_buffer;
  hikoboshi::universal::Span<float> q_buffer;
  hikoboshi::universal::Span<float> k_buffer;
  hikoboshi::universal::Span<float> v_buffer;
  hikoboshi::universal::Span<float> q_head_buffer;
  hikoboshi::universal::Span<float> k_head_buffer;
  hikoboshi::universal::Span<float> v_head_buffer;
  hikoboshi::universal::Span<float> scores_buffer;
  hikoboshi::universal::Span<float> head_out_buffer;
  hikoboshi::universal::Span<float> concat_buffer;
  hikoboshi::universal::Span<float> attn_buffer;
};

inline constexpr std::size_t transformer_attention_seq_hidden_count(
    const TransformerAttentionMemoryPlan& plan) noexcept {
  return plan.max_seq_len * plan.hidden_dim;
}

inline constexpr std::size_t transformer_attention_head_major_count(
    const TransformerAttentionMemoryPlan& plan) noexcept {
  return plan.head_count * plan.max_seq_len * plan.head_dim;
}

inline constexpr std::size_t transformer_attention_scores_count(
    const TransformerAttentionMemoryPlan& plan) noexcept {
  return plan.max_seq_len * plan.max_seq_len;
}

}  // namespace hikoboshi::modules::transformer::detail

#endif  // HIKOBOSHI_MODULES_TRANSFORMER_DETAIL_WORKSPACE_HPP

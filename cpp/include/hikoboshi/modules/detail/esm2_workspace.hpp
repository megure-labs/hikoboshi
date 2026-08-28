#ifndef HIKOBOSHI_MODULES_DETAIL_ESM2_WORKSPACE_HPP
#define HIKOBOSHI_MODULES_DETAIL_ESM2_WORKSPACE_HPP

/// @file
/// Caller-owned workspace plan and buffer table for the ESM2-8M encoder.
///
/// The ESM2 forward pass composes the registered attention compound module
/// and the FFN template family inside a per-layer loop; each sub-block
/// borrows its own caller-owned workspace, and the encoder loop owns a
/// double-buffered hidden-state pair (`hidden_state` and
/// `hidden_state_post_attn`) so the attention residual can land in one
/// buffer while the FFN reads from the other. The RoPE cos/sin tables are
/// precomputed once at workspace preparation time and reused across every
/// forward call.

#include <cstddef>

#include <hikoboshi/modules/ffn/detail/workspace.hpp>
#include <hikoboshi/modules/transformer/detail/workspace.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::modules::detail {

/// Maximum-shape plan recorded alongside the workspace buffers.
///
/// `max_seq_len`, `hidden_dim`, `head_count`, `head_dim`, and
/// `ffn_hidden_dim` describe the shape envelope the caller is willing to
/// handle. A single forward call must satisfy
/// `request.seq_len <= plan.max_seq_len` and the rest of the dimension
/// equalities. The plan is shared with the embedded attention and FFN
/// sub-workspaces so a single `prepare_*` helper produces a coherent
/// arena.
struct Esm2MemoryPlan {
  std::size_t max_seq_len;
  std::size_t hidden_dim;
  std::size_t head_count;
  std::size_t head_dim;
  std::size_t ffn_hidden_dim;
};

/// Caller-owned workspace buffers for one ESM2-8M forward pass.
///
/// Buffer roles:
///
/// - `hidden_state` — `[seq_len, hidden_dim]` ping-pong buffer that holds
///   the encoder activation read by the next sub-block. The embedding
///   lookup writes here first; each FFN residual lands back here.
/// - `hidden_state_post_attn` — `[seq_len, hidden_dim]` pong buffer that
///   the attention residual writes into and the FFN reads from. The
///   FFN body never aliases this buffer with its own output; the FFN
///   residual is computed externally so the encoder loop swaps buffers
///   explicitly.
/// - `attention_workspace` — borrowed sub-workspace for the attention
///   compound module. Its `plan` field must agree with `Esm2MemoryPlan`
///   on `max_seq_len`, `hidden_dim`, `head_count`, and `head_dim`.
/// - `ffn_workspace` — borrowed sub-workspace for the FFN template body.
///   Its `intermediate_capacity` must be at least
///   `max_seq_len * ffn_hidden_dim`.
/// - `ffn_norm_buffer` — `[seq_len, hidden_dim]` pre-FFN LayerNorm output;
///   Hikoboshi 0.1.0 instantiates the FFN template under `NoNormTag`, so
///   the encoder applies the pre-FFN norm itself and hands the normalized
///   activation to `ffn_layer_inline`.
/// - `ffn_residual_buffer` — `[seq_len, hidden_dim]` FFN pre-residual
///   output; the encoder writes here, then performs the residual add
///   into `hidden_state` so the attention output remains intact through
///   the FFN body for the residual sum.
/// - `rope_cos`, `rope_sin` — `[max_seq_len, head_dim / 2]` precomputed
///   rotary tables. Both buffers are populated once at workspace prep
///   time and reused for every forward call.
///
/// Spans hold borrowed pointers; the caller keeps the underlying storage
/// alive for the duration of the forward call.
struct Esm2Workspace {
  Esm2MemoryPlan plan;
  hikoboshi::universal::Span<float> hidden_state;
  hikoboshi::universal::Span<float> hidden_state_post_attn;
  hikoboshi::universal::Span<float> ffn_norm_buffer;
  hikoboshi::universal::Span<float> ffn_residual_buffer;
  hikoboshi::universal::Span<float> rope_cos;
  hikoboshi::universal::Span<float> rope_sin;
  hikoboshi::modules::transformer::detail::TransformerAttentionWorkspace
      attention_workspace;
  hikoboshi::modules::ffn::detail::FfnLayerWorkspace ffn_workspace;
};

inline constexpr std::size_t esm2_hidden_state_count(
    const Esm2MemoryPlan& plan) noexcept {
  return plan.max_seq_len * plan.hidden_dim;
}

inline constexpr std::size_t esm2_ffn_intermediate_count(
    const Esm2MemoryPlan& plan) noexcept {
  return plan.max_seq_len * plan.ffn_hidden_dim;
}

inline constexpr std::size_t esm2_rope_table_count(
    const Esm2MemoryPlan& plan) noexcept {
  return plan.max_seq_len * (plan.head_dim / 2U);
}

}  // namespace hikoboshi::modules::detail

#endif  // HIKOBOSHI_MODULES_DETAIL_ESM2_WORKSPACE_HPP

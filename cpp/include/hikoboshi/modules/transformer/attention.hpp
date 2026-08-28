#ifndef HIKOBOSHI_MODULES_TRANSFORMER_ATTENTION_HPP
#define HIKOBOSHI_MODULES_TRANSFORMER_ATTENTION_HPP

/// @file
/// Transformer multi-head scaled-dot-product attention compound module.
///
/// One forward pass implements the standard "Attention Is All You Need" block
/// composed of a pre-LayerNorm, Q/K/V projections, optional rotary position
/// embedding, scaled dot-product self-attention, output projection, and a
/// residual add. The implementation composes the existing registered
/// primitives (`hikoboshi.layer_norm.v1`, `hikoboshi.gemm.nt.v1`,
/// `hikoboshi.gemm.nn.v1`, `hikoboshi.softmax.row_wise.v1`, `hikoboshi.bias_add.v1`,
/// `hikoboshi.axpy.v1`) and adds no new primitive math. The compound module is
/// registered as `hikoboshi.attention.v1` in `module_op_registry()`.
///
/// The request types are architecture-agnostic: `LinearLayerWeightsView`
/// and `NormLayerWeightsView` describe weight payloads structurally rather
/// than naming them after any single architecture. ESM2-8M, BERT, ESM3, and
/// any other transformer-family architecture compose this module without a
/// parallel entry point. The view definitions live in
/// `hikoboshi::modules::common` so the FFN template family at
/// `hikoboshi/modules/ffn/ffn_layer.hpp` can consume the same types.

#include <cstddef>

#include <hikoboshi/modules/common/weights_views.hpp>
#include <hikoboshi/modules/transformer/detail/workspace.hpp>

namespace hikoboshi::modules::transformer {

using common::LinearLayerWeightsView;
using common::NormLayerWeightsView;

/// Request payload for one transformer attention block.
///
/// All tensors are row-major. `input_embeddings` is `[seq_len, hidden_dim]`,
/// shared by Q, K, and V (self-attention). `head_dim` must equal
/// `hidden_dim / head_count`; the compound module does not infer it.
///
/// `rope_cos` and `rope_sin` may both be `nullptr`, which disables RoPE
/// (useful for architectures like the Casey-fine-tuned ESM2-8M variant that
/// applies sinusoidal positional encoding once at the input rather than per
/// attention layer). When provided, both tables are `[seq_len, head_dim / 2]`
/// row-major and the canonical pairwise rotation is applied to Q and K
/// before the score matmul.
///
/// `attention_mask` is an additive `[seq_len, seq_len]` mask whose entries
/// are added to the unnormalized score row before softmax. Use
/// `hikoboshi::primitives::compute::kSoftmaxNegInf` to mask a position fully.
/// `nullptr` means no mask.
///
/// `workspace` is caller-owned and must satisfy
/// `workspace->plan.max_seq_len >= seq_len` and the rest of the plan's
/// dimensions. Sizing helpers live in
/// `hikoboshi::modules::transformer::detail`.
struct AttentionLayerRequest {
  const float* input_embeddings;
  std::size_t seq_len;
  std::size_t hidden_dim;
  std::size_t head_count;
  std::size_t head_dim;

  LinearLayerWeightsView wq;
  LinearLayerWeightsView wk;
  LinearLayerWeightsView wv;
  LinearLayerWeightsView wo;
  NormLayerWeightsView pre_norm;

  const float* rope_cos;
  const float* rope_sin;

  const float* attention_mask;

  detail::TransformerAttentionWorkspace* workspace;
};

/// Output payload for one transformer attention block.
///
/// `output_embeddings` receives the residual-added block output, shape
/// `[seq_len, hidden_dim]`. The buffer is caller-owned and may alias
/// `input_embeddings`; the implementation writes the new state at the end
/// after the residual add so aliasing is safe.
struct AttentionLayerOutput {
  float* output_embeddings;
};

/// Scalar-backend forward pass for the registered
/// `hikoboshi.attention.v1` compound module.
///
/// Composes the registered primitives listed in this op's
/// `required_primitive_op_ids`. The implementation is header-inline per the
/// `modules-header-inline` pattern; this `noexcept` wrapper exists so the
/// dispatch registry can capture a stable function-pointer address.
void attention_layer_scalar(const AttentionLayerRequest& request,
                            const AttentionLayerOutput& output) noexcept;

}  // namespace hikoboshi::modules::transformer

#endif  // HIKOBOSHI_MODULES_TRANSFORMER_ATTENTION_HPP

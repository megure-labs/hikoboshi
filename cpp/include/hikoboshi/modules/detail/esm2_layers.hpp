#ifndef HIKOBOSHI_MODULES_DETAIL_ESM2_LAYERS_HPP
#define HIKOBOSHI_MODULES_DETAIL_ESM2_LAYERS_HPP

/// @file
/// ESM2-8M architecture descriptor and weight payloads.
///
/// The descriptor and the per-layer weight bundles live in the modules
/// `detail` namespace because no public surface consumes them directly; CLI
/// and Python bindings hand opaque `PackageHandle` values to the engine, and
/// the engine forwards prepared state into the registered architecture
/// builder. The shapes mirror the stock ESM2-8M topology (6 transformer
/// blocks, hidden_dim=320, head_count=20, head_dim=16, ffn_intermediate=1280)
/// and the RoPE rotary positional family. Casey's fine-tuned checkpoint uses
/// a compacted vocab (29 rows) but is otherwise identical to the stock
/// architecture; the descriptor records the runtime vocab size so the
/// forward pass and the embedding-table consumer agree on a single value
/// without re-deriving it from the weights view.

#include <cstddef>

#include <hikoboshi/modules/common/weights_views.hpp>

namespace hikoboshi::modules::detail {

/// Stock ESM2-8M LayerNorm epsilon (FAIR ESM `RobertaLayerNorm` default).
inline constexpr float kEsm2LayerNormEpsilon = 1.0e-5F;

/// ESM2-8M descriptor.
///
/// Numeric dimensions stay runtime arguments so a future ESM3/ESM2-35M record
/// can reuse the same forward-pass template; the `vocab_size` field reflects
/// the actual embedding-table row count at runtime (29 for Casey's
/// checkpoint, 33 for stock ESM2-8M).
struct Esm2Descriptor {
  std::size_t vocab_size;
  std::size_t hidden_dimension;
  std::size_t layer_count;
  std::size_t head_count;
  std::size_t head_dim;
  std::size_t ffn_hidden_dimension;
  std::size_t max_sequence_length;
};

/// Per-block weight bundle for one ESM2 transformer layer.
///
/// `attn_pre_norm` and `ffn_pre_norm` are pre-block LayerNorm gamma/beta
/// pairs. The four attention projections (`wq`, `wk`, `wv`, `wo`) and the
/// two FFN linears (`ffn_in`, `ffn_out`) use the canonical
/// `y = x @ W^T + b` linear-view shape (`LinearLayerWeightsView::weight` is
/// row-major `[output_dim, input_dim]`). All views borrow storage owned by
/// the architecture builder.
struct Esm2LayerWeights {
  hikoboshi::modules::common::NormLayerWeightsView attn_pre_norm;
  hikoboshi::modules::common::LinearLayerWeightsView wq;
  hikoboshi::modules::common::LinearLayerWeightsView wk;
  hikoboshi::modules::common::LinearLayerWeightsView wv;
  hikoboshi::modules::common::LinearLayerWeightsView wo;
  hikoboshi::modules::common::NormLayerWeightsView ffn_pre_norm;
  hikoboshi::modules::common::LinearLayerWeightsView ffn_in;
  hikoboshi::modules::common::LinearLayerWeightsView ffn_out;
};

/// Full ESM2-8M weight payload.
///
/// `embedding_table` is row-major `[vocab_size, hidden_dimension]`; the
/// forward pass indexes it by token id. `final_norm` is the post-encoder
/// LayerNorm. `layers` points at `layer_count` contiguous
/// `Esm2LayerWeights` records; the storage is owned by the architecture
/// builder for the lifetime of the prepared state.
struct Esm2Weights {
  const float* embedding_table;
  std::size_t vocab_size;
  std::size_t hidden_dimension;
  std::size_t layer_count;
  const Esm2LayerWeights* layers;
  hikoboshi::modules::common::NormLayerWeightsView final_norm;
};

}  // namespace hikoboshi::modules::detail

#endif  // HIKOBOSHI_MODULES_DETAIL_ESM2_LAYERS_HPP

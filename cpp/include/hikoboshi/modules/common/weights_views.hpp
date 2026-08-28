#ifndef HIKOBOSHI_MODULES_COMMON_WEIGHTS_VIEWS_HPP
#define HIKOBOSHI_MODULES_COMMON_WEIGHTS_VIEWS_HPP

/// @file
/// Architecture-agnostic weight-view types shared across compound modules.
///
/// Hikoboshi compound modules describe their weight payloads structurally rather
/// than naming them after any single architecture. The view types in this
/// header are the canonical surfaces every compound module consumes: a
/// `y = x @ W^T + b` linear layer view and a LayerNorm gamma/beta pair view.
/// Both are header-only aggregate structs; the caller keeps the referenced
/// tensors alive for the duration of any compound-module call that borrows
/// them.
///
/// These types previously lived in
/// `hikoboshi::modules::transformer` (see `attention.hpp`). They are now
/// promoted to `hikoboshi::modules::common` so the FFN template family
/// (`cpp/include/hikoboshi/modules/ffn/ffn_layer.hpp`) and the transformer
/// attention module can both consume the same definitions.

#include <cstddef>

namespace hikoboshi::modules::common {

/// Generic view of a `y = x @ W^T + b` linear layer.
///
/// `weight` is row-major `[output_dim, input_dim]` (the transformer convention
/// that pairs with `gemm_nt`). `bias` may be `nullptr` when the linear layer
/// is bias-less; callers should not rely on a sentinel value. The view does
/// not own storage; the caller keeps the referenced tensors alive for the
/// duration of the compound-module call.
struct LinearLayerWeightsView {
  const float* weight;
  const float* bias;
  std::size_t output_dim;
  std::size_t input_dim;
};

/// Generic view of a LayerNorm weight pair.
///
/// `gamma` and `beta` are `[dim]` row vectors. Either pointer may be
/// `nullptr` for a parameter-free normalization; the LayerNorm primitive
/// applies that branch internally. `epsilon` is the additive variance term.
struct NormLayerWeightsView {
  const float* gamma;
  const float* beta;
  std::size_t dim;
  float epsilon;
};

}  // namespace hikoboshi::modules::common

#endif  // HIKOBOSHI_MODULES_COMMON_WEIGHTS_VIEWS_HPP

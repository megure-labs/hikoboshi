#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_MPNN_D64_WEIGHTS_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_MPNN_D64_WEIGHTS_HPP

/// @file
/// Borrowed typed weight views for the hikoboshi-mpnn-d64 registered architecture.

#include <cstddef>

#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::universal::detail {

/// Fixed hikoboshi-mpnn-d64 dimensions used by typed weight views.
inline constexpr std::size_t kMpnn64HiddenDimension = 64;
inline constexpr std::size_t kMpnn64AtomCount = 5;
inline constexpr std::size_t kMpnn64AxisCount = 3;
inline constexpr std::size_t kMpnn64AtomSourceFeatureCount = 5;
inline constexpr std::size_t kMpnn64ResidueFeatureCount =
    kMpnn64AtomCount * kMpnn64AxisCount + kMpnn64AtomSourceFeatureCount;
inline constexpr std::size_t kMpnn64EdgeFeatureCount = 416;
inline constexpr std::size_t kMpnn64MessageInputDimension =
    3 * kMpnn64HiddenDimension;
inline constexpr std::size_t kMpnn64FfnHiddenDimension =
    4 * kMpnn64HiddenDimension;
inline constexpr std::size_t kMpnn64PositionalEncodingCount = 16;
inline constexpr std::size_t kMpnn64PositionalEncodingInputDimension = 66;
inline constexpr float kMpnn64LayerNormEpsilon = 1.0e-5F;

/// Compatibility projection weights used by legacy synthetic fixtures.
struct Mpnn64InputWeights {
  const float* projection_weight = nullptr;  // [20, H]
  const float* projection_bias = nullptr;    // [H], optional
};

/// Borrowed contiguous tensor payload.
struct Mpnn64TensorWeights {
  hikoboshi::universal::Span<const float> values{};
};

/// Borrowed linear layer weights with row-major weight storage.
struct Mpnn64LinearWeights {
  hikoboshi::universal::Span<const float> weight{};
  hikoboshi::universal::Span<const float> bias{};
};

/// Borrowed layer-normalization scale and bias vectors.
struct Mpnn64NormWeights {
  hikoboshi::universal::Span<const float> weight{};
  hikoboshi::universal::Span<const float> bias{};
};

/// Edge embedding projection and normalization weights.
struct Mpnn64EdgeEmbeddingWeights {
  Mpnn64LinearWeights linear{};
  Mpnn64NormWeights norm{};
};

/// Feed-forward block weights for one MPNN layer.
struct Mpnn64FeedForwardWeights {
  Mpnn64LinearWeights W_in{};
  Mpnn64LinearWeights W_out{};
};

/// Borrowed typed weights for one hikoboshi-mpnn-d64 message-passing layer.
struct Mpnn64LayerWeights {
  const float* edge_projection_weight = nullptr;  // [R, H], optional
  const float* message_norm_gamma = nullptr;      // [H], optional
  const float* message_norm_beta = nullptr;       // [H], optional
  const float* message_weight = nullptr;          // [H, H], optional
  const float* message_bias = nullptr;            // [H], optional
  const float* ffn1_weight = nullptr;             // [H, H], optional
  const float* ffn1_bias = nullptr;               // [H], optional
  const float* ffn_norm_gamma = nullptr;          // [H], optional
  const float* ffn_norm_beta = nullptr;           // [H], optional
  const float* ffn2_weight = nullptr;             // [H, H], optional
  const float* ffn2_bias = nullptr;               // [H], optional

  Mpnn64LinearWeights W1{};   // weight [H, 3H], bias [H]
  Mpnn64LinearWeights W11{};  // weight [H, 3H], bias [H]
  Mpnn64LinearWeights W12{};  // weight [H, H], bias [H]
  Mpnn64LinearWeights W13{};  // weight [H, H], bias [H]
  Mpnn64LinearWeights W2{};   // weight [H, H], bias [H]
  Mpnn64LinearWeights W3{};   // weight [H, H], bias [H]
  Mpnn64FeedForwardWeights ffn{};
  Mpnn64NormWeights norm1{};
  Mpnn64NormWeights norm2{};
  Mpnn64NormWeights norm3{};
};

/// Borrowed typed view over all hikoboshi-mpnn-d64 runtime weights.
///
/// Real compiled weights populate the named schema fields. Compatibility
/// aliases remain available for older internal fixtures that do not carry the
/// full runtime tensor table.
struct Mpnn64Weights {
  // Compatibility aliases for synthetic fixtures that predate the real tensor
  // schema.
  Mpnn64InputWeights input{};
  const Mpnn64LayerWeights* layers = nullptr;  // [layer_count]
  const float* output_norm_gamma = nullptr;    // [H], optional
  const float* output_norm_beta = nullptr;     // [H], optional

  Mpnn64LinearWeights W_e{};
  Mpnn64EdgeEmbeddingWeights edge_embedding{};
  Mpnn64LinearWeights positional_encoding{};
  std::size_t layer_count = 0;
};

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_MPNN_D64_WEIGHTS_HPP

#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_PROTEINMPNN_V48_020_WEIGHTS_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_PROTEINMPNN_V48_020_WEIGHTS_HPP

/// @file
/// Borrowed typed weight views for vanilla ProteinMPNN v_48_020.

#include <array>
#include <cstddef>

#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::universal::detail {

inline constexpr std::size_t kProteinMpnnV48020Hidden = 128;
inline constexpr std::size_t kProteinMpnnV48020KNeighbors = 48;
inline constexpr std::size_t kProteinMpnnV48020Vocab = 21;
inline constexpr std::size_t kProteinMpnnV48020RbfCount = 16;
inline constexpr std::size_t kProteinMpnnV48020NumEncoderLayers = 3;
inline constexpr std::size_t kProteinMpnnV48020NumDecoderLayers = 3;
inline constexpr std::size_t kProteinMpnnV48020PositionalEmbeddingCount = 16;
inline constexpr std::size_t kProteinMpnnV48020PositionalInputDimension = 66;
inline constexpr std::size_t kProteinMpnnV48020EdgeFeatureDimension = 416;
inline constexpr std::size_t kProteinMpnnV48020EncoderW1InputDimension =
    3 * kProteinMpnnV48020Hidden;
inline constexpr std::size_t kProteinMpnnV48020DecoderNumInDimension =
    3 * kProteinMpnnV48020Hidden;
inline constexpr std::size_t kProteinMpnnV48020DecoderW1InputDimension =
    kProteinMpnnV48020Hidden + kProteinMpnnV48020DecoderNumInDimension;
inline constexpr std::size_t kProteinMpnnV48020FfnHidden =
    4 * kProteinMpnnV48020Hidden;
inline constexpr float kProteinMpnnV48020MessageScale = 30.0F;

struct ProteinMpnnV48020LinearWeights {
  hikoboshi::universal::Span<const float> weight{};
  hikoboshi::universal::Span<const float> bias{};

  static constexpr std::size_t tensor_count = 2;
};

struct ProteinMpnnV48020NormWeights {
  hikoboshi::universal::Span<const float> weight{};
  hikoboshi::universal::Span<const float> bias{};

  static constexpr std::size_t tensor_count = 2;
};

struct ProteinMpnnV48020EmbeddingWeights {
  hikoboshi::universal::Span<const float> weight{};

  static constexpr std::size_t tensor_count = 1;
};

struct ProteinMpnnV48020EdgeEmbeddingWeights {
  hikoboshi::universal::Span<const float> weight{};

  static constexpr std::size_t tensor_count = 1;
};

struct ProteinMpnnV48020PositionalEncodingWeights {
  ProteinMpnnV48020LinearWeights linear{};

  static constexpr std::size_t tensor_count =
      ProteinMpnnV48020LinearWeights::tensor_count;
};

struct ProteinMpnnV48020FeatureWeights {
  ProteinMpnnV48020PositionalEncodingWeights embeddings{};
  ProteinMpnnV48020EdgeEmbeddingWeights edge_embedding{};
  ProteinMpnnV48020NormWeights norm_edges{};

  static constexpr std::size_t tensor_count =
      ProteinMpnnV48020PositionalEncodingWeights::tensor_count +
      ProteinMpnnV48020EdgeEmbeddingWeights::tensor_count +
      ProteinMpnnV48020NormWeights::tensor_count;
};

struct ProteinMpnnV48020FeedForwardWeights {
  ProteinMpnnV48020LinearWeights W_in{};
  ProteinMpnnV48020LinearWeights W_out{};

  static constexpr std::size_t tensor_count =
      ProteinMpnnV48020LinearWeights::tensor_count * 2;
};

struct ProteinMpnnV48020EncoderLayerWeights {
  ProteinMpnnV48020LinearWeights W1{};
  ProteinMpnnV48020LinearWeights W2{};
  ProteinMpnnV48020LinearWeights W3{};
  ProteinMpnnV48020LinearWeights W11{};
  ProteinMpnnV48020LinearWeights W12{};
  ProteinMpnnV48020LinearWeights W13{};
  ProteinMpnnV48020NormWeights norm1{};
  ProteinMpnnV48020NormWeights norm2{};
  ProteinMpnnV48020NormWeights norm3{};
  ProteinMpnnV48020FeedForwardWeights dense{};

  static constexpr std::size_t tensor_count =
      ProteinMpnnV48020LinearWeights::tensor_count * 6 +
      ProteinMpnnV48020NormWeights::tensor_count * 3 +
      ProteinMpnnV48020FeedForwardWeights::tensor_count;
};

struct ProteinMpnnV48020DecoderLayerWeights {
  ProteinMpnnV48020LinearWeights W1{};
  ProteinMpnnV48020LinearWeights W2{};
  ProteinMpnnV48020LinearWeights W3{};
  ProteinMpnnV48020NormWeights norm1{};
  ProteinMpnnV48020NormWeights norm2{};
  ProteinMpnnV48020FeedForwardWeights dense{};

  static constexpr std::size_t tensor_count =
      ProteinMpnnV48020LinearWeights::tensor_count * 3 +
      ProteinMpnnV48020NormWeights::tensor_count * 2 +
      ProteinMpnnV48020FeedForwardWeights::tensor_count;
};

struct ProteinMpnnV48020Weights {
  static constexpr std::size_t hidden = kProteinMpnnV48020Hidden;
  static constexpr std::size_t k_neighbors = kProteinMpnnV48020KNeighbors;
  static constexpr std::size_t vocab = kProteinMpnnV48020Vocab;
  static constexpr std::size_t rbf_count = kProteinMpnnV48020RbfCount;
  static constexpr std::size_t num_encoder_layers =
      kProteinMpnnV48020NumEncoderLayers;
  static constexpr std::size_t num_decoder_layers =
      kProteinMpnnV48020NumDecoderLayers;
  static constexpr float message_scale = kProteinMpnnV48020MessageScale;

  ProteinMpnnV48020FeatureWeights features{};
  ProteinMpnnV48020LinearWeights W_e{};
  ProteinMpnnV48020EmbeddingWeights W_s{};
  std::array<ProteinMpnnV48020EncoderLayerWeights,
             kProteinMpnnV48020NumEncoderLayers>
      encoder_layers{};
  std::array<ProteinMpnnV48020DecoderLayerWeights,
             kProteinMpnnV48020NumDecoderLayers>
      decoder_layers{};
  ProteinMpnnV48020LinearWeights W_out{};

  static constexpr std::size_t tensor_count =
      ProteinMpnnV48020FeatureWeights::tensor_count +
      ProteinMpnnV48020LinearWeights::tensor_count +
      ProteinMpnnV48020EmbeddingWeights::tensor_count +
      ProteinMpnnV48020EncoderLayerWeights::tensor_count *
          kProteinMpnnV48020NumEncoderLayers +
      ProteinMpnnV48020DecoderLayerWeights::tensor_count *
          kProteinMpnnV48020NumDecoderLayers +
      ProteinMpnnV48020LinearWeights::tensor_count;
};

inline constexpr std::size_t kProteinMpnnV48020WeightViewTensorCount =
    ProteinMpnnV48020Weights::tensor_count;

static_assert(ProteinMpnnV48020EncoderLayerWeights::tensor_count == 22,
              "ProteinMPNN v_48_020 encoder layer tensor count mismatch");
static_assert(ProteinMpnnV48020DecoderLayerWeights::tensor_count == 14,
              "ProteinMPNN v_48_020 decoder layer tensor count mismatch");
static_assert(kProteinMpnnV48020WeightViewTensorCount == 118,
              "ProteinMPNN v_48_020 weight-view tensor count mismatch");

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_PROTEINMPNN_V48_020_WEIGHTS_HPP

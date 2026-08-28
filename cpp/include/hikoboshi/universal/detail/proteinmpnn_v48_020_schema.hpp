#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_PROTEINMPNN_V48_020_SCHEMA_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_PROTEINMPNN_V48_020_SCHEMA_HPP

/// @file
/// Compile-time tensor schema for vanilla ProteinMPNN v_48_020.

#include <array>
#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>

namespace hikoboshi::universal::detail {

struct ProteinMpnnV48020TensorSchemaEntry {
  std::string_view name;
  std::array<std::size_t, 2> shape;
  std::size_t rank;
  std::size_t element_count;
};

constexpr ProteinMpnnV48020TensorSchemaEntry proteinmpnn_v48_020_tensor(
    std::string_view name,
    std::size_t dim0) noexcept {
  return {name, {dim0, 0}, 1, dim0};
}

constexpr ProteinMpnnV48020TensorSchemaEntry proteinmpnn_v48_020_tensor(
    std::string_view name,
    std::size_t dim0,
    std::size_t dim1) noexcept {
  return {name, {dim0, dim1}, 2, dim0 * dim1};
}

inline constexpr std::size_t kProteinMpnnV48020TensorCount =
    kProteinMpnnV48020WeightViewTensorCount;

#define HIKOBOSHI_PROTEINMPNN_V48_020_ENCODER_LAYER(layer)                         \
  proteinmpnn_v48_020_tensor("encoder_layers." #layer ".norm1.weight",          \
                             kProteinMpnnV48020Hidden),                         \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".norm1.bias",        \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".norm2.weight",      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".norm2.bias",        \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".norm3.weight",      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".norm3.bias",        \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W1.weight",         \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020EncoderW1InputDimension),     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W1.bias",           \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W2.weight",         \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W2.bias",           \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W3.weight",         \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W3.bias",           \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W11.weight",        \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020EncoderW1InputDimension),     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W11.bias",          \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W12.weight",        \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W12.bias",          \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W13.weight",        \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".W13.bias",          \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".dense.W_in.weight", \
                                 kProteinMpnnV48020FfnHidden,                   \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".dense.W_in.bias",   \
                                 kProteinMpnnV48020FfnHidden),                  \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".dense.W_out.weight",\
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020FfnHidden),                  \
      proteinmpnn_v48_020_tensor("encoder_layers." #layer ".dense.W_out.bias",  \
                                 kProteinMpnnV48020Hidden)

#define HIKOBOSHI_PROTEINMPNN_V48_020_DECODER_LAYER(layer)                         \
  proteinmpnn_v48_020_tensor("decoder_layers." #layer ".norm1.weight",          \
                             kProteinMpnnV48020Hidden),                         \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".norm1.bias",        \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".norm2.weight",      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".norm2.bias",        \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".W1.weight",         \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020DecoderW1InputDimension),     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".W1.bias",           \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".W2.weight",         \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".W2.bias",           \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".W3.weight",         \
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".W3.bias",           \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".dense.W_in.weight", \
                                 kProteinMpnnV48020FfnHidden,                   \
                                 kProteinMpnnV48020Hidden),                     \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".dense.W_in.bias",   \
                                 kProteinMpnnV48020FfnHidden),                  \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".dense.W_out.weight",\
                                 kProteinMpnnV48020Hidden,                      \
                                 kProteinMpnnV48020FfnHidden),                  \
      proteinmpnn_v48_020_tensor("decoder_layers." #layer ".dense.W_out.bias",  \
                                 kProteinMpnnV48020Hidden)

inline constexpr ProteinMpnnV48020TensorSchemaEntry
    kProteinMpnnV48020TensorSchema[] = {
        proteinmpnn_v48_020_tensor(
            "features.embeddings.linear.weight",
            kProteinMpnnV48020PositionalEmbeddingCount,
            kProteinMpnnV48020PositionalInputDimension),
        proteinmpnn_v48_020_tensor("features.embeddings.linear.bias",
                                   kProteinMpnnV48020PositionalEmbeddingCount),
        proteinmpnn_v48_020_tensor("features.edge_embedding.weight",
                                   kProteinMpnnV48020Hidden,
                                   kProteinMpnnV48020EdgeFeatureDimension),
        proteinmpnn_v48_020_tensor("features.norm_edges.weight",
                                   kProteinMpnnV48020Hidden),
        proteinmpnn_v48_020_tensor("features.norm_edges.bias",
                                   kProteinMpnnV48020Hidden),
        proteinmpnn_v48_020_tensor("W_e.weight", kProteinMpnnV48020Hidden,
                                   kProteinMpnnV48020Hidden),
        proteinmpnn_v48_020_tensor("W_e.bias", kProteinMpnnV48020Hidden),
        proteinmpnn_v48_020_tensor("W_s.weight", kProteinMpnnV48020Vocab,
                                   kProteinMpnnV48020Hidden),
        HIKOBOSHI_PROTEINMPNN_V48_020_ENCODER_LAYER(0),
        HIKOBOSHI_PROTEINMPNN_V48_020_ENCODER_LAYER(1),
        HIKOBOSHI_PROTEINMPNN_V48_020_ENCODER_LAYER(2),
        HIKOBOSHI_PROTEINMPNN_V48_020_DECODER_LAYER(0),
        HIKOBOSHI_PROTEINMPNN_V48_020_DECODER_LAYER(1),
        HIKOBOSHI_PROTEINMPNN_V48_020_DECODER_LAYER(2),
        proteinmpnn_v48_020_tensor("W_out.weight", kProteinMpnnV48020Vocab,
                                   kProteinMpnnV48020Hidden),
        proteinmpnn_v48_020_tensor("W_out.bias", kProteinMpnnV48020Vocab),
};

inline constexpr std::size_t kProteinMpnnV48020SchemaTensorCount =
    sizeof(kProteinMpnnV48020TensorSchema) /
    sizeof(kProteinMpnnV48020TensorSchema[0]);

static_assert(kProteinMpnnV48020SchemaTensorCount ==
                  kProteinMpnnV48020TensorCount,
              "ProteinMPNN v_48_020 tensor schema count mismatch");

#undef HIKOBOSHI_PROTEINMPNN_V48_020_ENCODER_LAYER
#undef HIKOBOSHI_PROTEINMPNN_V48_020_DECODER_LAYER

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_PROTEINMPNN_V48_020_SCHEMA_HPP

#ifndef HIKOBOSHI_MODULES_MPNN_PROTEINMPNN_DECODER_HPP
#define HIKOBOSHI_MODULES_MPNN_PROTEINMPNN_DECODER_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::modules::mpnn {

struct ProteinMpnnDecoderMemoryPlan {
  std::size_t max_residue_count;
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
  std::size_t decoder_input_dimension;
  std::size_t ffn_hidden_dimension;
};

struct ProteinMpnnDecoderWorkspace {
  ProteinMpnnDecoderMemoryPlan plan;
  hikoboshi::universal::Span<float> message_input;
  hikoboshi::universal::Span<float> message_state;
  hikoboshi::universal::Span<float> projected_message_state;
  hikoboshi::universal::Span<float> residue_scratch;
  hikoboshi::universal::Span<float> ffn_hidden;
};

inline constexpr std::size_t proteinmpnn_decoder_message_input_count(
    const ProteinMpnnDecoderMemoryPlan& plan) noexcept {
  return plan.max_residue_count * plan.neighbor_count *
         (plan.hidden_dimension + plan.decoder_input_dimension);
}

inline constexpr std::size_t proteinmpnn_decoder_neighbor_hidden_count(
    const ProteinMpnnDecoderMemoryPlan& plan) noexcept {
  return plan.max_residue_count * plan.neighbor_count *
         plan.hidden_dimension;
}

inline constexpr std::size_t proteinmpnn_decoder_residue_hidden_count(
    const ProteinMpnnDecoderMemoryPlan& plan) noexcept {
  return plan.max_residue_count * plan.hidden_dimension;
}

inline constexpr std::size_t proteinmpnn_decoder_ffn_hidden_count(
    const ProteinMpnnDecoderMemoryPlan& plan) noexcept {
  return plan.max_residue_count * plan.ffn_hidden_dimension;
}

struct ProteinMpnnDecoderLayerRequest {
  const float* node_embeddings;  // row-major [L, H]
  const float* edge_context;     // row-major [L, K, decoder_input_dimension]
  const float* attention_mask;   // optional row-major [L, K]
  const float* residue_mask;     // optional [L]
  const hikoboshi::universal::detail::ProteinMpnnV48020DecoderLayerWeights*
      weights;
  ProteinMpnnDecoderWorkspace* workspace;
  std::size_t residue_count;
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
  std::size_t decoder_input_dimension;
  std::size_t ffn_hidden_dimension;
  float message_scale;
};

struct ProteinMpnnDecoderLayerOutput {
  float* updated_node_embeddings;  // row-major [L, H]
};

struct ProteinMpnnLogitsHeadRequest {
  const float* node_embeddings;  // row-major [L, H]
  const hikoboshi::universal::detail::ProteinMpnnV48020LinearWeights* weights;
  std::size_t residue_count;
  std::size_t hidden_dimension;
  std::size_t vocab_size;
};

struct ProteinMpnnLogitsHeadOutput {
  float* logits;  // row-major [L, vocab_size]
};

struct ProteinMpnnSequenceEmbeddingRequest {
  const std::int32_t* token_ids;  // [token_count]
  const hikoboshi::universal::detail::ProteinMpnnV48020EmbeddingWeights* weights;
  std::size_t token_count;
  std::size_t vocab_size;
  std::size_t hidden_dimension;
};

struct ProteinMpnnSequenceEmbeddingOutput {
  float* embeddings;  // row-major [token_count, hidden_dimension]
};

void proteinmpnn_decoder_layer_scalar(
    const ProteinMpnnDecoderLayerRequest& request,
    const ProteinMpnnDecoderLayerOutput& output) noexcept;

void proteinmpnn_logits_head_scalar(
    const ProteinMpnnLogitsHeadRequest& request,
    const ProteinMpnnLogitsHeadOutput& output) noexcept;

void proteinmpnn_sequence_embedding_scalar(
    const ProteinMpnnSequenceEmbeddingRequest& request,
    const ProteinMpnnSequenceEmbeddingOutput& output) noexcept;

}  // namespace hikoboshi::modules::mpnn

#endif  // HIKOBOSHI_MODULES_MPNN_PROTEINMPNN_DECODER_HPP

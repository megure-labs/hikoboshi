#ifndef HIKOBOSHI_MODULES_MPNN_PROTEINMPNN_ENCODER_HPP
#define HIKOBOSHI_MODULES_MPNN_PROTEINMPNN_ENCODER_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::modules::mpnn {

struct ProteinMpnnEncoderDescriptor {
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
  std::size_t layer_count;
  float message_scale;
};

struct ProteinMpnnEncoderRequest {
  const float* input_node_embeddings;  // optional row-major [L, H]; null -> zero
  const float* input_edge_embeddings;  // row-major [L, K, H]
  const std::int32_t* edge_indices;    // row-major [L, K]
  const float* mask_v;                 // optional [L]
  const float* mask_attend;            // optional row-major [L, K]
  const hikoboshi::universal::detail::ProteinMpnnV48020Weights* weights;
  hikoboshi::modules::detail::Mpnn64Workspace* workspace;
  std::size_t residue_count;
  ProteinMpnnEncoderDescriptor descriptor;
};

struct ProteinMpnnEncoderOutput {
  float* node_embeddings;       // row-major [L, H]
  float* edge_embeddings;       // row-major [L, K, H]
  std::int32_t* edge_indices;   // optional row-major [L, K]
  std::size_t residue_count;
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
};

hikoboshi::universal::Status proteinmpnn_encoder_scalar(
    const ProteinMpnnEncoderRequest& request,
    const ProteinMpnnEncoderOutput& output) noexcept;

}  // namespace hikoboshi::modules::mpnn

#endif  // HIKOBOSHI_MODULES_MPNN_PROTEINMPNN_ENCODER_HPP

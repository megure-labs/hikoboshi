#ifndef HIKOBOSHI_MODULES_MPNN_MESSAGE_LAYER_HPP
#define HIKOBOSHI_MODULES_MPNN_MESSAGE_LAYER_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>

namespace hikoboshi::modules::mpnn {

struct MpnnMessageLayerRequest {
  const float* input_embeddings;   // row-major [L, H]
  const float* edge_embeddings;    // row-major [L, K, H]
  const std::int32_t* neighbor_indices;  // row-major [L, K]
  const detail::Mpnn64LayerWeights* weights;
  detail::Mpnn64Workspace* workspace;
  std::size_t residue_count;
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
  float message_scale;
};

struct MpnnMessageLayerOutput {
  float* updated_node_embeddings;  // row-major [L, H]
};

struct MessageInputPackRequest {
  const float* input_embeddings;   // row-major [L, H]
  const float* edge_embeddings;    // row-major [L, K, H]
  const std::int32_t* neighbor_indices;  // row-major [L, K]
  detail::Mpnn64Workspace* workspace;
  std::size_t residue_count;
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
};

void message_input_pack_scalar(const MessageInputPackRequest& request) noexcept;

void mpnn_message_layer_scalar(const MpnnMessageLayerRequest& request,
                               const MpnnMessageLayerOutput& output) noexcept;

}  // namespace hikoboshi::modules::mpnn

#endif  // HIKOBOSHI_MODULES_MPNN_MESSAGE_LAYER_HPP

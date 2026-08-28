#ifndef HIKOBOSHI_MODULES_MPNN_DETAIL_MESSAGE_LAYER_INLINE_HPP
#define HIKOBOSHI_MODULES_MPNN_DETAIL_MESSAGE_LAYER_INLINE_HPP

#include <algorithm>
#include <cstddef>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp>
#include <hikoboshi/modules/mpnn/message_layer.hpp>
#include <hikoboshi/universal/inline.hpp>

namespace hikoboshi::modules::mpnn::detail {

HIKOBOSHI_FORCE_INLINE bool valid_neighbor_inline(
    const MpnnMessageLayerRequest& request,
    std::int32_t neighbor) noexcept {
  return neighbor >= 0 &&
         static_cast<std::size_t>(neighbor) < request.residue_count;
}

HIKOBOSHI_FORCE_INLINE std::size_t active_neighbor_count_inline(
    const MpnnMessageLayerRequest& request) noexcept {
  return std::min(request.neighbor_count, request.residue_count);
}

HIKOBOSHI_FORCE_INLINE std::size_t active_neighbor_count_inline(
    const MessageInputPackRequest& request) noexcept {
  return std::min(request.neighbor_count, request.residue_count);
}

HIKOBOSHI_FORCE_INLINE MessageInputPackRequest message_input_pack_request_inline(
    const MpnnMessageLayerRequest& request) noexcept {
  return {request.input_embeddings,
          request.edge_embeddings,
          request.neighbor_indices,
          request.workspace,
          request.residue_count,
          request.hidden_dimension,
          request.neighbor_count};
}

HIKOBOSHI_FORCE_INLINE void gather_neighbor_state_inline(
    const MessageInputPackRequest& request) noexcept {
  hikoboshi::primitives::compute::GatherScalarRequest gather{};
  gather.source = request.input_embeddings;
  gather.indices = request.neighbor_indices;
  gather.source_row_count = request.residue_count;
  gather.row_dimension = request.hidden_dimension;
  gather.index_count = request.residue_count * request.neighbor_count;
  hikoboshi::dispatch::gather_forward(hikoboshi::dispatch::ScalarTag{}, gather,
                                    request.workspace->gathered_state.data);
}

HIKOBOSHI_FORCE_INLINE void build_message_inputs_inline(
    const MessageInputPackRequest& request) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t input_dim =
      hikoboshi::modules::detail::kMpnn64MessageInputDimension;
  const std::size_t neighbor_count = active_neighbor_count_inline(request);

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    const float* query_state = request.input_embeddings + residue * hidden;
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot = residue * request.neighbor_count + neighbor_slot;
      float* input = workspace.residue_features.data + slot * input_dim;
      copy_values_inline(query_state, input, hidden);
      copy_values_inline(request.edge_embeddings + slot * hidden,
                         input + hidden, hidden);
      copy_values_inline(workspace.gathered_state.data + slot * hidden,
                         input + 2 * hidden, hidden);
    }
  }
}

HIKOBOSHI_FORCE_INLINE void message_input_pack_scalar_inline(
    const MessageInputPackRequest& request) noexcept {
  gather_neighbor_state_inline(request);
  build_message_inputs_inline(request);
}

HIKOBOSHI_FORCE_INLINE void apply_message_mlp_active_neighbors_inline(
    const MpnnMessageLayerRequest& request,
    std::size_t neighbor_count) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const hikoboshi::modules::detail::Mpnn64LayerWeights& layer = *request.weights;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t input_dim =
      hikoboshi::modules::detail::kMpnn64MessageInputDimension;

  message_input_pack_scalar_inline(message_input_pack_request_inline(request));

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot = residue * request.neighbor_count + neighbor_slot;
      const float* input = workspace.residue_features.data + slot * input_dim;
      float* message = workspace.message_state.data + slot * hidden;
      float* projected = workspace.projected_message_state.data + slot * hidden;

      linear_row_nt_inline(input, layer.W1, hidden, input_dim, message);
      gelu_inplace_inline(message, hidden);
      linear_row_nt_inline(message, layer.W2, hidden, hidden, projected);
      gelu_inplace_inline(projected, hidden);
      linear_row_nt_inline(projected, layer.W3, hidden, hidden, message);
    }
  }
}

HIKOBOSHI_FORCE_INLINE void apply_message_mlp_inline(
    const MpnnMessageLayerRequest& request) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const hikoboshi::modules::detail::Mpnn64LayerWeights& layer = *request.weights;
  const std::size_t slot_count =
      request.residue_count * request.neighbor_count;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t input_dim =
      hikoboshi::modules::detail::kMpnn64MessageInputDimension;
  const std::size_t neighbor_count = active_neighbor_count_inline(request);

  if (neighbor_count < request.neighbor_count) {
    apply_message_mlp_active_neighbors_inline(request, neighbor_count);
    return;
  }

  message_input_pack_scalar_inline(message_input_pack_request_inline(request));

  const std::size_t hidden_count = slot_count * hidden;
  linear_nt_inline(workspace.residue_features.data, layer.W1, slot_count,
                   hidden, input_dim, workspace.message_state.data);
  gelu_inplace_inline(workspace.message_state.data, hidden_count);
  linear_nt_inline(workspace.message_state.data, layer.W2, slot_count, hidden,
                   hidden, workspace.projected_message_state.data);
  gelu_inplace_inline(workspace.projected_message_state.data, hidden_count);
  linear_nt_inline(workspace.projected_message_state.data, layer.W3, slot_count,
                   hidden, hidden, workspace.message_state.data);
}

HIKOBOSHI_FORCE_INLINE void aggregate_messages_and_update_nodes_inline(
    const MpnnMessageLayerRequest& request,
    const MpnnMessageLayerOutput& output) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const hikoboshi::modules::detail::Mpnn64LayerWeights& layer = *request.weights;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t node_count = request.residue_count * hidden;
  const std::size_t neighbor_count = active_neighbor_count_inline(request);
  fill_zero_inline(workspace.residue_scratch.data, node_count);

  const float scale = 1.0F / request.message_scale;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    float* update = workspace.residue_scratch.data + residue * hidden;
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot = residue * request.neighbor_count + neighbor_slot;
      if (!valid_neighbor_inline(request, request.neighbor_indices[slot])) {
        continue;
      }
      const float* message = workspace.message_state.data + slot * hidden;
      axpy_into_inline(update, message, scale, hidden);
    }
  }

  layer_norm_residual_rows_inline(workspace.residue_scratch.data,
                                  request.input_embeddings, layer.norm1,
                                  request.residue_count, hidden,
                                  output.updated_node_embeddings);
}

HIKOBOSHI_FORCE_INLINE void mpnn_message_layer_scalar_inline(
    const MpnnMessageLayerRequest& request,
    const MpnnMessageLayerOutput& output) noexcept {
  apply_message_mlp_inline(request);
  aggregate_messages_and_update_nodes_inline(request, output);
}

}  // namespace hikoboshi::modules::mpnn::detail

#endif  // HIKOBOSHI_MODULES_MPNN_DETAIL_MESSAGE_LAYER_INLINE_HPP

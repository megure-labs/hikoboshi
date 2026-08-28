#include <hikoboshi/modules/mpnn/proteinmpnn_encoder.hpp>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hikoboshi::modules::mpnn {
namespace {

namespace hiko_u = hikoboshi::universal;
namespace pmp = hikoboshi::universal::detail;
namespace mdi = hikoboshi::modules::mpnn::detail;

constexpr hiko_u::Status kOk{hiko_u::StatusCode::Ok, ""};

hiko_u::Status invalid(const char* detail) noexcept {
  return hiko_u::invalid_argument_status(detail);
}

hiko_u::Status failed_precondition(const char* detail) noexcept {
  return hiko_u::failed_precondition_status(detail);
}

bool has_capacity(hiko_u::Span<float> span, std::size_t required) noexcept {
  return span.data != nullptr && span.size >= required;
}

bool has_capacity(hiko_u::Span<std::int32_t> span,
                  std::size_t required) noexcept {
  return span.data != nullptr && span.size >= required;
}

bool has_exact_size(hiko_u::Span<const float> span,
                    std::size_t required) noexcept {
  return span.data != nullptr && span.size == required;
}

bool has_linear_size(const pmp::ProteinMpnnV48020LinearWeights& weights,
                     std::size_t output_dimension,
                     std::size_t input_dimension) noexcept {
  return has_exact_size(weights.weight, output_dimension * input_dimension) &&
         has_exact_size(weights.bias, output_dimension);
}

bool has_norm_size(const pmp::ProteinMpnnV48020NormWeights& weights,
                   std::size_t dimension) noexcept {
  return has_exact_size(weights.weight, dimension) &&
         has_exact_size(weights.bias, dimension);
}

hikoboshi::modules::detail::Mpnn64LinearWeights adapt_linear(
    const pmp::ProteinMpnnV48020LinearWeights& weights) noexcept {
  return {weights.weight, weights.bias};
}

hikoboshi::modules::detail::Mpnn64NormWeights adapt_norm(
    const pmp::ProteinMpnnV48020NormWeights& weights) noexcept {
  return {weights.weight, weights.bias};
}

std::size_t slot_count(const ProteinMpnnEncoderRequest& request) noexcept {
  return request.residue_count * request.descriptor.neighbor_count;
}

bool valid_neighbor(const ProteinMpnnEncoderRequest& request,
                    std::int32_t neighbor) noexcept {
  return neighbor >= 0 &&
         static_cast<std::size_t>(neighbor) < request.residue_count;
}

bool workspace_matches(const ProteinMpnnEncoderRequest& request) noexcept {
  const hikoboshi::modules::detail::Mpnn64Workspace& workspace =
      *request.workspace;
  const hikoboshi::modules::detail::Mpnn64MemoryPlan& plan = workspace.plan;
  const std::size_t residues = request.residue_count;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t neighbors = request.descriptor.neighbor_count;
  const std::size_t slots = slot_count(request);
  const std::size_t ffn_hidden = 4 * hidden;

  if (plan.max_residue_count < residues || plan.hidden_dimension != hidden ||
      plan.neighbor_count != neighbors ||
      plan.layer_count < request.descriptor.layer_count) {
    return false;
  }
  return has_capacity(workspace.residue_features, slots * 3 * hidden) &&
         has_capacity(workspace.neighbor_indices, slots) &&
         has_capacity(workspace.residue_state, residues * hidden) &&
         has_capacity(workspace.gathered_state, slots * hidden) &&
         has_capacity(workspace.edge_state, slots * hidden) &&
         has_capacity(workspace.message_state, slots * hidden) &&
         has_capacity(workspace.projected_message_state, slots * hidden) &&
         has_capacity(workspace.residue_scratch, residues * hidden) &&
         has_capacity(workspace.ffn_hidden, residues * ffn_hidden);
}

hiko_u::Status validate_weights(
    const ProteinMpnnEncoderRequest& request) noexcept {
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t message_input = 3 * hidden;
  const std::size_t ffn_hidden = 4 * hidden;
  if (request.descriptor.layer_count >
      pmp::ProteinMpnnV48020Weights::num_encoder_layers) {
    return invalid("ProteinMPNN encoder layer_count exceeds v48 weight view");
  }
  for (std::size_t layer = 0; layer < request.descriptor.layer_count; ++layer) {
    const pmp::ProteinMpnnV48020EncoderLayerWeights& weights =
        request.weights->encoder_layers[layer];
    if (!has_linear_size(weights.W1, hidden, message_input) ||
        !has_linear_size(weights.W11, hidden, message_input)) {
      return invalid(
          "ProteinMPNN encoder W1/W11 weights must be [H,3H] and [H]");
    }
    if (!has_linear_size(weights.W2, hidden, hidden) ||
        !has_linear_size(weights.W3, hidden, hidden) ||
        !has_linear_size(weights.W12, hidden, hidden) ||
        !has_linear_size(weights.W13, hidden, hidden)) {
      return invalid(
          "ProteinMPNN encoder W2/W3/W12/W13 weights must be [H,H] and [H]");
    }
    if (!has_linear_size(weights.dense.W_in, ffn_hidden, hidden) ||
        !has_linear_size(weights.dense.W_out, hidden, ffn_hidden)) {
      return invalid(
          "ProteinMPNN encoder dense weights must be [4H,H] and [H,4H]");
    }
    if (!has_norm_size(weights.norm1, hidden) ||
        !has_norm_size(weights.norm2, hidden) ||
        !has_norm_size(weights.norm3, hidden)) {
      return invalid("ProteinMPNN encoder norm weights must be [H]");
    }
  }
  return kOk;
}

void copy_initial_state(const ProteinMpnnEncoderRequest& request) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t node_count = request.residue_count * hidden;
  const std::size_t edge_hidden_count = slot_count(request) * hidden;
  const std::size_t edge_index_count = slot_count(request);

  if (request.input_node_embeddings != nullptr) {
    mdi::copy_values_inline(request.input_node_embeddings,
                            workspace.residue_state.data, node_count);
  } else {
    mdi::fill_zero_inline(workspace.residue_state.data, node_count);
  }
  mdi::copy_values_inline(request.input_edge_embeddings, workspace.edge_state.data,
                          edge_hidden_count);
  std::memcpy(workspace.neighbor_indices.data, request.edge_indices,
              edge_index_count * sizeof(std::int32_t));
}

void gather_neighbor_state(const ProteinMpnnEncoderRequest& request) noexcept {
  hikoboshi::primitives::compute::GatherScalarRequest gather{};
  gather.source = request.workspace->residue_state.data;
  gather.indices = request.workspace->neighbor_indices.data;
  gather.source_row_count = request.residue_count;
  gather.row_dimension = request.descriptor.hidden_dimension;
  gather.index_count = slot_count(request);
  hikoboshi::dispatch::gather_forward(hikoboshi::dispatch::ScalarTag{}, gather,
                                    request.workspace->gathered_state.data);
}

void build_message_inputs(const ProteinMpnnEncoderRequest& request) noexcept {
  gather_neighbor_state(request);
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t neighbors = request.descriptor.neighbor_count;
  const std::size_t input_dim = 3 * hidden;

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    const float* query = workspace.residue_state.data + residue * hidden;
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbors;
         ++neighbor_slot) {
      const std::size_t slot = residue * neighbors + neighbor_slot;
      float* input = workspace.residue_features.data + slot * input_dim;
      mdi::copy_values_inline(query, input, hidden);
      mdi::copy_values_inline(workspace.edge_state.data + slot * hidden,
                              input + hidden, hidden);
      mdi::copy_values_inline(workspace.gathered_state.data + slot * hidden,
                              input + 2 * hidden, hidden);
    }
  }
}

void apply_message_mlp(
    const ProteinMpnnEncoderRequest& request,
    const pmp::ProteinMpnnV48020EncoderLayerWeights& weights) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t slots = slot_count(request);
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t input_dim = 3 * hidden;
  const std::size_t hidden_count = slots * hidden;

  build_message_inputs(request);
  mdi::linear_nt_inline(workspace.residue_features.data,
                        adapt_linear(weights.W1), slots, hidden, input_dim,
                        workspace.message_state.data);
  mdi::gelu_inplace_inline(workspace.message_state.data, hidden_count);
  mdi::linear_nt_inline(workspace.message_state.data, adapt_linear(weights.W2),
                        slots, hidden, hidden,
                        workspace.projected_message_state.data);
  mdi::gelu_inplace_inline(workspace.projected_message_state.data,
                           hidden_count);
  mdi::linear_nt_inline(workspace.projected_message_state.data,
                        adapt_linear(weights.W3), slots, hidden, hidden,
                        workspace.message_state.data);
}

void aggregate_messages_and_update_nodes(
    const ProteinMpnnEncoderRequest& request,
    const pmp::ProteinMpnnV48020EncoderLayerWeights& weights) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t neighbors = request.descriptor.neighbor_count;
  const std::size_t node_count = request.residue_count * hidden;
  mdi::fill_zero_inline(workspace.residue_scratch.data, node_count);

  const float scale = 1.0F / request.descriptor.message_scale;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    float* update = workspace.residue_scratch.data + residue * hidden;
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbors;
         ++neighbor_slot) {
      const std::size_t slot = residue * neighbors + neighbor_slot;
      if (!valid_neighbor(request, workspace.neighbor_indices.data[slot])) {
        continue;
      }
      float alpha = scale;
      if (request.mask_attend != nullptr) {
        alpha *= request.mask_attend[slot];
      }
      if (alpha == 0.0F) {
        continue;
      }
      mdi::axpy_into_inline(update, workspace.message_state.data + slot * hidden,
                            alpha, hidden);
    }
  }

  mdi::layer_norm_residual_rows_inline(
      workspace.residue_scratch.data, workspace.residue_state.data,
      adapt_norm(weights.norm1), request.residue_count, hidden,
      workspace.residue_state.data);
}

void apply_ffn(
    const ProteinMpnnEncoderRequest& request,
    const pmp::ProteinMpnnV48020EncoderLayerWeights& weights) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t ffn_hidden = 4 * hidden;

  mdi::linear_nt_inline(workspace.residue_state.data,
                        adapt_linear(weights.dense.W_in),
                        request.residue_count, ffn_hidden, hidden,
                        workspace.ffn_hidden.data);
  mdi::gelu_inplace_inline(workspace.ffn_hidden.data,
                           request.residue_count * ffn_hidden);
  mdi::linear_nt_inline(workspace.ffn_hidden.data,
                        adapt_linear(weights.dense.W_out),
                        request.residue_count, hidden, ffn_hidden,
                        workspace.residue_scratch.data);
  mdi::layer_norm_residual_rows_inline(
      workspace.residue_scratch.data, workspace.residue_state.data,
      adapt_norm(weights.norm2), request.residue_count, hidden,
      workspace.residue_state.data);
}

void apply_mask_v(const ProteinMpnnEncoderRequest& request) noexcept {
  if (request.mask_v == nullptr) {
    return;
  }
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    float* row = workspace.residue_state.data + residue * hidden;
    const float mask = request.mask_v[residue];
    for (std::size_t dim = 0; dim < hidden; ++dim) {
      row[dim] *= mask;
    }
  }
}

void apply_edge_update(
    const ProteinMpnnEncoderRequest& request,
    const pmp::ProteinMpnnV48020EncoderLayerWeights& weights) noexcept {
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t slots = slot_count(request);
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t input_dim = 3 * hidden;
  const std::size_t hidden_count = slots * hidden;

  build_message_inputs(request);
  mdi::linear_nt_inline(workspace.residue_features.data,
                        adapt_linear(weights.W11), slots, hidden, input_dim,
                        workspace.message_state.data);
  mdi::gelu_inplace_inline(workspace.message_state.data, hidden_count);
  mdi::linear_nt_inline(workspace.message_state.data, adapt_linear(weights.W12),
                        slots, hidden, hidden,
                        workspace.projected_message_state.data);
  mdi::gelu_inplace_inline(workspace.projected_message_state.data,
                           hidden_count);
  mdi::linear_nt_inline(workspace.projected_message_state.data,
                        adapt_linear(weights.W13), slots, hidden, hidden,
                        workspace.message_state.data);

  for (std::size_t slot = 0; slot < slots; ++slot) {
    if (!valid_neighbor(request, workspace.neighbor_indices.data[slot])) {
      continue;
    }
    mdi::layer_norm_residual_row_inline(
        workspace.message_state.data + slot * hidden,
        workspace.edge_state.data + slot * hidden, adapt_norm(weights.norm3),
        hidden, workspace.edge_state.data + slot * hidden);
  }
}

void apply_encoder_layers(const ProteinMpnnEncoderRequest& request) noexcept {
  for (std::size_t layer = 0; layer < request.descriptor.layer_count; ++layer) {
    const pmp::ProteinMpnnV48020EncoderLayerWeights& weights =
        request.weights->encoder_layers[layer];
    apply_message_mlp(request, weights);
    aggregate_messages_and_update_nodes(request, weights);
    apply_ffn(request, weights);
    apply_mask_v(request);
    apply_edge_update(request, weights);
  }
}

void write_output(const ProteinMpnnEncoderRequest& request,
                  const ProteinMpnnEncoderOutput& output) noexcept {
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t edge_hidden_count = slot_count(request) * hidden;
  const std::size_t edge_index_count = slot_count(request);
  mdi::copy_values_inline(request.workspace->residue_state.data,
                          output.node_embeddings,
                          request.residue_count * hidden);
  mdi::copy_values_inline(request.workspace->edge_state.data,
                          output.edge_embeddings, edge_hidden_count);
  if (output.edge_indices != nullptr) {
    std::memcpy(output.edge_indices, request.workspace->neighbor_indices.data,
                edge_index_count * sizeof(std::int32_t));
  }
}

}  // namespace

hiko_u::Status proteinmpnn_encoder_scalar(
    const ProteinMpnnEncoderRequest& request,
    const ProteinMpnnEncoderOutput& output) noexcept {
  if (request.input_edge_embeddings == nullptr) {
    return invalid("ProteinMPNN encoder input_edge_embeddings pointer is null");
  }
  if (request.edge_indices == nullptr) {
    return invalid("ProteinMPNN encoder edge_indices pointer is null");
  }
  if (request.weights == nullptr) {
    return invalid("ProteinMPNN encoder weights pointer is null");
  }
  if (request.workspace == nullptr) {
    return invalid("ProteinMPNN encoder workspace pointer is null");
  }
  if (output.node_embeddings == nullptr || output.edge_embeddings == nullptr) {
    return invalid("ProteinMPNN encoder output pointers must be non-null");
  }
  if (request.descriptor.hidden_dimension == 0 ||
      request.descriptor.neighbor_count == 0) {
    return invalid("ProteinMPNN encoder hidden_dimension and neighbor_count must be non-zero");
  }
  if (request.descriptor.message_scale <= 0.0F ||
      !std::isfinite(request.descriptor.message_scale)) {
    return invalid("ProteinMPNN encoder message_scale must be finite and positive");
  }
  if (output.residue_count < request.residue_count ||
      output.hidden_dimension != request.descriptor.hidden_dimension ||
      output.neighbor_count != request.descriptor.neighbor_count) {
    return invalid("ProteinMPNN encoder output shape disagrees with request");
  }
  const hiko_u::Status weight_status = validate_weights(request);
  if (!weight_status.ok()) {
    return weight_status;
  }
  if (!workspace_matches(request)) {
    return failed_precondition(
        "ProteinMPNN encoder workspace does not satisfy memory plan");
  }
  if (request.residue_count == 0) {
    return kOk;
  }

  copy_initial_state(request);
  apply_encoder_layers(request);
  write_output(request, output);
  return kOk;
}

}  // namespace hikoboshi::modules::mpnn

#include <hikoboshi/modules/mpnn/proteinmpnn_decoder.hpp>

#include <cstddef>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp>

namespace hikoboshi::modules::mpnn {
namespace {

namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_i = hikoboshi::modules::mpnn::detail;
namespace hiko_p = hikoboshi::primitives;
namespace hiko_u = hikoboshi::universal::detail;

hiko_d::Mpnn64LinearWeights adapt_linear(
    const hiko_u::ProteinMpnnV48020LinearWeights& weights) noexcept {
  return {weights.weight, weights.bias};
}

hiko_d::Mpnn64NormWeights adapt_norm(
    const hiko_u::ProteinMpnnV48020NormWeights& weights) noexcept {
  return {weights.weight, weights.bias};
}

void build_decoder_message_inputs(
    const ProteinMpnnDecoderLayerRequest& request) noexcept {
  ProteinMpnnDecoderWorkspace& workspace = *request.workspace;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t decoder_input = request.decoder_input_dimension;
  const std::size_t input_dim = hidden + decoder_input;

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    const float* node_row = request.node_embeddings + residue * hidden;
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < request.neighbor_count; ++neighbor_slot) {
      const std::size_t slot = residue * request.neighbor_count + neighbor_slot;
      float* input = workspace.message_input.data + slot * input_dim;
      hiko_i::copy_values_inline(node_row, input, hidden);
      hiko_i::copy_values_inline(request.edge_context + slot * decoder_input,
                              input + hidden, decoder_input);
    }
  }
}

void apply_decoder_message_mlp(
    const ProteinMpnnDecoderLayerRequest& request) noexcept {
  ProteinMpnnDecoderWorkspace& workspace = *request.workspace;
  const hiko_u::ProteinMpnnV48020DecoderLayerWeights& layer = *request.weights;
  const std::size_t slot_count =
      request.residue_count * request.neighbor_count;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t input_dim = hidden + request.decoder_input_dimension;
  const std::size_t hidden_count = slot_count * hidden;

  build_decoder_message_inputs(request);
  hiko_i::linear_nt_inline(workspace.message_input.data, adapt_linear(layer.W1),
                        slot_count, hidden, input_dim,
                        workspace.message_state.data);
  hiko_i::gelu_inplace_inline(workspace.message_state.data, hidden_count);
  hiko_i::linear_nt_inline(workspace.message_state.data, adapt_linear(layer.W2),
                        slot_count, hidden, hidden,
                        workspace.projected_message_state.data);
  hiko_i::gelu_inplace_inline(workspace.projected_message_state.data,
                           hidden_count);
  hiko_i::linear_nt_inline(workspace.projected_message_state.data,
                        adapt_linear(layer.W3), slot_count, hidden, hidden,
                        workspace.message_state.data);
}

void aggregate_decoder_messages(
    const ProteinMpnnDecoderLayerRequest& request,
    const ProteinMpnnDecoderLayerOutput& output) noexcept {
  ProteinMpnnDecoderWorkspace& workspace = *request.workspace;
  const hiko_u::ProteinMpnnV48020DecoderLayerWeights& layer = *request.weights;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t node_count = request.residue_count * hidden;
  hiko_i::fill_zero_inline(workspace.residue_scratch.data, node_count);

  const float scale = 1.0F / request.message_scale;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    float* update = workspace.residue_scratch.data + residue * hidden;
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < request.neighbor_count; ++neighbor_slot) {
      const std::size_t slot = residue * request.neighbor_count + neighbor_slot;
      const float mask =
          request.attention_mask == nullptr ? 1.0F : request.attention_mask[slot];
      if (mask == 0.0F) {
        continue;
      }
      const float* message = workspace.message_state.data + slot * hidden;
      hiko_i::axpy_into_inline(update, message, mask * scale, hidden);
    }
  }

  hiko_i::layer_norm_residual_rows_inline(
      workspace.residue_scratch.data, request.node_embeddings,
      adapt_norm(layer.norm1), request.residue_count, hidden,
      output.updated_node_embeddings);
}

void apply_decoder_ffn(const ProteinMpnnDecoderLayerRequest& request,
                       const ProteinMpnnDecoderLayerOutput& output) noexcept {
  ProteinMpnnDecoderWorkspace& workspace = *request.workspace;
  const hiko_u::ProteinMpnnV48020DecoderLayerWeights& layer = *request.weights;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t ffn_hidden = request.ffn_hidden_dimension;

  hiko_i::linear_nt_inline(output.updated_node_embeddings,
                        adapt_linear(layer.dense.W_in), request.residue_count,
                        ffn_hidden, hidden, workspace.ffn_hidden.data);
  hiko_i::gelu_inplace_inline(workspace.ffn_hidden.data,
                           request.residue_count * ffn_hidden);
  hiko_i::linear_nt_inline(workspace.ffn_hidden.data,
                        adapt_linear(layer.dense.W_out), request.residue_count,
                        hidden, ffn_hidden, workspace.residue_scratch.data);
  hiko_i::layer_norm_residual_rows_inline(
      workspace.residue_scratch.data, output.updated_node_embeddings,
      adapt_norm(layer.norm2), request.residue_count, hidden,
      output.updated_node_embeddings);
}

void apply_residue_mask_gate(
    const ProteinMpnnDecoderLayerRequest& request,
    const ProteinMpnnDecoderLayerOutput& output) noexcept {
  if (request.residue_mask == nullptr) {
    return;
  }
  const std::size_t hidden = request.hidden_dimension;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    float* row = output.updated_node_embeddings + residue * hidden;
    const float mask = request.residue_mask[residue];
    if (mask == 1.0F) {
      continue;
    }
    for (std::size_t d = 0; d < hidden; ++d) {
      row[d] *= mask;
    }
  }
}

}  // namespace

void proteinmpnn_decoder_layer_scalar(
    const ProteinMpnnDecoderLayerRequest& request,
    const ProteinMpnnDecoderLayerOutput& output) noexcept {
  apply_decoder_message_mlp(request);
  aggregate_decoder_messages(request, output);
  apply_decoder_ffn(request, output);
  apply_residue_mask_gate(request, output);
}

void proteinmpnn_logits_head_scalar(
    const ProteinMpnnLogitsHeadRequest& request,
    const ProteinMpnnLogitsHeadOutput& output) noexcept {
  hiko_p::linalg::GemmScalarRequest projection{};
  projection.lhs = request.node_embeddings;
  projection.rhs = request.weights->weight.data;
  projection.m = request.residue_count;
  projection.n = request.vocab_size;
  projection.k = request.hidden_dimension;
  hikoboshi::dispatch::gemm_nt_forward(hikoboshi::dispatch::ScalarTag{},
                                     projection, output.logits);

  if (request.weights->bias.data != nullptr &&
      request.weights->bias.size != 0U) {
    hiko_p::compute::BiasAddScalarRequest bias{};
    bias.input = output.logits;
    bias.bias = request.weights->bias.data;
    bias.row_count = request.residue_count;
    bias.row_dimension = request.vocab_size;
    hikoboshi::dispatch::bias_add_forward(hikoboshi::dispatch::ScalarTag{}, bias,
                                        output.logits);
  }
}

void proteinmpnn_sequence_embedding_scalar(
    const ProteinMpnnSequenceEmbeddingRequest& request,
    const ProteinMpnnSequenceEmbeddingOutput& output) noexcept {
  hiko_p::compute::GatherScalarRequest gather{};
  gather.source = request.weights->weight.data;
  gather.indices = request.token_ids;
  gather.source_row_count = request.vocab_size;
  gather.row_dimension = request.hidden_dimension;
  gather.index_count = request.token_count;
  hikoboshi::dispatch::gather_forward(hikoboshi::dispatch::ScalarTag{}, gather,
                                    output.embeddings);
}

}  // namespace hikoboshi::modules::mpnn

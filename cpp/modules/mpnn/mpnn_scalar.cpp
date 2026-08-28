#include <hikoboshi/modules/mpnn.hpp>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/mpnn/detail/ffn_layer_inline.hpp>
#include <hikoboshi/modules/mpnn/detail/message_layer_inline.hpp>
#include <hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp>
#include <hikoboshi/modules/mpnn/edge_rbf_features.hpp>
#include <hikoboshi/modules/mpnn/ffn_layer.hpp>
#include <hikoboshi/modules/mpnn/message_layer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace hikoboshi::modules::detail {
namespace {

namespace mpnn_inline = hikoboshi::modules::mpnn::detail;

std::size_t coord_offset(std::size_t residue,
                         std::size_t atom,
                         std::size_t axis) noexcept {
  return (residue * kMpnn64AtomCount + atom) * kMpnn64AxisCount + axis;
}

void capture_debug_tensor(const Mpnn64ForwardRequest& request,
                          float* target,
                          const float* source,
                          std::size_t count) noexcept {
  if (request.debug_capture == nullptr || target == nullptr) {
    return;
  }
  std::memcpy(target, source, count * sizeof(float));
}

void capture_edge_embedding_output(
    const Mpnn64ForwardRequest& request) noexcept {
  if (request.debug_capture == nullptr) {
    return;
  }
  const std::size_t count = request.residue_count *
                            request.descriptor.neighbor_count *
                            request.descriptor.hidden_dimension;
  capture_debug_tensor(request, request.debug_capture->edge_embedding_output,
                       request.workspace->edge_state.data, count);
}

float* message_layer_capture_target(Mpnn64DebugCapture& capture,
                                    std::size_t layer_index) noexcept {
  switch (layer_index) {
    case 0:
      return capture.message_layer_0_output;
    case 1:
      return capture.message_layer_1_output;
    case 2:
      return capture.message_layer_2_output;
    default:
      return nullptr;
  }
}

void capture_message_layer_output(const Mpnn64ForwardRequest& request,
                                  std::size_t layer_index) noexcept {
  if (request.debug_capture == nullptr) {
    return;
  }
  const std::size_t count =
      request.residue_count * request.descriptor.hidden_dimension;
  capture_debug_tensor(
      request, message_layer_capture_target(*request.debug_capture, layer_index),
      request.workspace->residue_state.data, count);
}

void capture_final_encoder_output(const Mpnn64ForwardRequest& request,
                                  const Mpnn64ForwardOutput& output) noexcept {
  if (request.debug_capture == nullptr) {
    return;
  }
  const std::size_t count =
      request.residue_count * request.descriptor.hidden_dimension;
  capture_debug_tensor(request, request.debug_capture->final_encoder_output,
                       output.embeddings, count);
}

#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
bool has_dumper(const Mpnn64ForwardRequest& request) noexcept {
  return request.intermediate_dumper.callback != nullptr;
}

void dump_tensor(const Mpnn64ForwardRequest& request,
                 const char* name,
                 const float* data,
                 std::size_t rank,
                 std::array<std::size_t, 4> shape) noexcept {
  if (!has_dumper(request)) {
    return;
  }
  Mpnn64IntermediateTensor tensor{};
  tensor.name = name;
  tensor.data = data;
  tensor.dtype = Mpnn64IntermediateDtype::Float32;
  tensor.rank = rank;
  for (std::size_t index = 0; index < 4; ++index) {
    tensor.shape[index] = shape[index];
  }
  request.intermediate_dumper.callback(
      tensor, request.intermediate_dumper.user_data);
}

void dump_tensor(const Mpnn64ForwardRequest& request,
                 const char* name,
                 const std::int32_t* data,
                 std::size_t rank,
                 std::array<std::size_t, 4> shape) noexcept {
  if (!has_dumper(request)) {
    return;
  }
  Mpnn64IntermediateTensor tensor{};
  tensor.name = name;
  tensor.data = data;
  tensor.dtype = Mpnn64IntermediateDtype::Int32;
  tensor.rank = rank;
  for (std::size_t index = 0; index < 4; ++index) {
    tensor.shape[index] = shape[index];
  }
  request.intermediate_dumper.callback(
      tensor, request.intermediate_dumper.user_data);
}

std::int32_t residue_index(const Mpnn64ForwardRequest& request,
                           std::size_t residue) noexcept {
  return request.residue_indices != nullptr
             ? request.residue_indices[residue]
             : static_cast<std::int32_t>(residue);
}

std::int32_t chain_label(const Mpnn64ForwardRequest& request,
                         std::size_t residue) noexcept {
  return request.chain_labels != nullptr ? request.chain_labels[residue] : 0;
}

std::size_t positional_class(const Mpnn64ForwardRequest& request,
                             std::size_t residue,
                             std::size_t neighbor) noexcept {
  if (chain_label(request, residue) != chain_label(request, neighbor)) {
    return kMpnn64PositionalClassCount - 1;
  }
  std::int32_t offset =
      residue_index(request, residue) - residue_index(request, neighbor);
  offset = std::max(-kMpnn64MaxRelativePosition,
                    std::min(kMpnn64MaxRelativePosition, offset));
  return static_cast<std::size_t>(offset + kMpnn64MaxRelativePosition);
}

void dump_knn_distances(const Mpnn64ForwardRequest& request) noexcept {
  if (!has_dumper(request)) {
    return;
  }
  const std::size_t slot_count =
      request.residue_count * request.descriptor.neighbor_count;
  std::vector<float> distances(slot_count, 0.0F);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const float squared =
        request.workspace->neighbor_squared_distances.data[slot];
    distances[slot] = std::isfinite(squared) ? std::sqrt(squared) : squared;
  }
  dump_tensor(request, "knn.distances", distances.data(), 2,
              {request.residue_count, request.descriptor.neighbor_count, 0, 0});
}

void dump_positional_onehot(const Mpnn64ForwardRequest& request) noexcept {
  if (!has_dumper(request)) {
    return;
  }
  const std::size_t slot_count =
      request.residue_count * request.descriptor.neighbor_count;
  std::vector<float> onehot(slot_count * kMpnn64PositionalClassCount, 0.0F);
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < request.descriptor.neighbor_count; ++neighbor_slot) {
      const std::size_t slot =
          residue * request.descriptor.neighbor_count + neighbor_slot;
      const std::int32_t neighbor =
          request.workspace->neighbor_indices.data[slot];
      std::size_t cls = kMpnn64PositionalClassCount - 1;
      if (mpnn_inline::valid_neighbor_inline(request, neighbor)) {
        cls = positional_class(request, residue,
                               static_cast<std::size_t>(neighbor));
      }
      onehot[slot * kMpnn64PositionalClassCount + cls] = 1.0F;
    }
  }
  dump_tensor(request, "positional.onehot", onehot.data(), 3,
              {request.residue_count, request.descriptor.neighbor_count,
               kMpnn64PositionalClassCount, 0});
}
#endif

void build_ca_coordinates(const Mpnn64ForwardRequest& request) noexcept {
  const std::size_t ca_atom =
      mpnn_inline::atom_index_inline(hikoboshi::universal::CanonicalAtom::CA);
  float* ca_coordinates = request.workspace->ca_coordinates.data;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    const bool missing =
        mpnn_inline::atom_missing_inline(request, residue, ca_atom);
    for (std::size_t axis = 0; axis < kMpnn64AxisCount; ++axis) {
      ca_coordinates[residue * kMpnn64AxisCount + axis] =
          missing ? 0.0F
                  : request.coordinates[coord_offset(residue, ca_atom, axis)];
    }
  }
}

void build_knn(const Mpnn64ForwardRequest& request) noexcept {
  const Mpnn64Descriptor& descriptor = request.descriptor;
  Mpnn64Workspace& workspace = *request.workspace;
  hikoboshi::primitives::compute::KnnScalarRequest knn{};
  knn.query_coordinates = workspace.ca_coordinates.data;
  knn.target_coordinates = workspace.ca_coordinates.data;
  knn.query_count = request.residue_count;
  knn.target_count = request.residue_count;
  knn.k = descriptor.neighbor_count;
  knn.include_self = true;
  // build_ca_coordinates writes (0, 0, 0) into ca_coordinates whenever a
  // residue's CA is Missing. Asking knn_scalar to treat zero coordinates
  // as invalid promotes those residues to D_max in the rank, matching
  // PyTorch ProteinMPNN's masked topk in `_dist_and_neighbors`. Without
  // this flag the missing-residue (0, 0, 0) is treated as a real position
  // near the origin, producing the d3ku8a_ KNN drift in mp3.
  knn.treat_zero_coords_as_invalid = true;
  hikoboshi::primitives::compute::KnnScalarOutput knn_output{};
  knn_output.neighbor_indices = workspace.neighbor_indices.data;
  knn_output.neighbor_squared_distances =
      workspace.neighbor_squared_distances.data;
  hikoboshi::dispatch::knn_forward(hikoboshi::dispatch::ScalarTag{}, knn,
                                 knn_output);
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  dump_tensor(request, "knn.indices", workspace.neighbor_indices.data, 2,
              {request.residue_count, descriptor.neighbor_count, 0, 0});
  dump_knn_distances(request);
#endif
}

void build_edge_rbf_features(const Mpnn64ForwardRequest& request) noexcept {
  const Mpnn64Descriptor& descriptor = request.descriptor;
  Mpnn64Workspace& workspace = *request.workspace;
  hikoboshi::modules::mpnn::EdgeRbfFeaturesRequest edge{};
  edge.coordinates = request.coordinates;
  edge.atom_sources = request.atom_sources;
  edge.neighbor_indices = workspace.neighbor_indices.data;
  edge.residue_indices = request.residue_indices;
  edge.chain_labels = request.chain_labels;
  edge.positional_weight = request.weights->positional_encoding.weight;
  edge.positional_bias = request.weights->positional_encoding.bias;
  edge.residue_count = request.residue_count;
  edge.neighbor_count = descriptor.neighbor_count;
  edge.rbf_count = descriptor.rbf_count;

  hikoboshi::modules::mpnn::EdgeRbfFeaturesOutput output{};
  output.atom_pair_squared_distances = workspace.residue_features.data;
  output.rbf_features = workspace.rbf_features.data;
  output.edge_features = workspace.residue_features.data;
  hikoboshi::modules::mpnn::edge_rbf_features_scalar(edge, output);
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  dump_tensor(request, "rbf.features", workspace.rbf_features.data, 4,
              {request.residue_count, descriptor.neighbor_count,
               kMpnn64AtomPairCount, descriptor.rbf_count});
  dump_positional_onehot(request);
#endif
}

void apply_edge_embedding(const Mpnn64ForwardRequest& request) noexcept {
  const Mpnn64Descriptor& descriptor = request.descriptor;
  Mpnn64Workspace& workspace = *request.workspace;
  const Mpnn64MemoryPlan plan{request.residue_count,
                              descriptor.hidden_dimension,
                              descriptor.neighbor_count,
                              descriptor.rbf_count,
                              descriptor.layer_count};
  const std::size_t slot_count = mpnn64_neighbor_slot_count(plan);
  const std::size_t edge_dim = mpnn64_edge_feature_dimension(plan);
  const std::size_t hidden_count = slot_count * descriptor.hidden_dimension;
  const std::size_t neighbor_count =
      mpnn_inline::active_neighbor_count_inline(request);

  if (neighbor_count < descriptor.neighbor_count) {
    float invalid_edge_input[kMpnn64EdgeFeatureCount] = {};
    float invalid_linear[kMpnn64HiddenDimension] = {};
    float invalid_norm[kMpnn64HiddenDimension] = {};
    float invalid_projected[kMpnn64HiddenDimension] = {};

    mpnn_inline::linear_row_nt_inline(
        invalid_edge_input, request.weights->edge_embedding.linear,
        descriptor.hidden_dimension, edge_dim, invalid_linear);
    mpnn_inline::layer_norm_rows_inline(
        invalid_linear, request.weights->edge_embedding.norm, 1,
        descriptor.hidden_dimension, invalid_norm);
    mpnn_inline::linear_row_nt_inline(
        invalid_norm, request.weights->W_e, descriptor.hidden_dimension,
        descriptor.hidden_dimension, invalid_projected);

#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
    if (has_dumper(request)) {
      for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
        for (std::size_t neighbor_slot = 0;
             neighbor_slot < descriptor.neighbor_count; ++neighbor_slot) {
          const std::size_t slot =
              residue * descriptor.neighbor_count + neighbor_slot;
          float* edge =
              workspace.edge_state.data + slot * descriptor.hidden_dimension;
          if (neighbor_slot >= neighbor_count ||
              !mpnn_inline::valid_neighbor_inline(
                  request, workspace.neighbor_indices.data[slot])) {
            mpnn_inline::copy_values_inline(invalid_norm, edge,
                                            descriptor.hidden_dimension);
            continue;
          }

          const float* edge_row =
              workspace.residue_features.data + slot * edge_dim;
          float* message =
              workspace.message_state.data + slot * descriptor.hidden_dimension;
          mpnn_inline::linear_row_nt_inline(
              edge_row, request.weights->edge_embedding.linear,
              descriptor.hidden_dimension, edge_dim, message);
          mpnn_inline::layer_norm_rows_inline(
              message, request.weights->edge_embedding.norm, 1,
              descriptor.hidden_dimension, edge);
        }
      }
      dump_tensor(request, "edge_embedding.output", workspace.edge_state.data, 3,
                  {request.residue_count, descriptor.neighbor_count,
                   descriptor.hidden_dimension, 0});
      for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
        for (std::size_t neighbor_slot = 0;
             neighbor_slot < descriptor.neighbor_count; ++neighbor_slot) {
          const std::size_t slot =
              residue * descriptor.neighbor_count + neighbor_slot;
          float* edge =
              workspace.edge_state.data + slot * descriptor.hidden_dimension;
          float* message =
              workspace.message_state.data + slot * descriptor.hidden_dimension;
          if (neighbor_slot >= neighbor_count ||
              !mpnn_inline::valid_neighbor_inline(
                  request, workspace.neighbor_indices.data[slot])) {
            mpnn_inline::copy_values_inline(invalid_projected, message,
                                            descriptor.hidden_dimension);
          } else {
            mpnn_inline::linear_row_nt_inline(
                edge, request.weights->W_e, descriptor.hidden_dimension,
                descriptor.hidden_dimension, message);
          }
          mpnn_inline::copy_values_inline(message, edge,
                                          descriptor.hidden_dimension);
        }
      }
      dump_tensor(request, "W_e.output", workspace.edge_state.data, 3,
                  {request.residue_count, descriptor.neighbor_count,
                   descriptor.hidden_dimension, 0});
      capture_edge_embedding_output(request);
      return;
    }
#endif

    for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
      for (std::size_t neighbor_slot = 0;
           neighbor_slot < descriptor.neighbor_count; ++neighbor_slot) {
        const std::size_t slot =
            residue * descriptor.neighbor_count + neighbor_slot;
        float* edge =
            workspace.edge_state.data + slot * descriptor.hidden_dimension;
        float* message =
            workspace.message_state.data + slot * descriptor.hidden_dimension;
        if (neighbor_slot >= neighbor_count ||
            !mpnn_inline::valid_neighbor_inline(
                request, workspace.neighbor_indices.data[slot])) {
          mpnn_inline::copy_values_inline(invalid_projected, edge,
                                          descriptor.hidden_dimension);
          mpnn_inline::copy_values_inline(invalid_projected, message,
                                          descriptor.hidden_dimension);
          continue;
        }

        const float* edge_row =
            workspace.residue_features.data + slot * edge_dim;
        mpnn_inline::linear_row_nt_inline(
            edge_row, request.weights->edge_embedding.linear,
            descriptor.hidden_dimension, edge_dim, message);
        mpnn_inline::layer_norm_rows_inline(
            message, request.weights->edge_embedding.norm, 1,
            descriptor.hidden_dimension, edge);
        mpnn_inline::linear_row_nt_inline(
            edge, request.weights->W_e, descriptor.hidden_dimension,
            descriptor.hidden_dimension, message);
        mpnn_inline::copy_values_inline(message, edge,
                                        descriptor.hidden_dimension);
      }
    }
    capture_edge_embedding_output(request);
    return;
  }

  mpnn_inline::linear_nt_inline(workspace.residue_features.data,
                                request.weights->edge_embedding.linear,
                                slot_count, descriptor.hidden_dimension,
                                edge_dim, workspace.message_state.data);

  hikoboshi::primitives::compute::LayerNormScalarRequest norm{};
  norm.input = workspace.message_state.data;
  norm.gamma = request.weights->edge_embedding.norm.weight.data;
  norm.beta = request.weights->edge_embedding.norm.bias.data;
  norm.row_count = slot_count;
  norm.row_dimension = descriptor.hidden_dimension;
  norm.epsilon = kMpnn64LayerNormEpsilon;
  hikoboshi::dispatch::layer_norm_forward(hikoboshi::dispatch::ScalarTag{}, norm,
                                        workspace.edge_state.data);
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  dump_tensor(request, "edge_embedding.output", workspace.edge_state.data, 3,
              {request.residue_count, descriptor.neighbor_count,
               descriptor.hidden_dimension, 0});
#endif

  mpnn_inline::linear_nt_inline(workspace.edge_state.data, request.weights->W_e,
                                slot_count, descriptor.hidden_dimension,
                                descriptor.hidden_dimension,
                                workspace.message_state.data);
  mpnn_inline::copy_values_inline(workspace.message_state.data,
                                  workspace.edge_state.data, hidden_count);
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  dump_tensor(request, "W_e.output", workspace.edge_state.data, 3,
              {request.residue_count, descriptor.neighbor_count,
               descriptor.hidden_dimension, 0});
#endif
  capture_edge_embedding_output(request);
}

void initialize_node_embeddings(const Mpnn64ForwardRequest& request) noexcept {
  mpnn_inline::fill_zero_inline(
      request.workspace->residue_state.data,
      request.residue_count * request.descriptor.hidden_dimension);
}

#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
void layer_op_name(char* buffer,
                   std::size_t size,
                   std::size_t layer_index,
                   const char* suffix) noexcept {
  std::snprintf(buffer, size, "layers.%zu.%s", layer_index, suffix);
}

void build_all_message_inputs(const Mpnn64ForwardRequest& request) noexcept {
  hikoboshi::modules::mpnn::MessageInputPackRequest pack{};
  pack.input_embeddings = request.workspace->residue_state.data;
  pack.edge_embeddings = request.workspace->edge_state.data;
  pack.neighbor_indices = request.workspace->neighbor_indices.data;
  pack.workspace = request.workspace;
  pack.residue_count = request.residue_count;
  pack.hidden_dimension = request.descriptor.hidden_dimension;
  pack.neighbor_count = request.descriptor.neighbor_count;
  mpnn_inline::message_input_pack_scalar_inline(pack);
}

void apply_message_layer_dump(const Mpnn64ForwardRequest& request,
                              const Mpnn64LayerWeights& layer,
                              std::size_t layer_index) noexcept {
  Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t slot_count =
      request.residue_count * request.descriptor.neighbor_count;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t input_dim = kMpnn64MessageInputDimension;
  const std::size_t hidden_count = slot_count * hidden;

  build_all_message_inputs(request);
  mpnn_inline::linear_nt_inline(workspace.residue_features.data, layer.W1,
                                slot_count, hidden, input_dim,
                                workspace.message_state.data);
  mpnn_inline::gelu_inplace_inline(workspace.message_state.data, hidden_count);
  mpnn_inline::linear_nt_inline(workspace.message_state.data, layer.W2,
                                slot_count, hidden, hidden,
                                workspace.projected_message_state.data);
  mpnn_inline::gelu_inplace_inline(workspace.projected_message_state.data,
                                   hidden_count);
  mpnn_inline::linear_nt_inline(workspace.projected_message_state.data,
                                layer.W3, slot_count, hidden, hidden,
                                workspace.message_state.data);

  char name[96] = {};
  layer_op_name(name, sizeof(name), layer_index, "message_mlp.pre_sum");
  dump_tensor(request, name, workspace.message_state.data, 3,
              {request.residue_count, request.descriptor.neighbor_count,
               hidden, 0});

  const std::size_t node_count = request.residue_count * hidden;
  mpnn_inline::fill_zero_inline(workspace.residue_scratch.data, node_count);
  const std::size_t neighbor_count =
      mpnn_inline::active_neighbor_count_inline(request);
  const float scale = 1.0F / request.descriptor.message_scale;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    float* update = workspace.residue_scratch.data + residue * hidden;
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot =
          residue * request.descriptor.neighbor_count + neighbor_slot;
      if (!mpnn_inline::valid_neighbor_inline(
              request, workspace.neighbor_indices.data[slot])) {
        continue;
      }
      mpnn_inline::axpy_into_inline(
          update, workspace.message_state.data + slot * hidden, scale, hidden);
    }
  }
  mpnn_inline::axpy_into_inline(workspace.residue_scratch.data,
                                workspace.residue_state.data, 1.0F, node_count);
  mpnn_inline::layer_norm_rows_inline(workspace.residue_scratch.data,
                                      layer.norm1, request.residue_count,
                                      hidden, workspace.residue_state.data);

  layer_op_name(name, sizeof(name), layer_index, "post_sum_residual_norm1");
  dump_tensor(request, name, workspace.residue_state.data, 2,
              {request.residue_count, hidden, 0, 0});
}

void apply_ffn_layer_dump(const Mpnn64ForwardRequest& request,
                          const Mpnn64LayerWeights& layer,
                          std::size_t layer_index) noexcept {
  Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.descriptor.hidden_dimension;

  mpnn_inline::linear_nt_inline(workspace.residue_state.data, layer.ffn.W_in,
                                request.residue_count,
                                kMpnn64FfnHiddenDimension, hidden,
                                workspace.ffn_hidden.data);
  mpnn_inline::gelu_inplace_inline(
      workspace.ffn_hidden.data,
      request.residue_count * kMpnn64FfnHiddenDimension);
  mpnn_inline::linear_nt_inline(workspace.ffn_hidden.data, layer.ffn.W_out,
                                request.residue_count, hidden,
                                kMpnn64FfnHiddenDimension,
                                workspace.residue_scratch.data);

  char name[96] = {};
  layer_op_name(name, sizeof(name), layer_index, "ffn.pre_residual");
  dump_tensor(request, name, workspace.residue_scratch.data, 2,
              {request.residue_count, hidden, 0, 0});

  mpnn_inline::axpy_into_inline(workspace.residue_scratch.data,
                                workspace.residue_state.data, 1.0F,
                                request.residue_count * hidden);
  mpnn_inline::layer_norm_rows_inline(workspace.residue_scratch.data,
                                      layer.norm2, request.residue_count,
                                      hidden, workspace.residue_state.data);
  mpnn_inline::apply_residue_mask_gate_inline(request);

  layer_op_name(name, sizeof(name), layer_index, "post_ffn_residual_norm2");
  dump_tensor(request, name, workspace.residue_state.data, 2,
              {request.residue_count, hidden, 0, 0});
}

void build_all_edge_update_inputs(const Mpnn64ForwardRequest& request) noexcept {
  const Mpnn64Descriptor& descriptor = request.descriptor;
  Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t input_dim = kMpnn64MessageInputDimension;

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    const float* query_state = workspace.residue_state.data + residue * hidden;
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < descriptor.neighbor_count; ++neighbor_slot) {
      const std::size_t slot =
          residue * descriptor.neighbor_count + neighbor_slot;
      float* input = workspace.residue_features.data + slot * input_dim;
      const std::int32_t neighbor = workspace.neighbor_indices.data[slot];
      if (!mpnn_inline::valid_neighbor_inline(request, neighbor)) {
        mpnn_inline::fill_zero_inline(input, input_dim);
        continue;
      }
      mpnn_inline::copy_values_inline(query_state, input, hidden);
      mpnn_inline::copy_values_inline(workspace.edge_state.data + slot * hidden,
                                      input + hidden, hidden);
      mpnn_inline::copy_values_inline(
          workspace.residue_state.data +
              static_cast<std::size_t>(neighbor) * hidden,
          input + 2 * hidden, hidden);
    }
  }
}

void apply_edge_update_dump(const Mpnn64ForwardRequest& request,
                            const Mpnn64LayerWeights& layer,
                            std::size_t layer_index) noexcept {
  const Mpnn64Descriptor& descriptor = request.descriptor;
  Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t slot_count =
      request.residue_count * descriptor.neighbor_count;
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t input_dim = kMpnn64MessageInputDimension;
  const std::size_t hidden_count = slot_count * hidden;

  build_all_edge_update_inputs(request);
  mpnn_inline::linear_nt_inline(workspace.residue_features.data, layer.W11,
                                slot_count, hidden, input_dim,
                                workspace.message_state.data);
  mpnn_inline::gelu_inplace_inline(workspace.message_state.data, hidden_count);
  mpnn_inline::linear_nt_inline(workspace.message_state.data, layer.W12,
                                slot_count, hidden, hidden,
                                workspace.projected_message_state.data);
  mpnn_inline::gelu_inplace_inline(workspace.projected_message_state.data,
                                   hidden_count);
  mpnn_inline::linear_nt_inline(workspace.projected_message_state.data,
                                layer.W13, slot_count, hidden, hidden,
                                workspace.message_state.data);

  char name[96] = {};
  layer_op_name(name, sizeof(name), layer_index, "edge_update.output");
  dump_tensor(request, name, workspace.message_state.data, 3,
              {request.residue_count, descriptor.neighbor_count, hidden, 0});

  const std::size_t neighbor_count =
      mpnn_inline::active_neighbor_count_inline(request);
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot =
          residue * descriptor.neighbor_count + neighbor_slot;
      if (!mpnn_inline::valid_neighbor_inline(
              request, workspace.neighbor_indices.data[slot])) {
        continue;
      }
      float* update = workspace.message_state.data + slot * hidden;
      const float* edge = workspace.edge_state.data + slot * hidden;
      mpnn_inline::layer_norm_residual_row_inline(
          update, edge, layer.norm3, hidden,
          workspace.edge_state.data + slot * hidden);
    }
  }

  layer_op_name(name, sizeof(name), layer_index,
                "post_edge_update_residual_norm3");
  dump_tensor(request, name, workspace.edge_state.data, 3,
              {request.residue_count, descriptor.neighbor_count, hidden, 0});
}

void apply_mpnn_layers_dump(const Mpnn64ForwardRequest& request) noexcept {
  for (std::size_t layer_index = 0;
       layer_index < request.descriptor.layer_count; ++layer_index) {
    const Mpnn64LayerWeights& layer = request.weights->layers[layer_index];
    apply_message_layer_dump(request, layer, layer_index);
    apply_ffn_layer_dump(request, layer, layer_index);
    apply_edge_update_dump(request, layer, layer_index);
    capture_message_layer_output(request, layer_index);
  }
}
#endif

void apply_mpnn_layers(const Mpnn64ForwardRequest& request) noexcept {
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  if (has_dumper(request)) {
    apply_mpnn_layers_dump(request);
    return;
  }
#endif
  for (std::size_t layer_index = 0;
       layer_index < request.descriptor.layer_count; ++layer_index) {
    const Mpnn64LayerWeights& layer = request.weights->layers[layer_index];
    hikoboshi::modules::mpnn::MpnnMessageLayerRequest message{};
    message.input_embeddings = request.workspace->residue_state.data;
    message.edge_embeddings = request.workspace->edge_state.data;
    message.neighbor_indices = request.workspace->neighbor_indices.data;
    message.weights = &layer;
    message.workspace = request.workspace;
    message.residue_count = request.residue_count;
    message.hidden_dimension = request.descriptor.hidden_dimension;
    message.neighbor_count = request.descriptor.neighbor_count;
    message.message_scale = request.descriptor.message_scale;
    hikoboshi::modules::mpnn::MpnnMessageLayerOutput message_output{};
    message_output.updated_node_embeddings =
        request.workspace->residue_state.data;
    mpnn_inline::mpnn_message_layer_scalar_inline(message, message_output);

    hikoboshi::modules::mpnn::MpnnFfnLayerRequest ffn{};
    ffn.input_embeddings = request.workspace->residue_state.data;
    ffn.weights = &layer;
    ffn.workspace = request.workspace;
    ffn.residue_count = request.residue_count;
    ffn.hidden_dimension = request.descriptor.hidden_dimension;
    ffn.ffn_hidden_dimension = kMpnn64FfnHiddenDimension;
    hikoboshi::modules::mpnn::MpnnFfnLayerOutput ffn_output{};
    ffn_output.output_embeddings = request.workspace->residue_state.data;
    mpnn_inline::mpnn_ffn_layer_scalar_inline(ffn, ffn_output);
    mpnn_inline::apply_residue_mask_gate_inline(request);

    mpnn_inline::apply_edge_update_inline(request, layer);
    capture_message_layer_output(request, layer_index);
  }
}

void write_output(const Mpnn64ForwardRequest& request,
                  const Mpnn64ForwardOutput& output) noexcept {
  mpnn_inline::copy_values_inline(
      request.workspace->residue_state.data, output.embeddings,
      request.residue_count * request.descriptor.hidden_dimension);
  capture_final_encoder_output(request, output);
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  dump_tensor(request, "final_encoder.output", output.embeddings, 2,
              {request.residue_count, request.descriptor.hidden_dimension, 0,
               0});
#endif
}

}  // namespace

universal::Status mpnn64_forward_scalar_unchecked(
    const Mpnn64ForwardRequest& request,
    const Mpnn64ForwardOutput& output) noexcept {
  build_ca_coordinates(request);
  build_knn(request);
  build_edge_rbf_features(request);
  apply_edge_embedding(request);
  initialize_node_embeddings(request);
  apply_mpnn_layers(request);
  write_output(request, output);
  return {universal::StatusCode::Ok, ""};
}

}  // namespace hikoboshi::modules::detail

#include <hikoboshi/modules/mpnn/inverse_fold.hpp>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp>
#include <hikoboshi/modules/mpnn/edge_rbf_features.hpp>
#include <hikoboshi/modules/mpnn/proteinmpnn_encoder.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hikoboshi::modules::mpnn {
namespace {

namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_i = hikoboshi::modules::mpnn::detail;
namespace hiko_u = hikoboshi::universal;
namespace pmp = hikoboshi::universal::detail;

constexpr hiko_u::Status kOk{hiko_u::StatusCode::Ok, ""};
constexpr std::size_t kHidden = pmp::kProteinMpnnV48020Hidden;
constexpr std::size_t kVocab = pmp::kProteinMpnnV48020Vocab;
constexpr std::size_t kRbf = pmp::kProteinMpnnV48020RbfCount;
constexpr std::size_t kLayers = pmp::kProteinMpnnV48020NumDecoderLayers;
constexpr std::size_t kDecoderInput =
    pmp::kProteinMpnnV48020DecoderNumInDimension;
constexpr std::size_t kFfnHidden = pmp::kProteinMpnnV48020FfnHidden;

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

hiko_d::Mpnn64LinearWeights adapt_linear(
    const pmp::ProteinMpnnV48020LinearWeights& weights) noexcept {
  return {weights.weight, weights.bias};
}

hiko_d::Mpnn64LinearWeights adapt_edge_linear(
    const pmp::ProteinMpnnV48020EdgeEmbeddingWeights& weights) noexcept {
  return {weights.weight, {nullptr, 0}};
}

hiko_d::Mpnn64NormWeights adapt_norm(
    const pmp::ProteinMpnnV48020NormWeights& weights) noexcept {
  return {weights.weight, weights.bias};
}

std::size_t coord_offset(std::size_t residue,
                         std::size_t atom,
                         std::size_t axis) noexcept {
  return (residue * hiko_u::kCanonicalAtomCount + atom) *
             hiko_u::kCoordinateAxisCount +
         axis;
}

std::size_t atom_offset(std::size_t residue, hiko_u::CanonicalAtom atom) noexcept {
  return residue * hiko_u::kCanonicalAtomCount + static_cast<std::size_t>(atom);
}

bool atom_present(const hiko_u::AtomSource* atom_sources,
                  std::size_t residue,
                  hiko_u::CanonicalAtom atom) noexcept {
  return atom_sources[atom_offset(residue, atom)] != hiko_u::AtomSource::Missing;
}

bool residue_present(const hiko_u::AtomSource* atom_sources,
                     std::size_t residue) noexcept {
  return atom_present(atom_sources, residue, hiko_u::CanonicalAtom::N) &&
         atom_present(atom_sources, residue, hiko_u::CanonicalAtom::CA) &&
         atom_present(atom_sources, residue, hiko_u::CanonicalAtom::C) &&
         atom_present(atom_sources, residue, hiko_u::CanonicalAtom::O);
}

bool valid_neighbor(std::int32_t neighbor, std::size_t residue_count) noexcept {
  return neighbor >= 0 && static_cast<std::size_t>(neighbor) < residue_count;
}

float chain_mask_value(const float* chain_mask,
                       const float* residue_mask,
                       std::size_t residue) noexcept {
  const float chain = chain_mask != nullptr ? chain_mask[residue] : 1.0F;
  return chain * residue_mask[residue];
}

bool valid_token(std::int32_t token_id) noexcept {
  return token_id >= 0 && static_cast<std::size_t>(token_id) < kVocab;
}

hiko_u::Status validate_tokens(const std::int32_t* tokens,
                            std::size_t residue_count) noexcept {
  if (tokens == nullptr && residue_count > 0) {
    return invalid("ProteinMPNN token pointer is null");
  }
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    if (!valid_token(tokens[residue])) {
      return invalid("ProteinMPNN token id is outside the v48 vocabulary");
    }
  }
  return kOk;
}

bool encoder_workspace_matches(const ProteinMpnnInverseFoldRequest& request,
                               std::size_t neighbor_count) noexcept {
  const hiko_d::Mpnn64Workspace& workspace = *request.encoder_workspace;
  const hiko_d::Mpnn64MemoryPlan& plan = workspace.plan;
  const std::size_t residues = request.residue_count;
  const std::size_t slots = residues * neighbor_count;
  const std::size_t edge_rbf = hiko_d::kMpnn64AtomPairCount * kRbf;
  const std::size_t edge_dim = hiko_d::kMpnn64PositionalFeatureCount + edge_rbf;

  if (plan.max_residue_count < residues || plan.hidden_dimension != kHidden ||
      plan.neighbor_count != neighbor_count || plan.rbf_count != kRbf ||
      plan.layer_count < pmp::kProteinMpnnV48020NumEncoderLayers) {
    return false;
  }
  return has_capacity(workspace.ca_coordinates, residues * 3U) &&
         has_capacity(workspace.residue_features,
                      std::max(slots * edge_dim, slots * 3U * kHidden)) &&
         has_capacity(workspace.neighbor_indices, slots) &&
         has_capacity(workspace.neighbor_squared_distances, slots) &&
         has_capacity(workspace.rbf_features, slots * edge_rbf) &&
         has_capacity(workspace.residue_state, residues * kHidden) &&
         has_capacity(workspace.gathered_state, slots * kHidden) &&
         has_capacity(workspace.edge_state, slots * kHidden) &&
         has_capacity(workspace.message_state, slots * kHidden) &&
         has_capacity(workspace.projected_message_state, slots * kHidden) &&
         has_capacity(workspace.residue_scratch, residues * kHidden) &&
         has_capacity(workspace.ffn_hidden, residues * kFfnHidden);
}

bool encoder_workspace_matches(const ProteinMpnnTeacherForcedRequest& request,
                               std::size_t neighbor_count) noexcept {
  ProteinMpnnInverseFoldRequest adapted{};
  adapted.encoder_workspace = request.encoder_workspace;
  adapted.residue_count = request.residue_count;
  return encoder_workspace_matches(adapted, neighbor_count);
}

bool decoder_workspace_matches(ProteinMpnnDecoderWorkspace* workspace,
                               std::size_t residue_count,
                               std::size_t neighbor_count) noexcept {
  if (workspace == nullptr) {
    return false;
  }
  const ProteinMpnnDecoderMemoryPlan& plan = workspace->plan;
  if (plan.max_residue_count < std::max<std::size_t>(residue_count, 1U) ||
      plan.hidden_dimension != kHidden ||
      plan.neighbor_count != neighbor_count ||
      plan.decoder_input_dimension != kDecoderInput ||
      plan.ffn_hidden_dimension != kFfnHidden) {
    return false;
  }
  return has_capacity(workspace->message_input,
                      proteinmpnn_decoder_message_input_count(plan)) &&
         has_capacity(workspace->message_state,
                      proteinmpnn_decoder_neighbor_hidden_count(plan)) &&
         has_capacity(workspace->projected_message_state,
                      proteinmpnn_decoder_neighbor_hidden_count(plan)) &&
         has_capacity(workspace->residue_scratch,
                      proteinmpnn_decoder_residue_hidden_count(plan)) &&
         has_capacity(workspace->ffn_hidden,
                      proteinmpnn_decoder_ffn_hidden_count(plan));
}

void build_residue_mask(const hiko_u::AtomSource* atom_sources,
                        std::size_t residue_count,
                        std::vector<float>& residue_mask,
                        std::vector<std::uint8_t>& validity_mask) {
  residue_mask.assign(residue_count, 0.0F);
  validity_mask.assign(residue_count, 0U);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    if (residue_present(atom_sources, residue)) {
      residue_mask[residue] = 1.0F;
      validity_mask[residue] = 1U;
    }
  }
}

void build_ca_coordinates(const float* coordinates,
                          const hiko_u::AtomSource* atom_sources,
                          std::size_t residue_count,
                          float* ca_coordinates) noexcept {
  const std::size_t ca_atom =
      static_cast<std::size_t>(hiko_u::CanonicalAtom::CA);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    const bool present =
        atom_present(atom_sources, residue, hiko_u::CanonicalAtom::CA);
    for (std::size_t axis = 0; axis < hiko_u::kCoordinateAxisCount; ++axis) {
      ca_coordinates[residue * 3U + axis] =
          present ? coordinates[coord_offset(residue, ca_atom, axis)] : 0.0F;
    }
  }
}

void build_knn(const float* coordinates,
               const hiko_u::AtomSource* atom_sources,
               const std::vector<std::uint8_t>& validity_mask,
               std::size_t residue_count,
               std::size_t neighbor_count,
               hiko_d::Mpnn64Workspace& workspace) noexcept {
  build_ca_coordinates(coordinates, atom_sources, residue_count,
                       workspace.ca_coordinates.data);
  hikoboshi::primitives::compute::KnnScalarRequest knn{};
  knn.query_coordinates = workspace.ca_coordinates.data;
  knn.target_coordinates = workspace.ca_coordinates.data;
  knn.query_validity_mask = validity_mask.data();
  knn.target_validity_mask = validity_mask.data();
  knn.query_count = residue_count;
  knn.target_count = residue_count;
  knn.k = neighbor_count;
  knn.include_self = true;
  knn.treat_zero_coords_as_invalid = false;
  hikoboshi::primitives::compute::KnnScalarOutput output{};
  output.neighbor_indices = workspace.neighbor_indices.data;
  output.neighbor_squared_distances =
      workspace.neighbor_squared_distances.data;
  hikoboshi::dispatch::knn_forward(hikoboshi::dispatch::ScalarTag{}, knn, output);
}

void build_edge_embeddings(
    const float* coordinates,
    const hiko_u::AtomSource* atom_sources,
    const std::int32_t* residue_indices,
    const std::int32_t* chain_labels,
    const pmp::ProteinMpnnV48020Weights& weights,
    std::size_t residue_count,
    std::size_t neighbor_count,
    hiko_d::Mpnn64Workspace& workspace,
    std::vector<float>& edge_embeddings) noexcept {
  EdgeRbfFeaturesRequest edge{};
  edge.coordinates = coordinates;
  edge.atom_sources = atom_sources;
  edge.neighbor_indices = workspace.neighbor_indices.data;
  edge.residue_indices = residue_indices;
  edge.chain_labels = chain_labels;
  edge.positional_weight = weights.features.embeddings.linear.weight;
  edge.positional_bias = weights.features.embeddings.linear.bias;
  edge.residue_count = residue_count;
  edge.neighbor_count = neighbor_count;
  edge.rbf_count = kRbf;

  EdgeRbfFeaturesOutput edge_output{};
  edge_output.atom_pair_squared_distances = workspace.residue_features.data;
  edge_output.rbf_features = workspace.rbf_features.data;
  edge_output.edge_features = workspace.residue_features.data;
  edge_rbf_features_scalar(edge, edge_output);

  const std::size_t slots = residue_count * neighbor_count;
  const std::size_t edge_dim = hiko_d::kMpnn64PositionalFeatureCount +
                               hiko_d::kMpnn64AtomPairCount * kRbf;
  hiko_i::linear_nt_inline(workspace.residue_features.data,
                        adapt_edge_linear(weights.features.edge_embedding),
                        slots, kHidden, edge_dim,
                        workspace.message_state.data);
  hiko_i::layer_norm_rows_inline(workspace.message_state.data,
                              adapt_norm(weights.features.norm_edges), slots,
                              kHidden, workspace.edge_state.data);
  hiko_i::linear_nt_inline(workspace.edge_state.data, adapt_linear(weights.W_e),
                        slots, kHidden, kHidden,
                        workspace.message_state.data);
  hiko_i::copy_values_inline(workspace.message_state.data, edge_embeddings.data(),
                          slots * kHidden);
}

void build_mask_attend(const std::vector<float>& residue_mask,
                       const std::int32_t* edge_indices,
                       std::size_t residue_count,
                       std::size_t neighbor_count,
                       std::vector<float>& mask_attend) {
  mask_attend.assign(residue_count * neighbor_count, 0.0F);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    for (std::size_t slot_index = 0; slot_index < neighbor_count; ++slot_index) {
      const std::size_t slot = residue * neighbor_count + slot_index;
      const std::int32_t neighbor = edge_indices[slot];
      if (valid_neighbor(neighbor, residue_count)) {
        mask_attend[slot] =
            residue_mask[residue] *
            residue_mask[static_cast<std::size_t>(neighbor)];
      }
    }
  }
}

hiko_u::Status encode_backbone(
    const float* coordinates,
    const hiko_u::AtomSource* atom_sources,
    const std::int32_t* residue_indices,
    const std::int32_t* chain_labels,
    const pmp::ProteinMpnnV48020Weights& weights,
    hiko_d::Mpnn64Workspace& workspace,
    std::size_t residue_count,
    std::size_t neighbor_count,
    std::vector<float>& residue_mask,
    std::vector<float>& h_v,
    std::vector<float>& h_e,
    std::vector<std::int32_t>& e_idx) {
  std::vector<std::uint8_t> validity_mask;
  build_residue_mask(atom_sources, residue_count, residue_mask, validity_mask);
  build_knn(coordinates, atom_sources, validity_mask, residue_count,
            neighbor_count, workspace);

  std::vector<float> h_e_initial(residue_count * neighbor_count * kHidden,
                                 0.0F);
  build_edge_embeddings(coordinates, atom_sources, residue_indices, chain_labels,
                        weights, residue_count, neighbor_count, workspace,
                        h_e_initial);
  std::vector<float> mask_attend;
  build_mask_attend(residue_mask, workspace.neighbor_indices.data,
                    residue_count, neighbor_count, mask_attend);

  h_v.assign(residue_count * kHidden, 0.0F);
  h_e.assign(residue_count * neighbor_count * kHidden, 0.0F);
  e_idx.assign(residue_count * neighbor_count, -1);

  ProteinMpnnEncoderRequest encoder{};
  encoder.input_node_embeddings = nullptr;
  encoder.input_edge_embeddings = h_e_initial.data();
  encoder.edge_indices = workspace.neighbor_indices.data;
  encoder.mask_v = residue_mask.data();
  encoder.mask_attend = mask_attend.data();
  encoder.weights = &weights;
  encoder.workspace = &workspace;
  encoder.residue_count = residue_count;
  encoder.descriptor = {kHidden, neighbor_count,
                        pmp::kProteinMpnnV48020NumEncoderLayers,
                        pmp::kProteinMpnnV48020MessageScale};

  ProteinMpnnEncoderOutput encoder_output{};
  encoder_output.node_embeddings = h_v.data();
  encoder_output.edge_embeddings = h_e.data();
  encoder_output.edge_indices = e_idx.data();
  encoder_output.residue_count = residue_count;
  encoder_output.hidden_dimension = kHidden;
  encoder_output.neighbor_count = neighbor_count;
  return proteinmpnn_encoder_scalar(encoder, encoder_output);
}

void build_h_es(const std::vector<float>& h_s,
                const std::vector<float>& h_e,
                const std::vector<std::int32_t>& e_idx,
                std::size_t residue_count,
                std::size_t neighbor_count,
                std::vector<float>& h_es) {
  h_es.assign(residue_count * neighbor_count * 2U * kHidden, 0.0F);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot = residue * neighbor_count + neighbor_slot;
      float* row = h_es.data() + slot * 2U * kHidden;
      hiko_i::copy_values_inline(h_e.data() + slot * kHidden, row, kHidden);
      const std::int32_t neighbor = e_idx[slot];
      if (valid_neighbor(neighbor, residue_count)) {
        hiko_i::copy_values_inline(
            h_s.data() + static_cast<std::size_t>(neighbor) * kHidden,
            row + kHidden, kHidden);
      }
    }
  }
}

void build_h_exv_encoder(const std::vector<float>& h_v_encoder,
                         const std::vector<float>& h_e,
                         const std::vector<std::int32_t>& e_idx,
                         std::size_t residue_count,
                         std::size_t neighbor_count,
                         std::vector<float>& h_exv_encoder) {
  h_exv_encoder.assign(residue_count * neighbor_count * kDecoderInput, 0.0F);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot = residue * neighbor_count + neighbor_slot;
      float* row = h_exv_encoder.data() + slot * kDecoderInput;
      hiko_i::copy_values_inline(h_e.data() + slot * kHidden, row, kHidden);
      const std::int32_t neighbor = e_idx[slot];
      if (valid_neighbor(neighbor, residue_count)) {
        hiko_i::copy_values_inline(
            h_v_encoder.data() + static_cast<std::size_t>(neighbor) * kHidden,
            row + 2U * kHidden, kHidden);
      }
    }
  }
}

void build_h_esv(const std::vector<float>& h_v_layer,
                 const std::vector<float>& h_es,
                 const std::vector<std::int32_t>& e_idx,
                 std::size_t residue_count,
                 std::size_t neighbor_count,
                 std::vector<float>& h_esv) {
  h_esv.assign(residue_count * neighbor_count * kDecoderInput, 0.0F);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot = residue * neighbor_count + neighbor_slot;
      float* row = h_esv.data() + slot * kDecoderInput;
      hiko_i::copy_values_inline(h_es.data() + slot * 2U * kHidden, row,
                              2U * kHidden);
      const std::int32_t neighbor = e_idx[slot];
      if (valid_neighbor(neighbor, residue_count)) {
        hiko_i::copy_values_inline(
            h_v_layer.data() + static_cast<std::size_t>(neighbor) * kHidden,
            row + 2U * kHidden, kHidden);
      }
    }
  }
}

void combine_context(const std::vector<float>& decoder_context,
                     const std::vector<float>& encoder_context,
                     const std::vector<float>& mask_bw,
                     const std::vector<float>& mask_fw,
                     std::size_t slot_count,
                     std::vector<float>& output) {
  output.assign(slot_count * kDecoderInput, 0.0F);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const float bw = mask_bw[slot];
    const float fw = mask_fw[slot];
    const float* decoder = decoder_context.data() + slot * kDecoderInput;
    const float* encoder = encoder_context.data() + slot * kDecoderInput;
    float* out = output.data() + slot * kDecoderInput;
    for (std::size_t dim = 0; dim < kDecoderInput; ++dim) {
      out[dim] = bw * decoder[dim] + fw * encoder[dim];
    }
  }
}

void apply_mask_to_context(const std::vector<float>& input,
                           const std::vector<float>& mask,
                           std::size_t slot_count,
                           std::vector<float>& output) {
  output.assign(slot_count * kDecoderInput, 0.0F);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    const float value = mask[slot];
    const float* row_in = input.data() + slot * kDecoderInput;
    float* row_out = output.data() + slot * kDecoderInput;
    for (std::size_t dim = 0; dim < kDecoderInput; ++dim) {
      row_out[dim] = value * row_in[dim];
    }
  }
}

void write_log_probs(const std::vector<float>& logits,
                     float* log_probs,
                     std::size_t residue_count) {
  hikoboshi::primitives::compute::LogSoftmaxScalarRequest request{};
  request.input = logits.data();
  request.row_count = residue_count;
  request.row_dimension = kVocab;
  hikoboshi::primitives::compute::LogSoftmaxScalarOutput output{};
  output.output = log_probs;
  hikoboshi::dispatch::log_softmax_forward(hikoboshi::dispatch::ScalarTag{},
                                         request, output);
}

float average_nll(const std::int32_t* tokens,
                  const float* log_probs,
                  const float* chain_mask,
                  const std::vector<float>& residue_mask,
                  std::size_t residue_count) noexcept {
  double loss_sum = 0.0;
  double mask_sum = 0.0;
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    const float active =
        chain_mask_value(chain_mask, residue_mask.data(), residue);
    if (active == 0.0F) {
      continue;
    }
    const std::int32_t token = tokens[residue];
    if (!valid_token(token)) {
      continue;
    }
    loss_sum -= static_cast<double>(
        log_probs[residue * kVocab + static_cast<std::size_t>(token)]);
    mask_sum += static_cast<double>(active);
  }
  return mask_sum > 0.0 ? static_cast<float>(loss_sum / mask_sum) : 0.0F;
}

void write_sequence(const std::int32_t* tokens,
                    std::size_t residue_count,
                    char* sequence) noexcept {
  if (sequence == nullptr) {
    return;
  }
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    sequence[residue] = proteinmpnn_v48_020_token_to_aa(tokens[residue]);
  }
  sequence[residue_count] = '\0';
}

hiko_u::Status build_decode_masks(
    const ProteinMpnnInverseFoldOptions& options,
    const float* chain_mask,
    const std::int32_t* input_decoding_order,
    const std::vector<float>& residue_mask,
    const std::vector<std::int32_t>& e_idx,
    std::size_t residue_count,
    std::size_t neighbor_count,
    std::int32_t* output_order,
    std::vector<std::int32_t>& local_order,
    std::vector<float>& mask_bw,
    std::vector<float>& mask_fw) {
  local_order.assign(residue_count, 0);
  ProteinMpnnDecodeOrderRequest order_request{};
  order_request.chain_mask = chain_mask;
  order_request.residue_mask = residue_mask.data();
  order_request.input_order = input_decoding_order;
  order_request.residue_count = residue_count;
  order_request.seed = options.decode_order_seed;
  order_request.use_input_decoding_order = options.use_input_decoding_order;
  ProteinMpnnDecodeOrderOutput order_output{};
  order_output.decoding_order =
      output_order != nullptr ? output_order : local_order.data();
  hiko_u::Status status =
      proteinmpnn_decode_order_scalar(order_request, order_output);
  if (!status.ok()) {
    return status;
  }
  if (output_order != nullptr) {
    std::memcpy(local_order.data(), output_order,
                residue_count * sizeof(std::int32_t));
  }

  mask_bw.assign(residue_count * neighbor_count, 0.0F);
  mask_fw.assign(residue_count * neighbor_count, 0.0F);
  ProteinMpnnCausalMaskRequest mask_request{};
  mask_request.decoding_order = local_order.data();
  mask_request.edge_indices = e_idx.data();
  mask_request.residue_mask = residue_mask.data();
  mask_request.residue_count = residue_count;
  mask_request.neighbor_count = neighbor_count;
  ProteinMpnnCausalMaskOutput mask_output{};
  mask_output.mask_bw = mask_bw.data();
  mask_output.mask_fw = mask_fw.data();
  return proteinmpnn_causal_masks_scalar(mask_request, mask_output);
}

hiko_u::Status validate_common(const float* coordinates,
                            const hiko_u::AtomSource* atom_sources,
                            const pmp::ProteinMpnnV48020Weights* weights,
                            hiko_d::Mpnn64Workspace* encoder_workspace,
                            ProteinMpnnDecoderWorkspace* decoder_workspace,
                            std::size_t residue_count) noexcept {
  if (residue_count > 0 && coordinates == nullptr) {
    return invalid("ProteinMPNN inverse fold coordinates pointer is null");
  }
  if (residue_count > 0 && atom_sources == nullptr) {
    return invalid("ProteinMPNN inverse fold atom_sources pointer is null");
  }
  if (weights == nullptr) {
    return invalid("ProteinMPNN inverse fold weights pointer is null");
  }
  if (encoder_workspace == nullptr || decoder_workspace == nullptr) {
    return invalid("ProteinMPNN inverse fold workspaces must be non-null");
  }
  return kOk;
}

hiko_u::Status run_teacher_forced_decoder(
    const pmp::ProteinMpnnV48020Weights& weights,
    ProteinMpnnDecoderWorkspace& decoder_workspace,
    const std::int32_t* tokens,
    const float* chain_mask,
    const std::vector<float>& residue_mask,
    const std::vector<float>& h_v_encoder,
    const std::vector<float>& h_e,
    const std::vector<std::int32_t>& e_idx,
    std::size_t residue_count,
    std::size_t neighbor_count,
    const ProteinMpnnInverseFoldOptions& options,
    const std::int32_t* input_decoding_order,
    float* output_log_probs,
    std::int32_t* output_order,
    float* sequence_score) {
  std::vector<std::int32_t> local_order;
  std::vector<float> mask_bw;
  std::vector<float> mask_fw;
  hiko_u::Status status = build_decode_masks(
      options, chain_mask, input_decoding_order, residue_mask, e_idx,
      residue_count, neighbor_count, output_order, local_order, mask_bw,
      mask_fw);
  if (!status.ok()) {
    return status;
  }

  std::vector<float> h_s(residue_count * kHidden, 0.0F);
  ProteinMpnnSequenceEmbeddingRequest seq{};
  seq.token_ids = tokens;
  seq.weights = &weights.W_s;
  seq.token_count = residue_count;
  seq.vocab_size = kVocab;
  seq.hidden_dimension = kHidden;
  ProteinMpnnSequenceEmbeddingOutput seq_output{};
  seq_output.embeddings = h_s.data();
  proteinmpnn_sequence_embedding_scalar(seq, seq_output);

  std::vector<float> h_es;
  std::vector<float> h_exv_encoder;
  std::vector<float> h_esv;
  std::vector<float> context;
  build_h_es(h_s, h_e, e_idx, residue_count, neighbor_count, h_es);
  build_h_exv_encoder(h_v_encoder, h_e, e_idx, residue_count, neighbor_count,
                      h_exv_encoder);

  std::vector<float> h_v = h_v_encoder;
  std::vector<float> next_h_v(residue_count * kHidden, 0.0F);
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    build_h_esv(h_v, h_es, e_idx, residue_count, neighbor_count, h_esv);
    combine_context(h_esv, h_exv_encoder, mask_bw, mask_fw,
                    residue_count * neighbor_count, context);
    ProteinMpnnDecoderLayerRequest layer_request{};
    layer_request.node_embeddings = h_v.data();
    layer_request.edge_context = context.data();
    layer_request.attention_mask = nullptr;
    layer_request.residue_mask = residue_mask.data();
    layer_request.weights = &weights.decoder_layers[layer];
    layer_request.workspace = &decoder_workspace;
    layer_request.residue_count = residue_count;
    layer_request.hidden_dimension = kHidden;
    layer_request.neighbor_count = neighbor_count;
    layer_request.decoder_input_dimension = kDecoderInput;
    layer_request.ffn_hidden_dimension = kFfnHidden;
    layer_request.message_scale = pmp::kProteinMpnnV48020MessageScale;
    ProteinMpnnDecoderLayerOutput layer_output{};
    layer_output.updated_node_embeddings = next_h_v.data();
    proteinmpnn_decoder_layer_scalar(layer_request, layer_output);
    h_v.swap(next_h_v);
  }

  std::vector<float> logits(residue_count * kVocab, 0.0F);
  ProteinMpnnLogitsHeadRequest head{};
  head.node_embeddings = h_v.data();
  head.weights = &weights.W_out;
  head.residue_count = residue_count;
  head.hidden_dimension = kHidden;
  head.vocab_size = kVocab;
  ProteinMpnnLogitsHeadOutput head_output{};
  head_output.logits = logits.data();
  proteinmpnn_logits_head_scalar(head, head_output);
  write_log_probs(logits, output_log_probs, residue_count);
  if (sequence_score != nullptr) {
    *sequence_score =
        average_nll(tokens, output_log_probs, chain_mask, residue_mask,
                    residue_count);
  }
  return kOk;
}

void build_h_es_row(const std::vector<float>& h_s,
                    const std::vector<float>& h_e,
                    const std::vector<std::int32_t>& e_idx,
                    std::size_t residue,
                    std::size_t residue_count,
                    std::size_t neighbor_count,
                    std::vector<float>& h_es_row) {
  h_es_row.assign(neighbor_count * 2U * kHidden, 0.0F);
  for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
       ++neighbor_slot) {
    const std::size_t slot = residue * neighbor_count + neighbor_slot;
    float* row = h_es_row.data() + neighbor_slot * 2U * kHidden;
    hiko_i::copy_values_inline(h_e.data() + slot * kHidden, row, kHidden);
    const std::int32_t neighbor = e_idx[slot];
    if (valid_neighbor(neighbor, residue_count)) {
      hiko_i::copy_values_inline(
          h_s.data() + static_cast<std::size_t>(neighbor) * kHidden,
          row + kHidden, kHidden);
    }
  }
}

void build_h_esv_row(const float* h_v_layer,
                     const std::vector<float>& h_es_row,
                     const std::vector<std::int32_t>& e_idx,
                     std::size_t residue,
                     std::size_t residue_count,
                     std::size_t neighbor_count,
                     std::vector<float>& h_esv_row) {
  h_esv_row.assign(neighbor_count * kDecoderInput, 0.0F);
  for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
       ++neighbor_slot) {
    const std::size_t slot = residue * neighbor_count + neighbor_slot;
    float* row = h_esv_row.data() + neighbor_slot * kDecoderInput;
    hiko_i::copy_values_inline(h_es_row.data() + neighbor_slot * 2U * kHidden,
                            row, 2U * kHidden);
    const std::int32_t neighbor = e_idx[slot];
    if (valid_neighbor(neighbor, residue_count)) {
      hiko_i::copy_values_inline(
          h_v_layer + static_cast<std::size_t>(neighbor) * kHidden,
          row + 2U * kHidden, kHidden);
    }
  }
}

void combine_context_row(const std::vector<float>& decoder_row,
                         const std::vector<float>& encoder_fw,
                         const std::vector<float>& mask_bw,
                         std::size_t residue,
                         std::size_t neighbor_count,
                         std::vector<float>& output) {
  output.assign(neighbor_count * kDecoderInput, 0.0F);
  for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
       ++neighbor_slot) {
    const std::size_t slot = residue * neighbor_count + neighbor_slot;
    const float bw = mask_bw[slot];
    const float* decoder = decoder_row.data() + neighbor_slot * kDecoderInput;
    const float* encoder = encoder_fw.data() + slot * kDecoderInput;
    float* out = output.data() + neighbor_slot * kDecoderInput;
    for (std::size_t dim = 0; dim < kDecoderInput; ++dim) {
      out[dim] = bw * decoder[dim] + encoder[dim];
    }
  }
}

void logits_for_one(const pmp::ProteinMpnnV48020Weights& weights,
                    const float* h_v,
                    float* logits) noexcept {
  ProteinMpnnLogitsHeadRequest head{};
  head.node_embeddings = h_v;
  head.weights = &weights.W_out;
  head.residue_count = 1;
  head.hidden_dimension = kHidden;
  head.vocab_size = kVocab;
  ProteinMpnnLogitsHeadOutput output{};
  output.logits = logits;
  proteinmpnn_logits_head_scalar(head, output);
}

std::int32_t fallback_true_token(const std::int32_t* true_tokens,
                                 std::size_t residue) noexcept {
  if (true_tokens != nullptr && valid_token(true_tokens[residue])) {
    return true_tokens[residue];
  }
  return static_cast<std::int32_t>(kVocab - 1U);
}

hiko_u::Status run_autoregressive_decoder(
    const pmp::ProteinMpnnV48020Weights& weights,
    ProteinMpnnDecoderWorkspace& decoder_workspace,
    const std::int32_t* true_tokens,
    const float* chain_mask,
    const std::vector<float>& residue_mask,
    const std::vector<float>& h_v_encoder,
    const std::vector<float>& h_e,
    const std::vector<std::int32_t>& e_idx,
    std::size_t residue_count,
    std::size_t neighbor_count,
    const ProteinMpnnInverseFoldOptions& options,
    const std::int32_t* input_decoding_order,
    ProteinMpnnInverseFoldOutput output) {
  std::vector<std::int32_t> local_order;
  std::vector<float> mask_bw;
  std::vector<float> mask_fw;
  hiko_u::Status status = build_decode_masks(
      options, chain_mask, input_decoding_order, residue_mask, e_idx,
      residue_count, neighbor_count, output.decoding_order, local_order,
      mask_bw, mask_fw);
  if (!status.ok()) {
    return status;
  }

  const std::size_t slots = residue_count * neighbor_count;
  std::vector<float> h_exv_encoder;
  std::vector<float> h_exv_encoder_fw;
  build_h_exv_encoder(h_v_encoder, h_e, e_idx, residue_count, neighbor_count,
                      h_exv_encoder);
  apply_mask_to_context(h_exv_encoder, mask_fw, slots, h_exv_encoder_fw);

  std::vector<float> h_s(residue_count * kHidden, 0.0F);
  std::vector<float> h_v_stack((kLayers + 1U) * residue_count * kHidden, 0.0F);
  if (!h_v_encoder.empty()) {
    hiko_i::copy_values_inline(h_v_encoder.data(), h_v_stack.data(),
                            residue_count * kHidden);
  }
  std::vector<float> h_es_row;
  std::vector<float> h_esv_decoder_row;
  std::vector<float> h_esv_row;
  std::vector<float> node_input(kHidden, 0.0F);
  std::vector<float> node_output(kHidden, 0.0F);
  std::vector<float> logits(kVocab, 0.0F);
  std::vector<float> probabilities(kVocab, 0.0F);
  std::vector<float> log_probs_row(kVocab, 0.0F);

  ProteinMpnnHostRng rng{};
  proteinmpnn_host_rng_seed(rng, options.seed);
  for (std::size_t order_index = 0; order_index < residue_count; ++order_index) {
    const std::size_t residue =
        static_cast<std::size_t>(local_order[order_index]);
    const float residue_active = residue_mask[residue];
    if (residue_active == 0.0F) {
      const std::int32_t token = fallback_true_token(true_tokens, residue);
      output.token_ids[residue] = token;
      std::fill_n(output.log_probs + residue * kVocab, kVocab, 0.0F);
      hiko_i::copy_values_inline(
          weights.W_s.weight.data + static_cast<std::size_t>(token) * kHidden,
          h_s.data() + residue * kHidden, kHidden);
      continue;
    }

    build_h_es_row(h_s, h_e, e_idx, residue, residue_count, neighbor_count,
                   h_es_row);
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
      const float* layer_state =
          h_v_stack.data() + layer * residue_count * kHidden;
      build_h_esv_row(layer_state, h_es_row, e_idx, residue, residue_count,
                      neighbor_count, h_esv_decoder_row);
      combine_context_row(h_esv_decoder_row, h_exv_encoder_fw, mask_bw,
                          residue, neighbor_count, h_esv_row);
      hiko_i::copy_values_inline(layer_state + residue * kHidden,
                              node_input.data(), kHidden);
      float mask_t = residue_active;
      ProteinMpnnDecoderLayerRequest layer_request{};
      layer_request.node_embeddings = node_input.data();
      layer_request.edge_context = h_esv_row.data();
      layer_request.attention_mask = nullptr;
      layer_request.residue_mask = &mask_t;
      layer_request.weights = &weights.decoder_layers[layer];
      layer_request.workspace = &decoder_workspace;
      layer_request.residue_count = 1;
      layer_request.hidden_dimension = kHidden;
      layer_request.neighbor_count = neighbor_count;
      layer_request.decoder_input_dimension = kDecoderInput;
      layer_request.ffn_hidden_dimension = kFfnHidden;
      layer_request.message_scale = pmp::kProteinMpnnV48020MessageScale;
      ProteinMpnnDecoderLayerOutput layer_output{};
      layer_output.updated_node_embeddings = node_output.data();
      proteinmpnn_decoder_layer_scalar(layer_request, layer_output);
      float* next_layer_state =
          h_v_stack.data() + (layer + 1U) * residue_count * kHidden;
      hiko_i::copy_values_inline(node_output.data(),
                              next_layer_state + residue * kHidden, kHidden);
    }

    const float* final_state =
        h_v_stack.data() + kLayers * residue_count * kHidden +
        residue * kHidden;
    logits_for_one(weights, final_state, logits.data());
    ProteinMpnnSampleRequest sample{};
    sample.logits = logits.data();
    sample.vocab_size = kVocab;
    sample.temperature = options.temperature;
    sample.greedy = options.greedy || options.temperature <= 0.0F;
    std::int32_t sampled_token = 0;
    ProteinMpnnSampleOutput sample_output{};
    sample_output.token_id = &sampled_token;
    sample_output.probabilities = probabilities.data();
    sample_output.log_probs = log_probs_row.data();
    status = proteinmpnn_sample_scalar(sample, rng, sample_output);
    if (!status.ok()) {
      return status;
    }

    const float design_mask = chain_mask_value(chain_mask, residue_mask.data(),
                                               residue);
    const std::int32_t token =
        design_mask != 0.0F ? sampled_token
                            : fallback_true_token(true_tokens, residue);
    output.token_ids[residue] = token;
    hiko_i::copy_values_inline(log_probs_row.data(),
                            output.log_probs + residue * kVocab, kVocab);
    hiko_i::copy_values_inline(
        weights.W_s.weight.data + static_cast<std::size_t>(token) * kHidden,
        h_s.data() + residue * kHidden, kHidden);
  }

  write_sequence(output.token_ids, residue_count, output.sequence);
  if (output.sequence_score != nullptr) {
    *output.sequence_score =
        average_nll(output.token_ids, output.log_probs, chain_mask, residue_mask,
                    residue_count);
  }
  return kOk;
}

}  // namespace

std::size_t proteinmpnn_v48_020_effective_neighbor_count(
    std::size_t residue_count) noexcept {
  return std::min<std::size_t>(pmp::kProteinMpnnV48020KNeighbors,
                               residue_count);
}

char proteinmpnn_v48_020_token_to_aa(std::int32_t token_id) noexcept {
  if (!valid_token(token_id)) {
    return 'X';
  }
  return kProteinMpnnV48020Alphabet[static_cast<std::size_t>(token_id)];
}

hiko_u::Status proteinmpnn_teacher_forced_forward_scalar(
    const ProteinMpnnTeacherForcedRequest& request,
    const ProteinMpnnTeacherForcedOutput& output) {
  hiko_u::Status status =
      validate_common(request.coordinates, request.atom_sources, request.weights,
                      request.encoder_workspace, request.decoder_workspace,
                      request.residue_count);
  if (!status.ok()) {
    return status;
  }
  if (output.residue_count < request.residue_count ||
      output.vocab_size != kVocab || output.log_probs == nullptr) {
    return invalid("ProteinMPNN teacher-forced output shape is invalid");
  }
  status = validate_tokens(request.token_ids, request.residue_count);
  if (!status.ok()) {
    return status;
  }
  const std::size_t neighbor_count =
      proteinmpnn_v48_020_effective_neighbor_count(request.residue_count);
  if (request.residue_count == 0) {
    return kOk;
  }
  if (!encoder_workspace_matches(request, neighbor_count) ||
      !decoder_workspace_matches(request.decoder_workspace, request.residue_count,
                                 neighbor_count)) {
    return failed_precondition(
        "ProteinMPNN teacher-forced workspace does not satisfy memory plan");
  }

  std::vector<float> residue_mask;
  std::vector<float> h_v;
  std::vector<float> h_e;
  std::vector<std::int32_t> e_idx;
  status = encode_backbone(request.coordinates, request.atom_sources,
                           request.residue_indices, request.chain_labels,
                           *request.weights, *request.encoder_workspace,
                           request.residue_count, neighbor_count, residue_mask,
                           h_v, h_e, e_idx);
  if (!status.ok()) {
    return status;
  }
  return run_teacher_forced_decoder(
      *request.weights, *request.decoder_workspace, request.token_ids,
      request.chain_mask, residue_mask, h_v, h_e, e_idx, request.residue_count,
      neighbor_count, request.options, request.input_decoding_order,
      output.log_probs, output.decoding_order, output.sequence_score);
}

hiko_u::Status proteinmpnn_inverse_fold_scalar(
    const ProteinMpnnInverseFoldRequest& request,
    const ProteinMpnnInverseFoldOutput& output) {
  hiko_u::Status status =
      validate_common(request.coordinates, request.atom_sources, request.weights,
                      request.encoder_workspace, request.decoder_workspace,
                      request.residue_count);
  if (!status.ok()) {
    return status;
  }
  if (output.residue_count < request.residue_count ||
      output.vocab_size != kVocab || output.token_ids == nullptr ||
      output.log_probs == nullptr || output.decoding_order == nullptr) {
    return invalid("ProteinMPNN inverse-fold output shape is invalid");
  }
  const std::size_t neighbor_count =
      proteinmpnn_v48_020_effective_neighbor_count(request.residue_count);
  if (request.residue_count == 0) {
    if (output.sequence != nullptr) {
      output.sequence[0] = '\0';
    }
    return kOk;
  }
  if (!encoder_workspace_matches(request, neighbor_count) ||
      !decoder_workspace_matches(request.decoder_workspace, request.residue_count,
                                 neighbor_count)) {
    return failed_precondition(
        "ProteinMPNN inverse-fold workspace does not satisfy memory plan");
  }

  std::vector<float> residue_mask;
  std::vector<float> h_v;
  std::vector<float> h_e;
  std::vector<std::int32_t> e_idx;
  status = encode_backbone(request.coordinates, request.atom_sources,
                           request.residue_indices, request.chain_labels,
                           *request.weights, *request.encoder_workspace,
                           request.residue_count, neighbor_count, residue_mask,
                           h_v, h_e, e_idx);
  if (!status.ok()) {
    return status;
  }
  return run_autoregressive_decoder(
      *request.weights, *request.decoder_workspace, request.true_token_ids,
      request.chain_mask, residue_mask, h_v, h_e, e_idx, request.residue_count,
      neighbor_count, request.options, request.input_decoding_order, output);
}

}  // namespace hikoboshi::modules::mpnn

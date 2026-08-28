#include <hikoboshi/modules/mpnn.hpp>

#include <cmath>
#include <cstddef>

namespace hikoboshi::modules {
namespace {

constexpr universal::Status kOk{universal::StatusCode::Ok, ""};

bool has_capacity(const universal::Span<float>& span, std::size_t required) noexcept {
  return span.data != nullptr && span.size >= required;
}

bool has_capacity(const universal::Span<std::int32_t>& span,
                  std::size_t required) noexcept {
  return span.data != nullptr && span.size >= required;
}

universal::Status invalid(const char* detail) noexcept {
  return {universal::StatusCode::InvalidArgument, detail};
}

universal::Status failed_precondition(const char* detail) noexcept {
  return {universal::StatusCode::FailedPrecondition, detail};
}

bool has_exact_size(universal::Span<const float> span,
                    std::size_t required) noexcept {
  return span.data != nullptr && span.size == required;
}

bool has_optional_size(universal::Span<const float> span,
                       std::size_t required) noexcept {
  return (span.data == nullptr && span.size == 0) ||
         (span.data != nullptr && span.size == required);
}

bool has_linear_size(const detail::Mpnn64LinearWeights& weights,
                     std::size_t output_dimension,
                     std::size_t input_dimension) noexcept {
  return has_exact_size(weights.weight, output_dimension * input_dimension) &&
         has_exact_size(weights.bias, output_dimension);
}

bool has_norm_size(const detail::Mpnn64NormWeights& weights,
                   std::size_t dimension) noexcept {
  return has_exact_size(weights.weight, dimension) &&
         has_exact_size(weights.bias, dimension);
}

bool workspace_matches(const Mpnn64Descriptor& descriptor,
                       std::size_t residue_count,
                       const detail::Mpnn64Workspace& workspace) noexcept {
  const detail::Mpnn64MemoryPlan& plan = workspace.plan;
  if (plan.max_residue_count < residue_count ||
      plan.hidden_dimension != descriptor.hidden_dimension ||
      plan.neighbor_count != descriptor.neighbor_count ||
      plan.rbf_count != descriptor.rbf_count ||
      plan.layer_count < descriptor.layer_count) {
    return false;
  }
  return has_capacity(workspace.ca_coordinates,
                      detail::mpnn64_ca_coordinate_count(plan)) &&
         has_capacity(workspace.residue_features,
                      detail::mpnn64_residue_feature_count(plan)) &&
         has_capacity(workspace.neighbor_indices,
                      detail::mpnn64_neighbor_slot_count(plan)) &&
         has_capacity(workspace.neighbor_squared_distances,
                      detail::mpnn64_neighbor_slot_count(plan)) &&
         has_capacity(workspace.rbf_features,
                      detail::mpnn64_neighbor_rbf_count(plan)) &&
         has_capacity(workspace.residue_state,
                      detail::mpnn64_residue_hidden_count(plan)) &&
         has_capacity(workspace.gathered_state,
                      detail::mpnn64_neighbor_hidden_count(plan)) &&
         has_capacity(workspace.edge_state,
                      detail::mpnn64_neighbor_hidden_count(plan)) &&
         has_capacity(workspace.message_state,
                      detail::mpnn64_neighbor_hidden_count(plan)) &&
         has_capacity(workspace.projected_message_state,
                      detail::mpnn64_neighbor_hidden_count(plan)) &&
         has_capacity(workspace.residue_scratch,
                      detail::mpnn64_residue_hidden_count(plan)) &&
         (descriptor.layer_count == 0 ||
          has_capacity(workspace.ffn_hidden,
                       detail::mpnn64_ffn_hidden_count(plan)));
}

universal::Status validate_real_edge_weights(
    const Mpnn64Descriptor& descriptor,
    const detail::Mpnn64Weights& weights) noexcept {
  detail::Mpnn64MemoryPlan scalar_plan{};
  scalar_plan.max_residue_count = 1;
  scalar_plan.hidden_dimension = descriptor.hidden_dimension;
  scalar_plan.neighbor_count = descriptor.neighbor_count;
  scalar_plan.rbf_count = descriptor.rbf_count;
  scalar_plan.layer_count = descriptor.layer_count;
  const std::size_t edge_input_dim =
      detail::mpnn64_edge_feature_dimension(scalar_plan);
  if (edge_input_dim != detail::kMpnn64EdgeFeatureCount) {
    return invalid("MPNN edge feature dimension must be 416");
  }
  if (detail::kMpnn64PositionalFeatureCount !=
          detail::kMpnn64PositionalEncodingCount ||
      detail::kMpnn64PositionalClassCount !=
          detail::kMpnn64PositionalEncodingInputDimension) {
    return invalid("MPNN positional encoding shape is inconsistent");
  }
  if (!has_exact_size(weights.edge_embedding.linear.weight,
                      descriptor.hidden_dimension * edge_input_dim)) {
    return invalid("MPNN edge_embedding.weight must be [64,416]");
  }
  if (!has_optional_size(weights.edge_embedding.linear.bias,
                         descriptor.hidden_dimension)) {
    return invalid("MPNN edge_embedding.bias must be absent or [64]");
  }
  if (!has_exact_size(weights.edge_embedding.norm.weight,
                      descriptor.hidden_dimension) ||
      !has_exact_size(weights.edge_embedding.norm.bias,
                      descriptor.hidden_dimension)) {
    return invalid("MPNN edge_embedding norm weights must be [64]");
  }
  if (!has_exact_size(weights.positional_encoding.weight,
                      detail::kMpnn64PositionalFeatureCount *
                          detail::kMpnn64PositionalClassCount)) {
    return invalid("MPNN positional_encoding.weight must be [16,66]");
  }
  if (!has_optional_size(weights.positional_encoding.bias,
                         detail::kMpnn64PositionalFeatureCount)) {
    return invalid("MPNN positional_encoding.bias must be absent or [16]");
  }
  if (!has_exact_size(weights.W_e.weight,
                      descriptor.hidden_dimension *
                          descriptor.hidden_dimension) ||
      !has_exact_size(weights.W_e.bias, descriptor.hidden_dimension)) {
    return invalid("MPNN W_e weights must be [64,64] and [64]");
  }
  if (descriptor.layer_count > weights.layer_count) {
    return invalid("MPNN layer weight count is smaller than descriptor");
  }
  for (std::size_t layer = 0; layer < descriptor.layer_count; ++layer) {
    const detail::Mpnn64LayerWeights& layer_weights = weights.layers[layer];
    if (!has_linear_size(layer_weights.W1, descriptor.hidden_dimension,
                         detail::kMpnn64MessageInputDimension)) {
      return invalid("MPNN layer W1 weights must be [64,192] and [64]");
    }
    if (!has_linear_size(layer_weights.W11, descriptor.hidden_dimension,
                         detail::kMpnn64MessageInputDimension)) {
      return invalid("MPNN layer W11 weights must be [64,192] and [64]");
    }
    if (!has_linear_size(layer_weights.W12, descriptor.hidden_dimension,
                         descriptor.hidden_dimension) ||
        !has_linear_size(layer_weights.W13, descriptor.hidden_dimension,
                         descriptor.hidden_dimension) ||
        !has_linear_size(layer_weights.W2, descriptor.hidden_dimension,
                         descriptor.hidden_dimension) ||
        !has_linear_size(layer_weights.W3, descriptor.hidden_dimension,
                         descriptor.hidden_dimension)) {
      return invalid("MPNN layer W12/W13/W2/W3 weights must be [64,64] and [64]");
    }
    if (!has_linear_size(layer_weights.ffn.W_in,
                         detail::kMpnn64FfnHiddenDimension,
                         descriptor.hidden_dimension) ||
        !has_linear_size(layer_weights.ffn.W_out, descriptor.hidden_dimension,
                         detail::kMpnn64FfnHiddenDimension)) {
      return invalid("MPNN layer FFN weights must be [256,64] and [64,256]");
    }
    if (!has_norm_size(layer_weights.norm1, descriptor.hidden_dimension) ||
        !has_norm_size(layer_weights.norm2, descriptor.hidden_dimension) ||
        !has_norm_size(layer_weights.norm3, descriptor.hidden_dimension)) {
      return invalid("MPNN layer norm1/norm2/norm3 weights must be [64]");
    }
  }
  return kOk;
}

}  // namespace

namespace detail {
universal::Status mpnn64_forward_scalar_unchecked(
    const Mpnn64ForwardRequest& request,
    const Mpnn64ForwardOutput& output) noexcept;
}  // namespace detail

universal::Status mpnn64_forward_scalar(
    const Mpnn64ForwardRequest& request,
    const Mpnn64ForwardOutput& output) noexcept {
  if (request.coordinates == nullptr) {
    return invalid("MPNN coordinates pointer is null");
  }
  if (request.atom_sources == nullptr) {
    return invalid("MPNN atom_source pointer is null");
  }
  if (request.weights == nullptr) {
    return invalid("MPNN weights pointer is null");
  }
  if (request.descriptor.layer_count > 0 && request.weights->layers == nullptr) {
    return invalid("MPNN layer weights pointer is null");
  }
  if (request.workspace == nullptr) {
    return invalid("MPNN workspace pointer is null");
  }
  if (output.embeddings == nullptr) {
    return invalid("MPNN output embedding pointer is null");
  }
  if (request.descriptor.hidden_dimension != detail::kMpnn64HiddenDimension ||
      output.hidden_dimension != request.descriptor.hidden_dimension) {
    return invalid("MPNN hidden dimension must be 64");
  }
  if (request.descriptor.neighbor_count == 0 || request.descriptor.rbf_count == 0) {
    return invalid("MPNN neighbor_count and rbf_count must be non-zero");
  }
  const universal::Status weight_status =
      validate_real_edge_weights(request.descriptor, *request.weights);
  if (weight_status.code != universal::StatusCode::Ok) {
    return weight_status;
  }
  if (request.descriptor.message_scale <= 0.0F ||
      !std::isfinite(request.descriptor.message_scale)) {
    return invalid("MPNN message_scale must be finite and positive");
  }
  if (output.residue_count < request.residue_count) {
    return invalid("MPNN output residue capacity is smaller than input");
  }
  if (!workspace_matches(request.descriptor, request.residue_count,
                         *request.workspace)) {
    return failed_precondition("MPNN workspace does not satisfy memory plan");
  }
  if (request.residue_count == 0) {
    return kOk;
  }
  return detail::mpnn64_forward_scalar_unchecked(request, output);
}

}  // namespace hikoboshi::modules

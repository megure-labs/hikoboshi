#ifndef HIKOBOSHI_MODULES_MPNN_DETAIL_MPNN_INNER_INLINE_HPP
#define HIKOBOSHI_MODULES_MPNN_DETAIL_MPNN_INNER_INLINE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/universal/inline.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::modules::mpnn::detail {

// Re-export the types and constants from `hikoboshi::modules::detail` so that
// `detail::Mpnn64Workspace`, `detail::Mpnn64LayerWeights`, etc. inside
// namespace `hikoboshi::modules::mpnn` continue to resolve as before. The
// new `hikoboshi::modules::mpnn::detail` namespace would otherwise shadow
// the outer `hikoboshi::modules::detail` namespace for unqualified-`detail::`
// references made from inside `hikoboshi::modules::mpnn` (e.g. in
// `message_layer.hpp` and `ffn_layer.hpp`).
using hikoboshi::modules::detail::Mpnn64InputWeights;
using hikoboshi::modules::detail::Mpnn64TensorWeights;
using hikoboshi::modules::detail::Mpnn64LinearWeights;
using hikoboshi::modules::detail::Mpnn64NormWeights;
using hikoboshi::modules::detail::Mpnn64EdgeEmbeddingWeights;
using hikoboshi::modules::detail::Mpnn64FeedForwardWeights;
using hikoboshi::modules::detail::Mpnn64LayerWeights;
using hikoboshi::modules::detail::Mpnn64Weights;
using hikoboshi::modules::detail::Mpnn64MemoryPlan;
using hikoboshi::modules::detail::Mpnn64Workspace;

HIKOBOSHI_FORCE_INLINE void fill_zero_inline(float* values,
                                           std::size_t count) noexcept {
  std::fill_n(values, count, 0.0F);
}

HIKOBOSHI_FORCE_INLINE void copy_values_inline(const float* source,
                                             float* target,
                                             std::size_t count) noexcept {
  std::memcpy(target, source, count * sizeof(float));
}

HIKOBOSHI_FORCE_INLINE void gelu_inplace_inline(float* values,
                                              std::size_t count) noexcept {
  hikoboshi::primitives::compute::GeluInplaceScalarRequest gelu{};
  gelu.input = values;
  gelu.count = count;
  gelu.policy = hikoboshi::primitives::compute::GeluPolicy::Exact;
  hikoboshi::primitives::compute::gelu_inplace_scalar(gelu, values);
}

HIKOBOSHI_FORCE_INLINE void axpy_into_inline(float* y,
                                           const float* x,
                                           float alpha,
                                           std::size_t count) noexcept {
  hikoboshi::primitives::compute::AxpyScalarRequest request{};
  request.x = x;
  request.y = y;
  request.alpha = alpha;
  request.count = count;
  hikoboshi::primitives::compute::axpy_scalar(request, y);
}

HIKOBOSHI_FORCE_INLINE float linear_dot_blocked_inline(
    const float* input,
    const float* weight,
    std::size_t input_dimension) noexcept {
  constexpr std::size_t kBlockSize = 8;
  float acc = 0.0F;
  std::size_t p = 0;
  for (; p + kBlockSize <= input_dimension; p += kBlockSize) {
    float block = 0.0F;
    for (std::size_t offset = 0; offset < kBlockSize; ++offset) {
      const std::size_t index = p + offset;
      block += input[index] * weight[index];
    }
    acc += block;
  }
  if (p < input_dimension) {
    float block = 0.0F;
    for (; p < input_dimension; ++p) {
      block += input[p] * weight[p];
    }
    acc += block;
  }
  return acc;
}

HIKOBOSHI_FORCE_INLINE float linear_dot_matched_inline(
    const float* input,
    const float* weight,
    std::size_t input_dimension) noexcept {
  return linear_dot_blocked_inline(input, weight, input_dimension);
}

HIKOBOSHI_FORCE_INLINE void linear_nt_blocked4_inline(
    const float* input,
    const hikoboshi::modules::detail::Mpnn64LinearWeights& weights,
    std::size_t row_count,
    std::size_t output_dimension,
    std::size_t input_dimension,
    float* output) noexcept {
  std::size_t row = 0;
  for (; row + 4 <= row_count; row += 4) {
    const float* row0 = input + (row + 0) * input_dimension;
    const float* row1 = row0 + input_dimension;
    const float* row2 = row1 + input_dimension;
    const float* row3 = row2 + input_dimension;
    std::size_t output_index = 0;
    for (; output_index + 4 <= output_dimension; output_index += 4) {
      const float* weight0 =
          weights.weight.data + output_index * input_dimension;
      const float* weight1 = weight0 + input_dimension;
      const float* weight2 = weight1 + input_dimension;
      const float* weight3 = weight2 + input_dimension;
      float acc00 = 0.0F;
      float acc01 = 0.0F;
      float acc02 = 0.0F;
      float acc03 = 0.0F;
      float acc10 = 0.0F;
      float acc11 = 0.0F;
      float acc12 = 0.0F;
      float acc13 = 0.0F;
      float acc20 = 0.0F;
      float acc21 = 0.0F;
      float acc22 = 0.0F;
      float acc23 = 0.0F;
      float acc30 = 0.0F;
      float acc31 = 0.0F;
      float acc32 = 0.0F;
      float acc33 = 0.0F;
      std::size_t p = 0;
      constexpr std::size_t kBlockSize = 8;
      for (; p + kBlockSize <= input_dimension; p += kBlockSize) {
        float block00 = 0.0F;
        float block01 = 0.0F;
        float block02 = 0.0F;
        float block03 = 0.0F;
        float block10 = 0.0F;
        float block11 = 0.0F;
        float block12 = 0.0F;
        float block13 = 0.0F;
        float block20 = 0.0F;
        float block21 = 0.0F;
        float block22 = 0.0F;
        float block23 = 0.0F;
        float block30 = 0.0F;
        float block31 = 0.0F;
        float block32 = 0.0F;
        float block33 = 0.0F;
        for (std::size_t offset = 0; offset < kBlockSize; ++offset) {
          const std::size_t index = p + offset;
          const float v0 = row0[index];
          const float v1 = row1[index];
          const float v2 = row2[index];
          const float v3 = row3[index];
          const float w0 = weight0[index];
          const float w1 = weight1[index];
          const float w2 = weight2[index];
          const float w3 = weight3[index];
          block00 += v0 * w0;
          block01 += v0 * w1;
          block02 += v0 * w2;
          block03 += v0 * w3;
          block10 += v1 * w0;
          block11 += v1 * w1;
          block12 += v1 * w2;
          block13 += v1 * w3;
          block20 += v2 * w0;
          block21 += v2 * w1;
          block22 += v2 * w2;
          block23 += v2 * w3;
          block30 += v3 * w0;
          block31 += v3 * w1;
          block32 += v3 * w2;
          block33 += v3 * w3;
        }
        acc00 += block00;
        acc01 += block01;
        acc02 += block02;
        acc03 += block03;
        acc10 += block10;
        acc11 += block11;
        acc12 += block12;
        acc13 += block13;
        acc20 += block20;
        acc21 += block21;
        acc22 += block22;
        acc23 += block23;
        acc30 += block30;
        acc31 += block31;
        acc32 += block32;
        acc33 += block33;
      }
      if (p < input_dimension) {
        for (; p < input_dimension; ++p) {
          const float v0 = row0[p];
          const float v1 = row1[p];
          const float v2 = row2[p];
          const float v3 = row3[p];
          const float w0 = weight0[p];
          const float w1 = weight1[p];
          const float w2 = weight2[p];
          const float w3 = weight3[p];
          acc00 += v0 * w0;
          acc01 += v0 * w1;
          acc02 += v0 * w2;
          acc03 += v0 * w3;
          acc10 += v1 * w0;
          acc11 += v1 * w1;
          acc12 += v1 * w2;
          acc13 += v1 * w3;
          acc20 += v2 * w0;
          acc21 += v2 * w1;
          acc22 += v2 * w2;
          acc23 += v2 * w3;
          acc30 += v3 * w0;
          acc31 += v3 * w1;
          acc32 += v3 * w2;
          acc33 += v3 * w3;
        }
      }
      if (weights.bias.data != nullptr && weights.bias.size != 0) {
        const float bias0 = weights.bias.data[output_index + 0];
        const float bias1 = weights.bias.data[output_index + 1];
        const float bias2 = weights.bias.data[output_index + 2];
        const float bias3 = weights.bias.data[output_index + 3];
        acc00 += bias0;
        acc01 += bias1;
        acc02 += bias2;
        acc03 += bias3;
        acc10 += bias0;
        acc11 += bias1;
        acc12 += bias2;
        acc13 += bias3;
        acc20 += bias0;
        acc21 += bias1;
        acc22 += bias2;
        acc23 += bias3;
        acc30 += bias0;
        acc31 += bias1;
        acc32 += bias2;
        acc33 += bias3;
      }
      float* out0 = output + (row + 0) * output_dimension + output_index;
      float* out1 = out0 + output_dimension;
      float* out2 = out1 + output_dimension;
      float* out3 = out2 + output_dimension;
      out0[0] = acc00;
      out0[1] = acc01;
      out0[2] = acc02;
      out0[3] = acc03;
      out1[0] = acc10;
      out1[1] = acc11;
      out1[2] = acc12;
      out1[3] = acc13;
      out2[0] = acc20;
      out2[1] = acc21;
      out2[2] = acc22;
      out2[3] = acc23;
      out3[0] = acc30;
      out3[1] = acc31;
      out3[2] = acc32;
      out3[3] = acc33;
    }
    for (; output_index < output_dimension; ++output_index) {
      const float* weight_row =
          weights.weight.data + output_index * input_dimension;
      float acc0 =
          linear_dot_blocked_inline(row0, weight_row, input_dimension);
      float acc1 =
          linear_dot_blocked_inline(row1, weight_row, input_dimension);
      float acc2 =
          linear_dot_blocked_inline(row2, weight_row, input_dimension);
      float acc3 =
          linear_dot_blocked_inline(row3, weight_row, input_dimension);
      if (weights.bias.data != nullptr && weights.bias.size != 0) {
        const float bias = weights.bias.data[output_index];
        acc0 += bias;
        acc1 += bias;
        acc2 += bias;
        acc3 += bias;
      }
      output[(row + 0) * output_dimension + output_index] = acc0;
      output[(row + 1) * output_dimension + output_index] = acc1;
      output[(row + 2) * output_dimension + output_index] = acc2;
      output[(row + 3) * output_dimension + output_index] = acc3;
    }
  }
  for (; row < row_count; ++row) {
    const float* input_row = input + row * input_dimension;
    float* output_row = output + row * output_dimension;
    for (std::size_t output_index = 0; output_index < output_dimension;
         ++output_index) {
      const float* weight_row =
          weights.weight.data + output_index * input_dimension;
      float acc =
          linear_dot_blocked_inline(input_row, weight_row, input_dimension);
      if (weights.bias.data != nullptr && weights.bias.size != 0) {
        acc += weights.bias.data[output_index];
      }
      output_row[output_index] = acc;
    }
  }
}

HIKOBOSHI_FORCE_INLINE void linear_nt_inline(
    const float* input,
    const hikoboshi::modules::detail::Mpnn64LinearWeights& weights,
    std::size_t row_count,
    std::size_t output_dimension,
    std::size_t input_dimension,
    float* output) noexcept {
  linear_nt_blocked4_inline(input, weights, row_count, output_dimension,
                            input_dimension, output);
}

HIKOBOSHI_FORCE_INLINE void linear_row_nt_inline(
    const float* input,
    const hikoboshi::modules::detail::Mpnn64LinearWeights& weights,
    std::size_t output_dimension,
    std::size_t input_dimension,
    float* output) noexcept {
  std::size_t output_index = 0;
  for (; output_index + 4 <= output_dimension; output_index += 4) {
    const float* weight0 =
        weights.weight.data + output_index * input_dimension;
    const float* weight1 = weight0 + input_dimension;
    const float* weight2 = weight1 + input_dimension;
    const float* weight3 = weight2 + input_dimension;
    float acc0 =
        linear_dot_matched_inline(input, weight0, input_dimension);
    float acc1 =
        linear_dot_matched_inline(input, weight1, input_dimension);
    float acc2 =
        linear_dot_matched_inline(input, weight2, input_dimension);
    float acc3 =
        linear_dot_matched_inline(input, weight3, input_dimension);
    if (weights.bias.data != nullptr && weights.bias.size != 0) {
      acc0 += weights.bias.data[output_index + 0];
      acc1 += weights.bias.data[output_index + 1];
      acc2 += weights.bias.data[output_index + 2];
      acc3 += weights.bias.data[output_index + 3];
    }
    output[output_index + 0] = acc0;
    output[output_index + 1] = acc1;
    output[output_index + 2] = acc2;
    output[output_index + 3] = acc3;
  }
  for (; output_index < output_dimension; ++output_index) {
    const float* weight_row =
        weights.weight.data + output_index * input_dimension;
    float acc = linear_dot_matched_inline(input, weight_row, input_dimension);
    if (weights.bias.data != nullptr && weights.bias.size != 0) {
      acc += weights.bias.data[output_index];
    }
    output[output_index] = acc;
  }
}

HIKOBOSHI_FORCE_INLINE void layer_norm_rows_inline(
    const float* input,
    const hikoboshi::modules::detail::Mpnn64NormWeights& weights,
    std::size_t row_count,
    std::size_t dimension,
    float* output) noexcept {
  hikoboshi::primitives::compute::LayerNormScalarRequest norm{};
  norm.input = input;
  norm.gamma = weights.weight.data;
  norm.beta = weights.bias.data;
  norm.row_count = row_count;
  norm.row_dimension = dimension;
  norm.epsilon = hikoboshi::modules::detail::kMpnn64LayerNormEpsilon;
  hikoboshi::dispatch::layer_norm_forward(hikoboshi::dispatch::ScalarTag{}, norm,
                                        output);
}

HIKOBOSHI_FORCE_INLINE void layer_norm_residual_rows_inline(
    float* input,
    const float* residual,
    const hikoboshi::modules::detail::Mpnn64NormWeights& weights,
    std::size_t row_count,
    std::size_t dimension,
    float* output) noexcept {
  axpy_into_inline(input, residual, 1.0F, row_count * dimension);
  layer_norm_rows_inline(input, weights, row_count, dimension, output);
}

HIKOBOSHI_FORCE_INLINE void layer_norm_residual_row_inline(
    float* input,
    const float* residual,
    const hikoboshi::modules::detail::Mpnn64NormWeights& weights,
    std::size_t dimension,
    float* output) noexcept {
  axpy_into_inline(input, residual, 1.0F, dimension);

  float mean = 0.0F;
  float m2 = 0.0F;
  for (std::size_t d = 0; d < dimension; ++d) {
    const float x = input[d];
    const float count = static_cast<float>(d + 1);
    const float delta = x - mean;
    mean += delta / count;
    const float delta2 = x - mean;
    m2 += delta * delta2;
  }
  const float variance = m2 / static_cast<float>(dimension);
  const float inv_std =
      1.0F /
      std::sqrt(variance + hikoboshi::modules::detail::kMpnn64LayerNormEpsilon);

  for (std::size_t d = 0; d < dimension; ++d) {
    const float normalized = (input[d] - mean) * inv_std;
    const float scaled = weights.weight.data != nullptr
                             ? normalized * weights.weight.data[d]
                             : normalized;
    output[d] = weights.bias.data != nullptr ? scaled + weights.bias.data[d]
                                             : scaled;
  }
}

HIKOBOSHI_FORCE_INLINE bool valid_neighbor_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request,
    std::int32_t neighbor) noexcept {
  return neighbor >= 0 &&
         static_cast<std::size_t>(neighbor) < request.residue_count;
}

HIKOBOSHI_FORCE_INLINE std::size_t active_neighbor_count_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request) noexcept {
  return std::min(request.descriptor.neighbor_count, request.residue_count);
}

HIKOBOSHI_FORCE_INLINE std::size_t atom_index_inline(
    hikoboshi::universal::CanonicalAtom atom) noexcept {
  return static_cast<std::size_t>(atom);
}

HIKOBOSHI_FORCE_INLINE std::size_t atom_source_offset_inline(
    std::size_t residue,
    std::size_t atom) noexcept {
  return residue * hikoboshi::modules::detail::kMpnn64AtomCount + atom;
}

HIKOBOSHI_FORCE_INLINE bool atom_missing_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request,
    std::size_t residue,
    std::size_t atom) noexcept {
  return request.atom_sources[atom_source_offset_inline(residue, atom)] ==
         hikoboshi::universal::AtomSource::Missing;
}

HIKOBOSHI_FORCE_INLINE bool residue_mask_valid_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request,
    std::size_t residue) noexcept {
  return !atom_missing_inline(
      request, residue,
      atom_index_inline(hikoboshi::universal::CanonicalAtom::CA));
}

HIKOBOSHI_FORCE_INLINE void apply_residue_mask_gate_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request) noexcept {
  const std::size_t hidden = request.descriptor.hidden_dimension;
  float* residue_state = request.workspace->residue_state.data;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    if (!residue_mask_valid_inline(request, residue)) {
      fill_zero_inline(residue_state + residue * hidden, hidden);
    }
  }
}

HIKOBOSHI_FORCE_INLINE void build_edge_update_inputs_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request) noexcept {
  const hikoboshi::modules::Mpnn64Descriptor& descriptor = request.descriptor;
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t input_dim =
      hikoboshi::modules::detail::kMpnn64MessageInputDimension;
  const std::size_t neighbor_count = active_neighbor_count_inline(request);

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    const float* query_state = workspace.residue_state.data + residue * hidden;
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot =
          residue * descriptor.neighbor_count + neighbor_slot;
      float* input = workspace.residue_features.data + slot * input_dim;
      const std::int32_t neighbor = workspace.neighbor_indices.data[slot];
      if (!valid_neighbor_inline(request, neighbor)) {
        fill_zero_inline(input, input_dim);
        continue;
      }
      copy_values_inline(query_state, input, hidden);
      copy_values_inline(workspace.edge_state.data + slot * hidden,
                         input + hidden, hidden);
      copy_values_inline(workspace.residue_state.data +
                             static_cast<std::size_t>(neighbor) * hidden,
                         input + 2 * hidden, hidden);
    }
  }
}

HIKOBOSHI_FORCE_INLINE void apply_edge_update_active_neighbors_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request,
    const hikoboshi::modules::detail::Mpnn64LayerWeights& layer,
    std::size_t neighbor_count) noexcept {
  const hikoboshi::modules::Mpnn64Descriptor& descriptor = request.descriptor;
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t input_dim =
      hikoboshi::modules::detail::kMpnn64MessageInputDimension;

  build_edge_update_inputs_inline(request);
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0; neighbor_slot < neighbor_count;
         ++neighbor_slot) {
      const std::size_t slot =
          residue * descriptor.neighbor_count + neighbor_slot;
      if (!valid_neighbor_inline(request,
                                 workspace.neighbor_indices.data[slot])) {
        continue;
      }

      const float* input = workspace.residue_features.data + slot * input_dim;
      float* message = workspace.message_state.data + slot * hidden;
      float* projected = workspace.projected_message_state.data + slot * hidden;

      linear_row_nt_inline(input, layer.W11, hidden, input_dim, message);
      gelu_inplace_inline(message, hidden);
      linear_row_nt_inline(message, layer.W12, hidden, hidden, projected);
      gelu_inplace_inline(projected, hidden);
      linear_row_nt_inline(projected, layer.W13, hidden, hidden, message);

      const float* edge = workspace.edge_state.data + slot * hidden;
      layer_norm_residual_row_inline(
          message, edge, layer.norm3, hidden,
          workspace.edge_state.data + slot * hidden);
    }
  }
}

HIKOBOSHI_FORCE_INLINE void apply_edge_update_inline(
    const hikoboshi::modules::Mpnn64ForwardRequest& request,
    const hikoboshi::modules::detail::Mpnn64LayerWeights& layer) noexcept {
  const hikoboshi::modules::Mpnn64Descriptor& descriptor = request.descriptor;
  const hikoboshi::modules::detail::Mpnn64MemoryPlan plan{
      request.residue_count, descriptor.hidden_dimension,
      descriptor.neighbor_count, descriptor.rbf_count, descriptor.layer_count};
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t slot_count =
      hikoboshi::modules::detail::mpnn64_neighbor_slot_count(plan);
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t input_dim =
      hikoboshi::modules::detail::kMpnn64MessageInputDimension;
  const std::size_t neighbor_count = active_neighbor_count_inline(request);

  if (neighbor_count < descriptor.neighbor_count) {
    apply_edge_update_active_neighbors_inline(request, layer, neighbor_count);
    return;
  }

  build_edge_update_inputs_inline(request);
  const std::size_t hidden_count = slot_count * hidden;
  linear_nt_inline(workspace.residue_features.data, layer.W11, slot_count,
                   hidden, input_dim, workspace.message_state.data);
  gelu_inplace_inline(workspace.message_state.data, hidden_count);
  linear_nt_inline(workspace.message_state.data, layer.W12, slot_count, hidden,
                   hidden, workspace.projected_message_state.data);
  gelu_inplace_inline(workspace.projected_message_state.data, hidden_count);
  linear_nt_inline(workspace.projected_message_state.data, layer.W13,
                   slot_count, hidden, hidden, workspace.message_state.data);

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < descriptor.neighbor_count; ++neighbor_slot) {
      const std::size_t slot =
          residue * descriptor.neighbor_count + neighbor_slot;
      if (!valid_neighbor_inline(request,
                                 workspace.neighbor_indices.data[slot])) {
        continue;
      }
      float* update = workspace.message_state.data + slot * hidden;
      const float* edge = workspace.edge_state.data + slot * hidden;
      layer_norm_residual_row_inline(
          update, edge, layer.norm3, hidden,
          workspace.edge_state.data + slot * hidden);
    }
  }
}

}  // namespace hikoboshi::modules::mpnn::detail

#endif  // HIKOBOSHI_MODULES_MPNN_DETAIL_MPNN_INNER_INLINE_HPP

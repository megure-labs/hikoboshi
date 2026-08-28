#ifndef HIKOBOSHI_MODULES_MPNN_DETAIL_FFN_LAYER_INLINE_HPP
#define HIKOBOSHI_MODULES_MPNN_DETAIL_FFN_LAYER_INLINE_HPP

#include <cstddef>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp>
#include <hikoboshi/modules/mpnn/ffn_layer.hpp>
#include <hikoboshi/universal/inline.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::modules::mpnn::detail {

HIKOBOSHI_FORCE_INLINE void ffn_add_bias_inline(
    float* rows,
    hikoboshi::universal::Span<const float> bias,
    std::size_t row_count,
    std::size_t dimension) noexcept {
  if (bias.data == nullptr || bias.size == 0) {
    return;
  }
  hikoboshi::primitives::compute::BiasAddScalarRequest request{};
  request.input = rows;
  request.bias = bias.data;
  request.row_count = row_count;
  request.row_dimension = dimension;
  hikoboshi::primitives::compute::bias_add_scalar(request, rows);
}

HIKOBOSHI_FORCE_INLINE void ffn_linear_nt_inline(
    const float* input,
    const hikoboshi::modules::detail::Mpnn64LinearWeights& weights,
    std::size_t row_count,
    std::size_t output_dimension,
    std::size_t input_dimension,
    float* output) noexcept {
  hikoboshi::primitives::linalg::GemmScalarRequest projection{};
  projection.lhs = input;
  projection.rhs = weights.weight.data;
  projection.m = row_count;
  projection.n = output_dimension;
  projection.k = input_dimension;
  hikoboshi::dispatch::gemm_nt_forward(hikoboshi::dispatch::ScalarTag{}, projection,
                                     output);
  ffn_add_bias_inline(output, weights.bias, row_count, output_dimension);
}

HIKOBOSHI_FORCE_INLINE void mpnn_ffn_layer_scalar_inline(
    const MpnnFfnLayerRequest& request,
    const MpnnFfnLayerOutput& output) noexcept {
  const hikoboshi::modules::detail::Mpnn64LayerWeights& layer = *request.weights;
  hikoboshi::modules::detail::Mpnn64Workspace& workspace = *request.workspace;
  const std::size_t hidden = request.hidden_dimension;
  const std::size_t ffn_hidden = request.ffn_hidden_dimension;

  ffn_linear_nt_inline(request.input_embeddings, layer.ffn.W_in,
                       request.residue_count, ffn_hidden, hidden,
                       workspace.ffn_hidden.data);
  gelu_inplace_inline(workspace.ffn_hidden.data,
                      request.residue_count * ffn_hidden);
  ffn_linear_nt_inline(workspace.ffn_hidden.data, layer.ffn.W_out,
                       request.residue_count, hidden, ffn_hidden,
                       workspace.residue_scratch.data);

  layer_norm_residual_rows_inline(workspace.residue_scratch.data,
                                  request.input_embeddings, layer.norm2,
                                  request.residue_count, hidden,
                                  output.output_embeddings);
}

}  // namespace hikoboshi::modules::mpnn::detail

#endif  // HIKOBOSHI_MODULES_MPNN_DETAIL_FFN_LAYER_INLINE_HPP

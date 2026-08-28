#ifndef HIKOBOSHI_MODULES_MPNN_FFN_LAYER_HPP
#define HIKOBOSHI_MODULES_MPNN_FFN_LAYER_HPP

#include <cstddef>

#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>

namespace hikoboshi::modules::mpnn {

struct MpnnFfnLayerRequest {
  const float* input_embeddings;  // row-major [L, H]
  const detail::Mpnn64LayerWeights* weights;
  detail::Mpnn64Workspace* workspace;
  std::size_t residue_count;
  std::size_t hidden_dimension;
  std::size_t ffn_hidden_dimension;
};

struct MpnnFfnLayerOutput {
  float* output_embeddings;  // row-major [L, H]
};

void mpnn_ffn_layer_scalar(const MpnnFfnLayerRequest& request,
                           const MpnnFfnLayerOutput& output) noexcept;

}  // namespace hikoboshi::modules::mpnn

#endif  // HIKOBOSHI_MODULES_MPNN_FFN_LAYER_HPP

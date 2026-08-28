#include <hikoboshi/modules/mpnn/ffn_layer.hpp>

#include <hikoboshi/modules/mpnn/detail/ffn_layer_inline.hpp>

namespace hikoboshi::modules::mpnn {

void mpnn_ffn_layer_scalar(const MpnnFfnLayerRequest& request,
                           const MpnnFfnLayerOutput& output) noexcept {
  detail::mpnn_ffn_layer_scalar_inline(request, output);
}

}  // namespace hikoboshi::modules::mpnn

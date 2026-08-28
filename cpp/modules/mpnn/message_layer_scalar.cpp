#include <hikoboshi/modules/mpnn/message_layer.hpp>

#include <hikoboshi/modules/mpnn/detail/message_layer_inline.hpp>

namespace hikoboshi::modules::mpnn {

void message_input_pack_scalar(const MessageInputPackRequest& request) noexcept {
  detail::message_input_pack_scalar_inline(request);
}

void mpnn_message_layer_scalar(const MpnnMessageLayerRequest& request,
                               const MpnnMessageLayerOutput& output) noexcept {
  detail::mpnn_message_layer_scalar_inline(request, output);
}

}  // namespace hikoboshi::modules::mpnn

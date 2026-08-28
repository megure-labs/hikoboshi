#include <hikoboshi/modules/ffn/ffn_layer.hpp>

#include <hikoboshi/dispatch/backend_tag.hpp>
#include <hikoboshi/modules/ffn/detail/ffn_layer_inline.hpp>

namespace hikoboshi::modules::ffn {

void ffn_layer_scalar_gelu_nonorm_noresidual_bias_fast(
    const FfnLayerRequest& request,
    const FfnLayerOutput& output) noexcept {
  detail::ffn_layer_inline<GeluTag, NoNormTag, NoResidualTag,
                           /*HasBias=*/true,
                           hikoboshi::dispatch::FastParityTag,
                           hikoboshi::dispatch::ScalarTag>(request, output);
}

void ffn_layer_scalar_gelu_nonorm_noresidual_bias_strict(
    const FfnLayerRequest& request,
    const FfnLayerOutput& output) noexcept {
  detail::ffn_layer_inline<GeluTag, NoNormTag, NoResidualTag,
                           /*HasBias=*/true,
                           hikoboshi::dispatch::StrictParityTag,
                           hikoboshi::dispatch::ScalarTag>(request, output);
}

}  // namespace hikoboshi::modules::ffn

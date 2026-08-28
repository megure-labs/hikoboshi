#include <hikoboshi/modules/transformer/attention.hpp>

#include <hikoboshi/modules/transformer/detail/attention_inline.hpp>

namespace hikoboshi::modules::transformer {

void attention_layer_scalar(const AttentionLayerRequest& request,
                            const AttentionLayerOutput& output) noexcept {
  detail::attention_layer_scalar_inline(request, output);
}

}  // namespace hikoboshi::modules::transformer

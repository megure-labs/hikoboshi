#ifndef HIKOBOSHI_MODULES_FFN_DETAIL_FFN_LAYER_INLINE_HPP
#define HIKOBOSHI_MODULES_FFN_DETAIL_FFN_LAYER_INLINE_HPP

/// @file
/// Header-inline body of the FFN template family.
///
/// `ffn_layer_inline<...>` composes the registered scalar primitives into one
/// position-wise feed-forward block. The header-inline form follows the
/// `modules-header-inline` pattern used by transformer attention: the public
/// `.cpp` wrapper exists only to emit a non-inline symbol for ABI consumers
/// and the dispatch registry, while every other call site (notably the
/// future ESM2-8M forward pass) includes this header directly so the
/// optimizer can fold the body into the surrounding architecture loop and
/// dead-code-eliminate the unused tag branches.

#include <cstddef>
#include <type_traits>

#include <hikoboshi/dispatch/backend_tag.hpp>
#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/common/weights_views.hpp>
#include <hikoboshi/modules/ffn/detail/workspace.hpp>
#include <hikoboshi/modules/ffn/ffn_layer.hpp>
#include <hikoboshi/universal/inline.hpp>

namespace hikoboshi::modules::ffn::detail {

namespace hiko_p = hikoboshi::primitives;
namespace hiko_d = hikoboshi::dispatch;

/// Apply one `y = x @ W^T + b` linear projection plus an optional bias add.
/// `HasBias = false` skips the bias step entirely; `HasBias = true` reads
/// `linear.bias` unconditionally, so the caller must provide a non-null
/// pointer in that specialization.
template <bool HasBias, typename ParityTag>
HIKOBOSHI_FORCE_INLINE void apply_linear_nt_with_optional_bias_inline(
    const float* input, const common::LinearLayerWeightsView& linear,
    std::size_t row_count, float* output) noexcept {
  hiko_p::linalg::GemmScalarRequest projection{};
  projection.lhs = input;
  projection.rhs = linear.weight;
  projection.m = row_count;
  projection.n = linear.output_dim;
  projection.k = linear.input_dim;
  hiko_d::gemm_nt_forward(hiko_d::ScalarTag{}, ParityTag{}, projection, output);

  if constexpr (HasBias) {
    hiko_p::compute::BiasAddScalarRequest bias{};
    bias.input = output;
    bias.bias = linear.bias;
    bias.row_count = row_count;
    bias.row_dimension = linear.output_dim;
    hiko_d::bias_add_forward(hiko_d::ScalarTag{}, bias, output);
  }
}

/// Apply the activation step in place. Specialized by `ActivationTag`;
/// Hikoboshi 0.1.0 ships `GeluTag` only.
template <typename ActivationTag>
HIKOBOSHI_FORCE_INLINE void apply_activation_inplace_inline(
    float* values, std::size_t count) noexcept;

template <>
HIKOBOSHI_FORCE_INLINE void apply_activation_inplace_inline<GeluTag>(
    float* values, std::size_t count) noexcept {
  hiko_p::compute::GeluInplaceScalarRequest request{};
  request.input = values;
  request.count = count;
  request.policy = hiko_p::compute::GeluPolicy::Exact;
  hiko_d::gelu_inplace_forward(hiko_d::ScalarTag{}, request, values);
}

/// Templated FFN body. The closed-tag-axis specialization pattern lets the
/// host compiler dead-code-eliminate every branch the architecture does not
/// use, and constant-propagate the activation choice into the elementwise
/// loop. The runtime numeric dimensions (`rows`, `hidden_dim`,
/// `intermediate_dim`) stay parameters of the request struct.
///
/// Hikoboshi 0.1.0 instantiates only the tuple
/// `(GeluTag, NoNormTag, NoResidualTag, HasBias=true, ParityTag, ScalarTag)`
/// from the ABI wrapper in `cpp/modules/ffn/ffn_layer_scalar.cpp`. The
/// template is well-typed for every closed tag value the type system
/// currently declares, so future architectures can add new ABI wrappers
/// without re-opening the template body.
template <typename ActivationTag,
          typename NormPlacementTag,
          typename ResidualTag,
          bool HasBias,
          typename ParityTag,
          typename BackendTag>
HIKOBOSHI_FORCE_INLINE void ffn_layer_inline(
    const FfnLayerRequest& request,
    const FfnLayerOutput& output) noexcept {
  static_assert(std::is_same_v<BackendTag, hiko_d::ScalarTag>,
                "ffn_layer_inline only supports the scalar backend in 0.1.0");
  static_assert(std::is_same_v<NormPlacementTag, NoNormTag>,
                "ffn_layer_inline only supports NoNormTag in 0.1.0; "
                "architectures wrap norm steps externally");
  static_assert(std::is_same_v<ResidualTag, NoResidualTag>,
                "ffn_layer_inline only supports NoResidualTag in 0.1.0; "
                "architectures wrap residual adds externally");

  FfnLayerWorkspace& workspace = *request.workspace;
  const std::size_t rows = request.rows;

  apply_linear_nt_with_optional_bias_inline<HasBias, ParityTag>(
      request.input_embeddings, request.w_in, rows,
      workspace.intermediate_buffer);

  apply_activation_inplace_inline<ActivationTag>(
      workspace.intermediate_buffer, rows * request.intermediate_dim);

  apply_linear_nt_with_optional_bias_inline<HasBias, ParityTag>(
      workspace.intermediate_buffer, request.w_out, rows,
      output.output_embeddings);
}

}  // namespace hikoboshi::modules::ffn::detail

#endif  // HIKOBOSHI_MODULES_FFN_DETAIL_FFN_LAYER_INLINE_HPP

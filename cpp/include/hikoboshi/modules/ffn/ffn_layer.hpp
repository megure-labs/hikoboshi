#ifndef HIKOBOSHI_MODULES_FFN_FFN_LAYER_HPP
#define HIKOBOSHI_MODULES_FFN_FFN_LAYER_HPP

/// @file
/// CUTLASS-style FFN template family.
///
/// `hikoboshi.ffn_layer.v1` is a position-wise feed-forward block whose closed
/// tag axes (activation, norm placement, residual, has-bias, parity, backend)
/// become C++ template parameters and whose unbounded numeric dimensions
/// (`rows`, `hidden_dim`, `intermediate_dim`) remain runtime arguments. The
/// composition itself — `gemm_nt → (bias_add) → activation → gemm_nt →
/// (bias_add)` — is identical across every transformer-family architecture
/// the closed-op DSL plans to support; the variation lives entirely in the
/// closed tag axes. Architectures that need norm wrapping or residual
/// addition compose those steps externally; the FFN body for the
/// `NoNormTag` / `NoResidualTag` tuple is the bare two-linear-with-activation
/// sequence.
///
/// Hikoboshi 0.1.0 instantiates exactly one tag tuple
/// (`GeluTag, NoNormTag, NoResidualTag, HasBias=true, ParityTag, ScalarTag`)
/// in two parity flavors (`StrictParityTag`, `FastParityTag`). Other tag
/// combinations compile cleanly but are not invoked from any shipped
/// architecture; they exist as the closed specialization surface for future
/// architectures (SiLU/SwiGLU activations, PreNorm/PostNorm wrapping,
/// in-module residual addition, bias-less linears, …) without forcing the
/// closed op set to grow another module-op id.
///
/// MPNN-64 retains its architecture-specific `hikoboshi.mpnn.ffn_layer.v1`
/// because that compound applies a fused `layer_norm_residual_rows` primitive
/// rather than the bare two-linear sequence; the two FFN compounds coexist
/// in the module-op registry.

#include <cstddef>

#include <hikoboshi/modules/common/weights_views.hpp>
#include <hikoboshi/modules/ffn/detail/workspace.hpp>

namespace hikoboshi::modules::ffn {

/// Activation-choice tag axis. Only `GeluTag` is shipped in 0.1.0 because
/// ESM2-8M and the MPNN-style architectures both use exact-erf GELU; SiLU,
/// ReLU, and SwiGLU tags will be added when a future architecture needs
/// them.
struct GeluTag {};

/// Norm-placement tag axis. `NoNormTag` means the architecture wraps the
/// FFN externally (pre- or post-norm is layered around the call site).
/// Pre- and post-norm-in-module tags are reserved for future architectures
/// that want the inline body to handle the norm step.
struct NoNormTag {};

/// Residual-add tag axis. `NoResidualTag` means the architecture sums the
/// residual externally. `WithResidualTag` is reserved for future
/// architectures that want the inline body to fold the residual add into
/// the FFN call.
struct NoResidualTag {};

/// Request payload for one FFN block.
///
/// All tensors are row-major. `input_embeddings` is `[rows, hidden_dim]`.
/// `w_in` projects to `[rows, intermediate_dim]` (so its
/// `output_dim == intermediate_dim` and `input_dim == hidden_dim`); `w_out`
/// projects back to `[rows, hidden_dim]` (so its
/// `output_dim == hidden_dim` and `input_dim == intermediate_dim`). The
/// bias pointers on each `LinearLayerWeightsView` may be `nullptr` when
/// `HasBias = false`; the inline body never reads them in that case.
///
/// `workspace` is caller-owned and must satisfy
/// `workspace->intermediate_capacity >= rows * intermediate_dim`.
struct FfnLayerRequest {
  const float* input_embeddings;
  std::size_t rows;
  std::size_t hidden_dim;
  std::size_t intermediate_dim;
  common::LinearLayerWeightsView w_in;
  common::LinearLayerWeightsView w_out;
  detail::FfnLayerWorkspace* workspace;
};

/// Output payload for one FFN block.
///
/// `output_embeddings` is `[rows, hidden_dim]` row-major. The buffer is
/// caller-owned. The implementation writes directly to it; aliasing with
/// `input_embeddings` is permitted because the body fully consumes the
/// input before writing the first output element.
struct FfnLayerOutput {
  float* output_embeddings;
};

/// ABI wrapper for the
/// `(GeluTag, NoNormTag, NoResidualTag, HasBias=true, FastParityTag,
/// ScalarTag)` specialization. The dispatch entry for the registered
/// `hikoboshi.ffn_layer.v1` module-op points at this symbol; callers that
/// want explicit parity control may instead call
/// `ffn_layer_scalar_gelu_nonorm_noresidual_bias_strict` for the strict
/// parity counterpart.
void ffn_layer_scalar_gelu_nonorm_noresidual_bias_fast(
    const FfnLayerRequest& request,
    const FfnLayerOutput& output) noexcept;

/// Strict-parity ABI wrapper for the
/// `(GeluTag, NoNormTag, NoResidualTag, HasBias=true, StrictParityTag,
/// ScalarTag)` specialization. Available through this named symbol for
/// parity-golden tests; the registry's `dispatch_entry` points at the
/// fast-parity wrapper.
void ffn_layer_scalar_gelu_nonorm_noresidual_bias_strict(
    const FfnLayerRequest& request,
    const FfnLayerOutput& output) noexcept;

}  // namespace hikoboshi::modules::ffn

#endif  // HIKOBOSHI_MODULES_FFN_FFN_LAYER_HPP

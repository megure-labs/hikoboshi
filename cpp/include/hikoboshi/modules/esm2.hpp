#ifndef HIKOBOSHI_MODULES_ESM2_HPP
#define HIKOBOSHI_MODULES_ESM2_HPP

/// @file
/// ESM2-8M forward-pass entry point.
///
/// `hikoboshi.esm2.v1.encoder` is a registered compound module that
/// composes the existing transformer attention compound module, the FFN
/// template family, and the scalar LayerNorm primitive into the full
/// ESM2-8M encoder forward pass. The encoder produces `[seq_len, 320]`
/// per-residue embeddings consumed by `RawDotV1` scoring and hard-SW
/// alignment.
///
/// The request/output structs are architecture-agnostic in shape (they
/// carry a `Esm2Descriptor` to drive the runtime dimensions) so a future
/// ESM3 or larger ESM2 variant can reuse the same forward template by
/// registering a fresh module-op record that points at the same scalar
/// entry function. Hikoboshi 0.1.0 ships only the ESM2-8M tuple.

#include <cstddef>
#include <cstdint>

#include <hikoboshi/modules/detail/esm2_layers.hpp>
#include <hikoboshi/modules/detail/esm2_workspace.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::modules {

using Esm2Descriptor = hikoboshi::modules::detail::Esm2Descriptor;

/// Request payload for one ESM2-8M encoder forward call.
///
/// `token_ids` is a row-major `[seq_len]` buffer of token ids referencing
/// the embedding-table rows. The forward pass does not own the token
/// buffer; the engine layer's tokenizer step writes it before invoking
/// the encoder. `weights` and `workspace` are caller-owned and must
/// outlive the call.
struct Esm2ForwardRequest {
  const std::int32_t* token_ids;
  std::size_t seq_len;
  Esm2Descriptor descriptor;
  const detail::Esm2Weights* weights;
  detail::Esm2Workspace* workspace;
};

/// Output payload for one ESM2-8M encoder forward call.
///
/// `embeddings` is row-major `[seq_len, hidden_dimension]`. The buffer is
/// caller-owned. `seq_len` and `hidden_dimension` echo the request shape
/// so downstream consumers can confirm coherence without re-reading the
/// descriptor.
struct Esm2ForwardOutput {
  float* embeddings;
  std::size_t seq_len;
  std::size_t hidden_dimension;
};

/// Scalar-backend ESM2-8M encoder forward pass.
///
/// Composes the registered attention and FFN compound modules plus the
/// scalar LayerNorm primitive. Returns `Ok` on success or a descriptive
/// `InvalidArgument` / `FailedPrecondition` status describing the first
/// validation failure. The function is `noexcept` so the dispatch
/// registry can take a stable function-pointer address.
hikoboshi::universal::Status esm2_forward_scalar(
    const Esm2ForwardRequest& request,
    const Esm2ForwardOutput& output) noexcept;

}  // namespace hikoboshi::modules

#endif  // HIKOBOSHI_MODULES_ESM2_HPP

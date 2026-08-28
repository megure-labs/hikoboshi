#ifndef HIKOBOSHI_ALGORITHMS_PAIRWISE_HPP
#define HIKOBOSHI_ALGORITHMS_PAIRWISE_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/algorithms/detail/pairwise_workspace.hpp>
#include <hikoboshi/algorithms/metrics.hpp>
#include <hikoboshi/modules/detail/esm2_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/esm2.hpp>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>
#include <hikoboshi/universal/weights.hpp>

namespace hikoboshi::algorithms {

inline constexpr float kPairwiseDefaultGapOpen = -1.4F;
inline constexpr float kPairwiseDefaultGapExtension = -0.15F;
inline constexpr float kPairwiseDefaultSoftTemperature = 1.0F;
inline constexpr float kPairwiseSoftPosteriorThreshold = 0.5F;

struct PairwiseOptions {
  float gap_open = kPairwiseDefaultGapOpen;
  float gap_extension = kPairwiseDefaultGapExtension;
  float soft_gap_open = kPairwiseDefaultGapOpen;
  float soft_gap_extension = kPairwiseDefaultGapExtension;
};

struct PairwiseEmbeddingRequest {
  hikoboshi::universal::EmbeddingView query_embedding{};
  hikoboshi::universal::EmbeddingView target_embedding{};
  hikoboshi::universal::StructureView query_structure{};
  hikoboshi::universal::StructureView target_structure{};
  PairwiseOptions options{};
  // Branch selectors after API AlignmentMode lowering. When `soft_mode` is
  // false, hard local affine SW runs by default even if `hard_mode` is false
  // so existing algorithm callers preserve hard-SW behavior. Setting only
  // `soft_mode` keeps the historical soft-only path. Setting both flags runs
  // hard as the primary path and also reports the soft score. The soft branch
  // requires the workspace to have been prepared with `allocate_soft_sw=true`.
  bool hard_mode = false;
  bool soft_mode = false;
  float temperature = kPairwiseDefaultSoftTemperature;
};

struct PairwiseStructureRequest {
  hikoboshi::universal::StructureView query{};
  hikoboshi::universal::StructureView target{};
  hikoboshi::modules::Mpnn64Descriptor descriptor{};
  const hikoboshi::modules::detail::Mpnn64Weights* weights = nullptr;
  PairwiseOptions options{};
  // Branch selectors propagated through to the embedding pipeline after MPNN
  // encoding. Mirrors the embedding-request fields.
  bool hard_mode = false;
  bool soft_mode = false;
  float temperature = kPairwiseDefaultSoftTemperature;
};

struct PairwiseResult {
  hikoboshi::universal::AlignmentPath path{};
  double raw_sw_score = 0.0;
  MetricBlock metrics{};
};

/// Request payload for the sequence-input pairwise route.
///
/// The caller hands over the already-tokenized token spans plus the
/// borrowed `WeightsView` for the compiled ESM2 package. The algorithms
/// layer is responsible for binding the safetensors tensor slots into a
/// typed `Esm2Weights` view, allocating the per-call `Esm2Workspace`,
/// running both encoder forward passes, and dispatching the resulting
/// embedding pair through the shared raw-dot + SW pipeline.
struct PairwiseSequenceRequest {
  hikoboshi::universal::Span<const std::int32_t> query_token_ids{};
  hikoboshi::universal::Span<const std::int32_t> target_token_ids{};
  hikoboshi::modules::Esm2Descriptor descriptor{};
  const hikoboshi::universal::WeightsView* weights_view = nullptr;
  PairwiseOptions options{};
  bool hard_mode = false;
  bool soft_mode = false;
  float temperature = kPairwiseDefaultSoftTemperature;
};

[[nodiscard]] hikoboshi::universal::Status run_pairwise_embeddings(
    const PairwiseEmbeddingRequest& request,
    detail::PairwiseWorkspace& workspace,
    PairwiseResult& result) noexcept;

[[nodiscard]] hikoboshi::universal::Status run_pairwise_structures(
    const PairwiseStructureRequest& request,
    detail::PairwiseWorkspace& workspace,
    PairwiseResult& result) noexcept;

/// Run the sequence-input pairwise pipeline.
///
/// Encodes both query and target token spans through the ESM2-8M scalar
/// forward pass, builds the raw-dot score matrix, and dispatches the
/// resulting embedding-shaped pair through the shared hard-SW (or soft-SW)
/// alignment path. The pairwise workspace must be prepared with
/// `allocate_mpnn = false` and `embedding_dimension` matching the
/// descriptor's hidden dimension.
[[nodiscard]] hikoboshi::universal::Status run_pairwise_sequences(
    const PairwiseSequenceRequest& request,
    detail::PairwiseWorkspace& workspace,
    PairwiseResult& result) noexcept;

void assemble_pairwise_result(
    double raw_sw_score,
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::EmbeddingView& query_embedding,
    const hikoboshi::universal::EmbeddingView& target_embedding,
    const hikoboshi::universal::StructureView& query_structure,
    const hikoboshi::universal::StructureView& target_structure,
    PairwiseResult& result) noexcept;

namespace detail {

/// Run the ESM2-8M scalar forward pass for one tokenized sequence.
///
/// Binds the embedded safetensors `WeightsView` into a typed ESM2 weight
/// view, allocates a per-call `Esm2Workspace`, runs `esm2_forward_scalar`,
/// and writes the resulting `[seq_len, hidden_dim]` embeddings into
/// `embeddings_out`. The caller owns the output buffer and is responsible
/// for sizing it to `token_ids.size * descriptor.hidden_dimension` floats.
/// Used by both the pairwise sequence-input branch and the all-vs-all
/// sequence-input branch so the typed-binding logic lives in one place.
[[nodiscard]] hikoboshi::universal::Status encode_esm2_sequence(
    const hikoboshi::universal::WeightsView& weights_view,
    const hikoboshi::modules::Esm2Descriptor& descriptor,
    hikoboshi::universal::Span<const std::int32_t> token_ids,
    float* embeddings_out) noexcept;

}  // namespace detail

}  // namespace hikoboshi::algorithms

#endif  // HIKOBOSHI_ALGORITHMS_PAIRWISE_HPP

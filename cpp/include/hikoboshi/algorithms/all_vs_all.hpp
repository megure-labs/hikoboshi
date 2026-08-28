#ifndef HIKOBOSHI_ALGORITHMS_ALL_VS_ALL_HPP
#define HIKOBOSHI_ALGORITHMS_ALL_VS_ALL_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/modules/detail/esm2_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/esm2.hpp>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>
#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>
#include <hikoboshi/universal/weights.hpp>

namespace hikoboshi::algorithms {

struct AllVsAllOptions {
  bool include_self = false;
  PairwiseOptions pairwise{};
  // Branch selectors propagated through to every pair's
  // PairwiseEmbeddingRequest after API AlignmentMode lowering. When
  // `soft_mode` is false, hard-SW runs by default even if `hard_mode` is
  // false; setting only `soft_mode` keeps the historical soft-only path, and
  // setting both flags runs both passes. Soft/both require the per-pair
  // workspace to have been prepared with `allocate_soft_sw = true` (the
  // all-vs-all driver sets this on the workspace plan when `soft_mode` is
  // true here).
  bool hard_mode = false;
  bool soft_mode = false;
  float temperature = kPairwiseDefaultSoftTemperature;
};

struct AllVsAllEmbeddingRequest {
  hikoboshi::universal::Span<const hikoboshi::universal::EmbeddingView> embeddings{};
  AllVsAllOptions options{};
};

struct AllVsAllStructureRequest {
  hikoboshi::universal::Span<const hikoboshi::universal::StructureView> structures{};
  hikoboshi::modules::Mpnn64Descriptor descriptor{};
  const hikoboshi::modules::detail::Mpnn64Weights* weights = nullptr;
  AllVsAllOptions options{};
};

/// One sequence entry inside an `AllVsAllSequenceRequest`. The optional
/// `name` is propagated into result diagnostics so the streaming sink can
/// label records; `token_ids` references caller-owned storage.
struct AllVsAllSequenceEntry {
  std::string_view name{};
  hikoboshi::universal::Span<const std::int32_t> token_ids{};
};

struct AllVsAllSequenceRequest {
  hikoboshi::universal::Span<const AllVsAllSequenceEntry> sequences{};
  hikoboshi::modules::Esm2Descriptor descriptor{};
  const hikoboshi::universal::WeightsView* weights_view = nullptr;
  AllVsAllOptions options{};
};

/// Algorithm-layer pair-list requests.
///
/// Each `PairList*Request` mirrors the matching `AllVsAll*Request` source
/// input but carries an explicit `pairs` list instead of enumerating every
/// `i < j` combination. Each entry is a `(query_index, target_index)` pair
/// into the route's source span — the api/engine layer resolves the
/// caller-supplied string IDs to these indices before calling down. Records
/// are emitted one per `pairs` entry, in the order `pairs` lists them.
/// `AllVsAllOptions` is reused unchanged; its `include_self` field is inert
/// for pair-list.
struct PairListEmbeddingRequest {
  hikoboshi::universal::Span<const hikoboshi::universal::EmbeddingView>
      embeddings{};
  hikoboshi::universal::Span<const std::pair<std::size_t, std::size_t>> pairs{};
  // Optional sizing hints from the resolved pair list. The implementation
  // validates against `pairs` and never trusts an undersized hint.
  std::size_t max_query_length = 0;
  std::size_t max_target_length = 0;
  AllVsAllOptions options{};
};

struct PairListStructureRequest {
  hikoboshi::universal::Span<const hikoboshi::universal::StructureView>
      structures{};
  hikoboshi::modules::Mpnn64Descriptor descriptor{};
  const hikoboshi::modules::detail::Mpnn64Weights* weights = nullptr;
  hikoboshi::universal::Span<const std::pair<std::size_t, std::size_t>> pairs{};
  // Optional sizing hints from the resolved pair list. The implementation
  // validates against `pairs` and never trusts an undersized hint.
  std::size_t max_query_length = 0;
  std::size_t max_target_length = 0;
  AllVsAllOptions options{};
};

struct PairListSequenceRequest {
  hikoboshi::universal::Span<const AllVsAllSequenceEntry> sequences{};
  hikoboshi::modules::Esm2Descriptor descriptor{};
  const hikoboshi::universal::WeightsView* weights_view = nullptr;
  hikoboshi::universal::Span<const std::pair<std::size_t, std::size_t>> pairs{};
  // Optional sizing hints from the resolved pair list. The implementation
  // validates against `pairs` and never trusts an undersized hint.
  std::size_t max_query_length = 0;
  std::size_t max_target_length = 0;
  AllVsAllOptions options{};
};

struct PerWorkerPairwiseBundleRequest {
  detail::PairwiseWorkspacePlan pairwise_plan{};
  detail::PairwiseWorkspacePlan encoder_plan{};
  bool prepare_pairwise = true;
  bool prepare_encoder = false;
  std::size_t max_result_step_count = 0;
};

class PairwiseResultSink {
 public:
  virtual ~PairwiseResultSink() = default;
  [[nodiscard]] virtual hikoboshi::universal::Status receive(
      const PairwiseResultRecord& record) = 0;
};

struct BoundedRecordStagingRequest {
  hikoboshi::universal::Span<const PairwiseResultRecord> ordered_records{};
  hikoboshi::universal::Span<const PairwiseResultRecord*> output_buffer{};
  PairwiseResultSink* sink = nullptr;
};

struct BoundedRecordStagingResult {
  std::size_t emitted_count = 0;
  std::size_t flush_count = 0;
};

struct PerWorkerPairwiseBundlePairRequest {
  std::size_t pair_index = 0;
  hikoboshi::universal::Span<const hikoboshi::universal::EmbeddingView> embeddings{};
  hikoboshi::universal::Span<const hikoboshi::universal::StructureView> structures{};
  const AllVsAllOptions* options = nullptr;
};

[[nodiscard]] hikoboshi::universal::Status prepare_per_worker_pairwise_bundle(
    detail::PerWorkerPairwiseBundle& bundle,
    const PerWorkerPairwiseBundleRequest& request);

[[nodiscard]] hikoboshi::universal::Status run_per_worker_pairwise_bundle_pair(
    const PerWorkerPairwiseBundlePairRequest& request,
    detail::PerWorkerPairwiseBundle& bundle,
    PairwiseResultRecord& record);

[[nodiscard]] hikoboshi::universal::Status bounded_record_staging(
    const BoundedRecordStagingRequest& request,
    BoundedRecordStagingResult& result);

namespace detail {

/// Estimate the per-thread ESM2 encoder workspace in bytes for the actual
/// maximum token count. Mirrors
/// `estimate_all_vs_all_structure_encoder_workspace_bytes`: it sums the
/// per-call `prepare_esm2_workspace` buffer set for the wrapped
/// `[<cls>, aa..., <eos>]` length (`max_token_count + 2`), whose dominant
/// term is the O(L^2) attention score scratch. The engine sizes the
/// sequence-route phase-1 thread budget from this; using the real
/// `max_token_count` instead of the descriptor's 65536 guard keeps the
/// budget from collapsing to a single worker. Returns false on overflow.
[[nodiscard]] bool estimate_all_vs_all_sequence_encoder_workspace_bytes(
    std::size_t max_token_count,
    const hikoboshi::modules::Esm2Descriptor& descriptor,
    std::size_t& bytes) noexcept;

}  // namespace detail

[[nodiscard]] hikoboshi::universal::Status run_all_vs_all_embeddings(
    const AllVsAllEmbeddingRequest& request,
    PairwiseResultSink& sink);

[[nodiscard]] hikoboshi::universal::Status run_all_vs_all_embeddings(
    const AllVsAllEmbeddingRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    hikoboshi::universal::Span<detail::AllVsAllWorkerWorkspace>
        worker_workspaces);

[[nodiscard]] hikoboshi::universal::Status run_all_vs_all_structures(
    const AllVsAllStructureRequest& request,
    PairwiseResultSink& sink);

[[nodiscard]] hikoboshi::universal::Status run_all_vs_all_structures(
    const AllVsAllStructureRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    hikoboshi::universal::Span<detail::AllVsAllWorkerWorkspace>
        worker_workspaces);

/// Run all-vs-all comparison over named tokenized sequences.
///
/// Each sequence is encoded once via the ESM2-8M forward pass into an
/// in-memory per-residue embedding cache; pair execution then re-uses the
/// existing embedding-input enumerator so streaming sink semantics, soft
/// SW handling, and lexicographic ordering match the structure path.
[[nodiscard]] hikoboshi::universal::Status run_all_vs_all_sequences(
    const AllVsAllSequenceRequest& request,
    PairwiseResultSink& sink);

/// Threaded overload of `run_all_vs_all_sequences`. Parallelizes the
/// per-sequence ESM2 encode and the per-pair alignment dispatch over `pool`
/// exactly as the structure route does, reusing the same
/// `acquire_all_vs_all_threading` workspaces. With `pool == nullptr` or
/// `thread_count <= 1` it is byte-identical to the serial overload; the
/// deterministic pair-ordered sink keeps parallel output identical to
/// serial regardless of worker scheduling.
[[nodiscard]] hikoboshi::universal::Status run_all_vs_all_sequences(
    const AllVsAllSequenceRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    hikoboshi::universal::Span<detail::AllVsAllWorkerWorkspace>
        worker_workspaces);

/// Run pair-list comparison over the caller-supplied `(query, target)` index
/// pairs in `request.pairs`, emitting one record per pair to `sink` in input
/// order. Pair-list shares the embedding cache, alignment worker dispatch,
/// and streaming sink with the `run_all_vs_all_*` family; unlike all-vs-all
/// it aligns only the listed pairs and never enumerates an unlisted one.
/// Declared by npc1a — the body returns `StatusCode::Unimplemented` until
/// npc1b factors the shared dispatcher and lands the implementation.
[[nodiscard]] hikoboshi::universal::Status run_pair_list_embeddings(
    const PairListEmbeddingRequest& request,
    PairwiseResultSink& sink);

[[nodiscard]] hikoboshi::universal::Status run_pair_list_structures(
    const PairListStructureRequest& request,
    PairwiseResultSink& sink);

/// Threaded overload of `run_pair_list_structures`. Parallelizes the
/// per-structure MPNN encode and per-pair dispatch over `pool`; completed
/// pair records are drained in caller input order. With
/// `pool == nullptr` or `thread_count <= 1` it is byte-identical to the
/// serial overload, and the encode-once invariant is preserved.
[[nodiscard]] hikoboshi::universal::Status run_pair_list_structures(
    const PairListStructureRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    hikoboshi::universal::Span<detail::AllVsAllWorkerWorkspace>
        worker_workspaces);

[[nodiscard]] hikoboshi::universal::Status run_pair_list_sequences(
    const PairListSequenceRequest& request,
    PairwiseResultSink& sink);

/// Threaded overload of `run_pair_list_sequences`. Parallelizes the
/// per-sequence ESM2 encode and per-pair dispatch over `pool`; completed
/// pair records are drained in caller input order. With
/// `pool == nullptr` or `thread_count <= 1` it is byte-identical to the
/// serial overload, and the encode-once invariant is preserved.
[[nodiscard]] hikoboshi::universal::Status run_pair_list_sequences(
    const PairListSequenceRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    hikoboshi::universal::Span<detail::AllVsAllWorkerWorkspace>
        worker_workspaces);

#ifndef NDEBUG
/// Debug-only encode-once instrumentation for the pair-list routes (npc1b
/// packet section E). `pair_list_debug_encode_count` returns the number of
/// per-protein encode passes the `run_pair_list_*` routes have performed
/// since the last `pair_list_debug_reset_encode_count`. The encode-once
/// invariant holds iff that count equals the unique-protein count of the
/// pair-list invocation. Both functions exist only in debug (`!NDEBUG`)
/// builds; release builds compile the counter out entirely.
void pair_list_debug_reset_encode_count() noexcept;
[[nodiscard]] std::size_t pair_list_debug_encode_count() noexcept;
#endif

}  // namespace hikoboshi::algorithms

#endif  // HIKOBOSHI_ALGORITHMS_ALL_VS_ALL_HPP

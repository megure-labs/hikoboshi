#ifndef HIKOBOSHI_API_ENGINE_HPP
#define HIKOBOSHI_API_ENGINE_HPP

/// @file
/// In-memory public C++ engine for Hikoboshi 0.1.0 workflows.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/requests.hpp>
#include <hikoboshi/api/results.hpp>
#include <hikoboshi/api/version.hpp>
#include <hikoboshi/universal/execution_options.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/planner.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/weights.hpp>

namespace hikoboshi::api {

/// Parity-mode selector for sequence-input encode/pairwise/all-vs-all.
///
/// Hikoboshi 0.1.0 routes scalar GEMM kernels through a build-default parity
/// mode (BLIS-style fast kernel by default). Sequence-route public requests
/// carry the user's parity preference for forward documentation; the runtime
/// kernel selection still goes through the established
/// `HIKOBOSHI_GEMM_PARITY_MODE` env knob until per-call parity dispatch lands
/// in a later packet.
enum class ParityTag {
  Strict,
  Fast,
};

/// One named sequence entry inside an all-vs-all sequence request.
struct SequenceEntry {
  std::string_view name{};
  universal::Span<const std::int32_t> token_ids{};
};

/// Encode one tokenized sequence into per-residue embeddings.
///
/// `token_ids` references the integer ids produced by the package's embedded
/// tokenizer table (the CLI/Python adapter performs tokenization before
/// constructing the request because the api layer is not permitted to
/// include the weights TU that owns the table). `package_id` selects the
/// compiled package via the canonical id or one of its declared aliases.
struct EncodeSequenceRequest {
  universal::Span<const std::int32_t> token_ids{};
  std::string_view package_id{};
  ParityTag parity = ParityTag::Fast;
  EncodeOptions options{};
};

/// Pairwise alignment request for two tokenized sequences.
///
/// Mirrors `PairwiseEmbeddingRequest`: `mode` defaults to hard
/// Smith-Waterman. Pass `AlignmentMode::Soft` for soft-only scoring or
/// `AlignmentMode::Both` to return hard and soft scores in one request.
/// Sequence-input alignments produce FASTA artifacts only — there are no
/// coordinates for PDB superposition.
struct PairwiseSequenceRequest {
  universal::Span<const std::int32_t> query_token_ids{};
  universal::Span<const std::int32_t> target_token_ids{};
  std::string_view package_id{};
  ParityTag parity = ParityTag::Fast;
  AlignmentOptions alignment{};
  AlignmentMode mode = AlignmentMode::Hard;
  float temperature = kDefaultSoftTemperature;
};

/// All-vs-all request over named, tokenized sequences.
///
/// Sequences are encoded once per worker and re-used across all pair
/// comparisons; the streaming sink receives per-pair records in lexicographic
/// `(query_index, target_index)` order. Identical semantics to
/// `AllVsAllEmbeddingRequest` minus the structure metrics that require atom
/// coordinates.
struct AllVsAllSequenceRequest {
  universal::Span<const SequenceEntry> sequences{};
  std::string_view package_id{};
  ParityTag parity = ParityTag::Fast;
  AllVsAllOptions options{};
};

/// Pair-list request over named, tokenized sequences.
///
/// Mirrors `AllVsAllSequenceRequest` but aligns only the caller-supplied
/// `pairs` instead of every `i < j` combination: each unique protein across
/// the pair list is encoded once, then exactly the listed pairs are aligned,
/// emitting one record per pair in input order. See `PairList*Request` in
/// `requests.hpp` for the shared `pairs` contract (case-sensitive IDs,
/// fail-fast on an ID absent from `sequences`, input-order output).
///
/// This request lives in `engine.hpp` rather than `requests.hpp` because —
/// exactly like `AllVsAllSequenceRequest` — it depends on the `SequenceEntry`
/// and `ParityTag` types declared in this header.
struct PairListSequenceRequest {
  universal::Span<const SequenceEntry> sequences{};
  std::string_view package_id{};
  ParityTag parity = ParityTag::Fast;
  std::vector<std::pair<std::string, std::string>> pairs{};
  AllVsAllOptions options{};
};

/// Configuration for the in-memory Hikoboshi API engine.
///
/// Default construction requests automatic backend selection, automatic
/// thread-count selection, and no caller-provided package. Adapters normally
/// resolve the compiled default package in the weights layer and pass the
/// resulting `PackageHandle` into this config before running encode,
/// pairwise, or all-vs-all workflows.
struct EngineConfig {
  /// Compatibility weight view for callers that still pass tensor views.
  universal::WeightsHandle weights{nullptr, nullptr};
  /// Backend and thread preferences.
  universal::ExecutionOptions execution{universal::Backend::Auto, 0};
  /// Preferred package/scoring descriptor and prepared state.
  universal::PackageHandle package{nullptr, nullptr};
  /// Optional planner policy. Null selects the scalar-only 0.1.0 policy.
  const universal::PlannerPolicy* planner_policy = nullptr;
};

/// In-memory public C++ entry point for Hikoboshi 0.1.0.
///
/// The engine accepts normalized structures, canonical coordinate views, or
/// residue embeddings. It does not own file IO; CLI and Python adapters load
/// files before constructing these requests. Pairwise workflows run hard local
/// affine Smith-Waterman over raw-dot hikoboshi-mpnn-d64 score matrices.
class Engine {
 public:
  /// Construct an engine from public package and execution configuration.
  explicit Engine(EngineConfig config = {});

  /// Return the immutable configuration captured by this engine.
  const EngineConfig& config() const noexcept;

  /// Encode one normalized structure into hikoboshi-mpnn-d64 residue embeddings.
  [[nodiscard]] universal::Result<EncodeResult> encode(
      const EncodeStructureRequest& request) const;
  /// Encode one canonical coordinate view into hikoboshi-mpnn-d64 embeddings.
  [[nodiscard]] universal::Result<EncodeResult> encode(
      const EncodeCoordsRequest& request) const;
  /// Encode one tokenized sequence into hikoboshi-esm2-8m residue embeddings.
  [[nodiscard]] universal::Result<EncodeResult> encode(
      const EncodeSequenceRequest& request) const;

  /// Design one or more sequences for a normalized backbone through the
  /// compiled ProteinMPNN inverse-folding package.
  [[nodiscard]] universal::Result<InverseFoldResult> inverse_fold(
      const InverseFoldRequest& request) const;

  /// Align two normalized structures and return path, metrics, and warnings.
  [[nodiscard]] universal::Result<PairwiseResult> pairwise(
      const PairwiseStructureRequest& request) const;
  /// Align two canonical coordinate inputs.
  [[nodiscard]] universal::Result<PairwiseResult> pairwise(
      const PairwiseCoordsRequest& request) const;
  /// Align two precomputed embedding matrices.
  [[nodiscard]] universal::Result<PairwiseResult> pairwise(
      const PairwiseEmbeddingRequest& request) const;
  /// Align two tokenized sequences through the ESM2-8M encoder + raw-dot
  /// scoring + hard-SW (or soft-SW) alignment path. Returns FASTA-ready
  /// alignment cells; coordinate-dependent metrics are reported as
  /// `NA`/unavailable.
  [[nodiscard]] universal::Result<PairwiseResult> pairwise(
      const PairwiseSequenceRequest& request) const;

  /// Stream pairwise records for all `i < j` structure pairs by default.
  [[nodiscard]] universal::Status all_vs_all(
      const AllVsAllStructureRequest& request,
      PairwiseResultSink& sink) const;
  /// Stream pairwise records for all `i < j` coordinate pairs by default.
  [[nodiscard]] universal::Status all_vs_all(
      const AllVsAllCoordsRequest& request,
      PairwiseResultSink& sink) const;
  /// Stream pairwise records for all `i < j` embedding pairs by default.
  [[nodiscard]] universal::Status all_vs_all(
      const AllVsAllEmbeddingRequest& request,
      PairwiseResultSink& sink) const;
  /// Stream pairwise records for all `i < j` sequence pairs by default.
  /// Each sequence is encoded once per worker before pair execution; output
  /// records reuse the `PairwiseResultRecord` shape with a FASTA-only
  /// metric subset.
  [[nodiscard]] universal::Status all_vs_all(
      const AllVsAllSequenceRequest& request,
      PairwiseResultSink& sink) const;

  /// Convenience wrapper that collects structure all-vs-all records in memory.
  ///
  /// Deprecated since 0.1.0: this path pre-allocates `pair_count` records and
  /// does not bound peak memory at scale. Prefer the streaming
  /// `Engine::all_vs_all(..., PairwiseResultSink&)` overloads, or the helper
  /// `hikoboshi::api::stream_all_vs_all` that drives a TSV streaming sink.
  /// Removal is targeted for 0.2.0.
  [[deprecated(
      "use Engine::all_vs_all(..., PairwiseResultSink&) or "
      "hikoboshi::api::stream_all_vs_all to bound peak memory")]] [[nodiscard]]
  universal::Result<AllVsAllResult> collect_all_vs_all(
      const AllVsAllStructureRequest& request) const;
  /// Convenience wrapper that collects coordinate all-vs-all records.
  ///
  /// Deprecated since 0.1.0; see structure overload above for migration.
  [[deprecated(
      "use Engine::all_vs_all(..., PairwiseResultSink&) or "
      "hikoboshi::api::stream_all_vs_all to bound peak memory")]] [[nodiscard]]
  universal::Result<AllVsAllResult> collect_all_vs_all(
      const AllVsAllCoordsRequest& request) const;
  /// Convenience wrapper that collects embedding all-vs-all records.
  ///
  /// Deprecated since 0.1.0; see structure overload above for migration.
  [[deprecated(
      "use Engine::all_vs_all(..., PairwiseResultSink&) or "
      "hikoboshi::api::stream_all_vs_all to bound peak memory")]] [[nodiscard]]
  universal::Result<AllVsAllResult> collect_all_vs_all(
      const AllVsAllEmbeddingRequest& request) const;

  /// Align a caller-supplied list of named structure pairs.
  ///
  /// Pair-list is the third alignment driver mode alongside `pairwise` and
  /// `all_vs_all`: it dedups the unique proteins across `request.pairs`,
  /// encodes each exactly once, and aligns only the listed pairs, collecting
  /// one record per pair in input order. Records reuse the `AllVsAllResult`
  /// shape; this entry mirrors the `collect_all_vs_all` convenience
  /// signature. Declared by npc1a — the body returns
  /// `StatusCode::Unimplemented` until npc1b lands the pair-list dedup +
  /// encode-once + per-pair dispatch implementation.
  [[nodiscard]] universal::Result<AllVsAllResult> collect_pair_list(
      const PairListStructureRequest& request) const;
  /// Align a caller-supplied list of named coordinate pairs.
  ///
  /// See the structure overload; declaration-only until npc1b.
  [[nodiscard]] universal::Result<AllVsAllResult> collect_pair_list(
      const PairListCoordsRequest& request) const;
  /// Align a caller-supplied list of named embedding pairs.
  ///
  /// See the structure overload; declaration-only until npc1b.
  [[nodiscard]] universal::Result<AllVsAllResult> collect_pair_list(
      const PairListEmbeddingRequest& request) const;
  /// Align a caller-supplied list of named sequence pairs.
  ///
  /// See the structure overload; declaration-only until npc1b.
  [[nodiscard]] universal::Result<AllVsAllResult> collect_pair_list(
      const PairListSequenceRequest& request) const;

  /// Return product version metadata.
  VersionInfo version_info() const noexcept;
  /// Return compiled/runtime backend and workflow capability metadata.
  BackendCapabilities backend_capabilities() const noexcept;

 private:
  EngineConfig config_;
  std::shared_ptr<void> threading_;
};

/// Run all-vs-all and collect emitted pair records in memory.
///
/// Deprecated since 0.1.0: this convenience wrapper holds every pair record at
/// once and is not safe for the symmetric-N² scale targeted by Hikoboshi 0.1.0.
/// Prefer `Engine::all_vs_all(..., PairwiseResultSink&)` for in-memory
/// streaming, or `hikoboshi::api::stream_all_vs_all` to drive a TSV streaming
/// sink without ever materializing an `AllVsAllResult`. Removal is targeted
/// for 0.2.0.
[[deprecated(
    "use Engine::all_vs_all(..., PairwiseResultSink&) or "
    "hikoboshi::api::stream_all_vs_all to bound peak memory")]] [[nodiscard]]
universal::Result<AllVsAllResult> collect_all_vs_all(
    const Engine& engine,
    const AllVsAllStructureRequest& request);
[[deprecated(
    "use Engine::all_vs_all(..., PairwiseResultSink&) or "
    "hikoboshi::api::stream_all_vs_all to bound peak memory")]] [[nodiscard]]
universal::Result<AllVsAllResult> collect_all_vs_all(
    const Engine& engine,
    const AllVsAllCoordsRequest& request);
[[deprecated(
    "use Engine::all_vs_all(..., PairwiseResultSink&) or "
    "hikoboshi::api::stream_all_vs_all to bound peak memory")]] [[nodiscard]]
universal::Result<AllVsAllResult> collect_all_vs_all(
    const Engine& engine,
    const AllVsAllEmbeddingRequest& request);

/// Run all-vs-all and stream emitted pair records to `sink`.
///
/// These free-function helpers mirror the deprecated `collect_all_vs_all`
/// helpers. They forward to the streaming `Engine::all_vs_all(..., sink)`
/// overload so adapters do not have to reach back into the `Engine` class for
/// the streaming path. Use them with `TsvStreamingAllVsAllSink` to write a
/// summary TSV without ever materializing an `AllVsAllResult`.
[[nodiscard]] universal::Status stream_all_vs_all(
    const Engine& engine,
    const AllVsAllStructureRequest& request,
    PairwiseResultSink& sink);
[[nodiscard]] universal::Status stream_all_vs_all(
    const Engine& engine,
    const AllVsAllCoordsRequest& request,
    PairwiseResultSink& sink);
[[nodiscard]] universal::Status stream_all_vs_all(
    const Engine& engine,
    const AllVsAllEmbeddingRequest& request,
    PairwiseResultSink& sink);
[[nodiscard]] universal::Status stream_all_vs_all(
    const Engine& engine,
    const AllVsAllSequenceRequest& request,
    PairwiseResultSink& sink);

}  // namespace hikoboshi::api

#endif  // HIKOBOSHI_API_ENGINE_HPP

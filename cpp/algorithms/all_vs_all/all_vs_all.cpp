#include <hikoboshi/algorithms/all_vs_all.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>
#include <hikoboshi/algorithms/detail/streaming_sink.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>

namespace hikoboshi::algorithms::detail {
namespace {

bool checked_add(std::size_t lhs,
                 std::size_t rhs,
                 std::size_t& output) noexcept {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    return false;
  }
  output = lhs + rhs;
  return true;
}

bool checked_mul(std::size_t lhs,
                 std::size_t rhs,
                 std::size_t& output) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  output = lhs * rhs;
  return true;
}

bool add_bytes(std::size_t& total,
               std::size_t element_count,
               std::size_t element_size) noexcept {
  std::size_t bytes = 0;
  if (!checked_mul(element_count, element_size, bytes)) {
    return false;
  }
  return checked_add(total, bytes, total);
}

bool add_mpnn_workspace_storage_bytes(
    const hikoboshi::modules::detail::Mpnn64MemoryPlan& plan,
    std::size_t& total) noexcept {
  namespace pmd = hikoboshi::modules::detail;
  return add_bytes(total, pmd::mpnn64_ca_coordinate_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_residue_feature_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_neighbor_slot_count(plan),
                   sizeof(std::int32_t)) &&
         add_bytes(total, pmd::mpnn64_neighbor_slot_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_neighbor_rbf_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_residue_hidden_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_neighbor_hidden_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_neighbor_hidden_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_neighbor_hidden_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_neighbor_hidden_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_residue_hidden_count(plan),
                   sizeof(float)) &&
         add_bytes(total, pmd::mpnn64_ffn_hidden_count(plan), sizeof(float));
}

}  // namespace

bool estimate_pairwise_workspace_storage_bytes(
    const PairwiseWorkspacePlan& plan,
    std::size_t& bytes) noexcept {
  bytes = 0;

  std::size_t query_embedding_count = 0;
  std::size_t target_embedding_count = 0;
  std::size_t pair_cell_count = 0;
  std::size_t sw_query_cells = 0;
  std::size_t sw_target_cells = 0;
  std::size_t sw_cell_count = 0;
  std::size_t max_path_steps = 0;
  if (!checked_mul(plan.max_query_length,
                   plan.embedding_dimension,
                   query_embedding_count) ||
      !checked_mul(plan.max_target_length,
                   plan.embedding_dimension,
                   target_embedding_count) ||
      !checked_mul(plan.max_query_length,
                   plan.max_target_length,
                   pair_cell_count) ||
      !checked_add(plan.max_query_length, 1U, sw_query_cells) ||
      !checked_add(plan.max_target_length, 1U, sw_target_cells) ||
      !checked_mul(sw_query_cells, sw_target_cells, sw_cell_count) ||
      !checked_add(plan.max_query_length, plan.max_target_length,
                   max_path_steps)) {
    return false;
  }

  if (!add_bytes(bytes, query_embedding_count, sizeof(float)) ||
      !add_bytes(bytes, target_embedding_count, sizeof(float)) ||
      !add_bytes(bytes, pair_cell_count, sizeof(float)) ||
      !add_bytes(bytes, sw_cell_count, sizeof(float)) ||
      !add_bytes(bytes, sw_cell_count, sizeof(float)) ||
      !add_bytes(bytes, sw_cell_count, sizeof(float)) ||
      !add_bytes(bytes, pair_cell_count, sizeof(SwTraceDirection)) ||
      !add_bytes(bytes, pair_cell_count, sizeof(SwTraceDirection)) ||
      !add_bytes(bytes, pair_cell_count, sizeof(SwTraceDirection)) ||
      !add_bytes(bytes, max_path_steps,
                 sizeof(hikoboshi::universal::AlignmentStep))) {
    return false;
  }

  if (plan.allocate_mpnn) {
    hikoboshi::modules::detail::Mpnn64MemoryPlan query_plan{};
    query_plan.max_residue_count = plan.max_query_length;
    query_plan.hidden_dimension = plan.mpnn_descriptor.hidden_dimension;
    query_plan.neighbor_count = plan.mpnn_descriptor.neighbor_count;
    query_plan.rbf_count = plan.mpnn_descriptor.rbf_count;
    query_plan.layer_count = plan.mpnn_descriptor.layer_count;

    hikoboshi::modules::detail::Mpnn64MemoryPlan target_plan = query_plan;
    target_plan.max_residue_count = plan.max_target_length;
    if (!add_mpnn_workspace_storage_bytes(query_plan, bytes) ||
        !add_mpnn_workspace_storage_bytes(target_plan, bytes)) {
      return false;
    }
  }

  return true;
}

bool estimate_all_vs_all_structure_encoder_workspace_bytes(
    std::size_t max_residue_count,
    const hikoboshi::modules::Mpnn64Descriptor& descriptor,
    std::size_t& bytes) noexcept {
  PairwiseWorkspacePlan plan{};
  plan.max_query_length = max_residue_count;
  plan.max_target_length = 0U;
  plan.embedding_dimension = descriptor.hidden_dimension;
  plan.allocate_mpnn = true;
  plan.mpnn_descriptor = descriptor;
  return estimate_pairwise_workspace_storage_bytes(plan, bytes);
}

bool estimate_all_vs_all_sequence_encoder_workspace_bytes(
    std::size_t max_token_count,
    const hikoboshi::modules::Esm2Descriptor& descriptor,
    std::size_t& bytes) noexcept {
  bytes = 0;

  // The ESM2 forward pass sees the wrapped [<cls>, aa..., <eos>] span, so the
  // per-call workspace is sized for max_token_count + 2 rows.
  constexpr std::size_t kSpecialTokenOverhead = 2U;
  std::size_t seq_len = 0;
  if (!checked_add(max_token_count, kSpecialTokenOverhead, seq_len)) {
    return false;
  }

  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t head_count = descriptor.head_count;
  const std::size_t head_dim = descriptor.head_dim;
  const std::size_t half = head_dim / 2U;
  const std::size_t ffn = descriptor.ffn_hidden_dimension;

  // Buffer shapes mirror `prepare_esm2_workspace` exactly:
  //   * 10 [seq_len, hidden] activation/projection buffers,
  //   * 4 [head_count, seq_len, head_dim] head-major buffers,
  //   * 2 [seq_len, head_dim/2] rope tables,
  //   * the [seq_len, seq_len] attention score scratch (the O(L^2) dominant
  //     term), and
  //   * the [seq_len, ffn_hidden] FFN intermediate buffer.
  std::size_t hidden_count = 0;
  std::size_t head_major_rows = 0;
  std::size_t head_major = 0;
  std::size_t rope_count = 0;
  std::size_t scores_count = 0;
  std::size_t ffn_count = 0;
  if (!checked_mul(seq_len, hidden, hidden_count) ||
      !checked_mul(head_count, seq_len, head_major_rows) ||
      !checked_mul(head_major_rows, head_dim, head_major) ||
      !checked_mul(seq_len, half, rope_count) ||
      !checked_mul(seq_len, seq_len, scores_count) ||
      !checked_mul(seq_len, ffn, ffn_count)) {
    return false;
  }

  std::size_t total_floats = 0;
  std::size_t scaled = 0;
  if (!checked_mul(hidden_count, 10U, scaled) ||
      !checked_add(total_floats, scaled, total_floats) ||
      !checked_mul(head_major, 4U, scaled) ||
      !checked_add(total_floats, scaled, total_floats) ||
      !checked_mul(rope_count, 2U, scaled) ||
      !checked_add(total_floats, scaled, total_floats) ||
      !checked_add(total_floats, scores_count, total_floats) ||
      !checked_add(total_floats, ffn_count, total_floats)) {
    return false;
  }
  return add_bytes(bytes, total_floats, sizeof(float));
}

std::size_t select_all_vs_all_phase1_thread_count_for_budget(
    std::size_t requested_thread_count,
    std::size_t item_count,
    std::size_t workspace_bytes_per_thread,
    std::size_t workspace_budget_bytes) noexcept {
  if (requested_thread_count <= 1U ||
      item_count < kAllVsAllParallelEncodingThreshold) {
    return 1U;
  }

  std::size_t selected = std::min(requested_thread_count, item_count);
  if (workspace_bytes_per_thread == 0U || workspace_budget_bytes == 0U) {
    return selected;
  }

  const std::size_t memory_limited_threads =
      workspace_budget_bytes / workspace_bytes_per_thread;
  if (memory_limited_threads == 0U) {
    return 1U;
  }
  return std::max<std::size_t>(1U,
                               std::min(selected, memory_limited_threads));
}

}  // namespace hikoboshi::algorithms::detail

namespace hikoboshi::algorithms {
namespace {

using hikoboshi::algorithms::detail::AllVsAllWorkerWorkspace;
using hikoboshi::algorithms::detail::PairIndex;
using hikoboshi::algorithms::detail::PairTile;
using hikoboshi::algorithms::detail::PairwiseWorkspace;
using hikoboshi::algorithms::detail::QueryTiledPlan;
using hikoboshi::universal::EmbeddingView;
using hikoboshi::universal::Span;
using hikoboshi::universal::Status;
using hikoboshi::universal::StructureView;

constexpr std::size_t kParallelPairThreshold = 45;
constexpr std::size_t kCostAwarePairSchedulingThreshold = 16;
constexpr std::size_t kParallelEncodingThreshold =
    detail::kAllVsAllParallelEncodingThreshold;

#if defined(HIKOBOSHI_ALLVSALL_PAIR_SCHEDULER_QUERY_TILED)
constexpr bool kQueryTiledEnabled = true;
#else
constexpr bool kQueryTiledEnabled = false;
#endif

#if defined(HIKOBOSHI_ALLVSALL_SINK_STREAMING)
// On the streaming branch the dispatcher submits pair_ids in ascending
// order so each worker's per-worker sequencer ring is naturally sorted on
// insertion. With the legacy ring buffer this was a hard deadlock-avoidance
// invariant (rotation would pin a slow head pair). With the new
// per-worker-buffers + sequencer design (ssf1) it is only a courtesy that
// keeps the sequencer's head-of-each-ring scan O(W); the sequencer no
// longer head-of-line-blocks workers when one pair is slow.
//
// Tile dispatch (query-tiled) keeps pair_id-ascending order *between*
// tiles, with Q_tile=1 forced and a small T_tile so each per-worker tile
// is one query row × T_tile target chunk and pair_id increments
// monotonically as the worker iterates the tile. The legacy ring path
// requires that the in-flight pair_id spread stay below the ring slot
// count; with slot_count = 65536 and W up to 180, T_tile = 64 keeps the
// spread comfortably bounded. The new sequencer path does not require any
// in-flight bound from this knob, but reuses the same value because the
// dispatch loop is shared.
constexpr std::size_t kStreamingQueryTileTargetSize = 64U;
#endif

#if !defined(HIKOBOSHI_ALLVSALL_SINK_STREAMING) && \
    !defined(HIKOBOSHI_BENCH_NO_DETERMINISTIC_SINK)
constexpr std::size_t kMaxParallelRecordBytes = 256U * 1024U * 1024U;
#endif

struct PairwiseStatusFailure {
  Status status;
};

struct PairDispatchItem {
  std::size_t pair_index = 0;
  std::uint64_t predicted_cost = 0;
};

// Maps a linear `pair_index` in `[0, pair_count)` to the
// `(query_index, target_index)` source items that pair aligns.
//
// `dispatch_pairs` is the shared per-pair execution helper (npc1b). Its
// only point of variation between the two driver modes is this plain
// tagged struct — not a virtual interface:
//
//   - all-vs-all uses `SymmetricEnumeration`: `pair_index` is decoded
//     through the upper-triangular `i < j` (or `i <= j`) geometry, and
//     cost-aware tiling / parallel scheduling stay valid;
//   - pair-list uses `ResolvedList`: `pair_index` indexes a caller-
//     supplied list of `(query_index, target_index)` pairs already
//     resolved against the source span.
//
// Records are always emitted in `pair_index`-ascending order: for
// all-vs-all that is lexicographic `(i, j)` order; for pair-list it is
// caller input order. Only the symmetric kind is eligible for the
// cost-aware tiled scheduler; the resolved-list kind uses an input-order
// preserving parallel path.
struct PairSource {
  enum class Kind { SymmetricEnumeration, ResolvedList };

  Kind kind = Kind::SymmetricEnumeration;
  // SymmetricEnumeration: the `i < j` combination geometry.
  std::size_t item_count = 0;
  bool include_self = false;
  // ResolvedList: caller-resolved `(query_index, target_index)` pairs.
  const std::pair<std::size_t, std::size_t>* list = nullptr;

  [[nodiscard]] bool symmetric() const noexcept {
    return kind == Kind::SymmetricEnumeration;
  }

  [[nodiscard]] PairIndex resolve(std::size_t pair_index) const noexcept {
    if (kind == Kind::SymmetricEnumeration) {
      return detail::pair_index_to_ij(pair_index, item_count, include_self);
    }
    return {list[pair_index].first, list[pair_index].second};
  }
};

PairSource symmetric_pair_source(std::size_t item_count,
                                 bool include_self) noexcept {
  PairSource source{};
  source.kind = PairSource::Kind::SymmetricEnumeration;
  source.item_count = item_count;
  source.include_self = include_self;
  return source;
}

PairSource resolved_list_pair_source(
    Span<const std::pair<std::size_t, std::size_t>> pairs) noexcept {
  PairSource source{};
  source.kind = PairSource::Kind::ResolvedList;
  source.list = pairs.data;
  return source;
}

std::size_t structure_coordinate_count(std::size_t residue_count) noexcept {
  return residue_count * hikoboshi::universal::kCanonicalAtomCount *
         hikoboshi::universal::kCoordinateAxisCount;
}

std::size_t structure_atom_source_count(std::size_t residue_count) noexcept {
  return residue_count * hikoboshi::universal::kCanonicalAtomCount;
}

Status validate_embedding(const EmbeddingView& embedding) noexcept {
  if (embedding.residue_count == 0U) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all embedding input must contain at least one residue");
  }
  if (embedding.dimension == 0U) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all embedding dimension must be non-zero");
  }
  if (embedding.values.data == nullptr ||
      embedding.values.size < embedding.residue_count * embedding.dimension) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all embedding values are invalid");
  }
  return hikoboshi::universal::ok_status();
}

Status validate_structure(const StructureView& structure) noexcept {
  if (structure.residue_count == 0U) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all structure input must contain at least one residue");
  }
  if (structure.coordinates.data == nullptr ||
      structure.coordinates.size <
          structure_coordinate_count(structure.residue_count)) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all structure coordinates are invalid");
  }
  if (structure.atom_sources.data == nullptr ||
      structure.atom_sources.size <
          structure_atom_source_count(structure.residue_count)) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all structure atom sources are invalid");
  }
  return hikoboshi::universal::ok_status();
}

Status validate_gap_options(const PairwiseOptions& options) noexcept {
  if (!std::isfinite(options.gap_open) ||
      !std::isfinite(options.gap_extension)) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all gap parameters must be finite");
  }
  return hikoboshi::universal::ok_status();
}

Status validate_embedding_set(
    hikoboshi::universal::Span<const EmbeddingView> embeddings,
    std::size_t& max_residue_count,
    std::size_t& dimension) noexcept {
  max_residue_count = 0;
  dimension = 0;
  for (std::size_t index = 0; index < embeddings.size; ++index) {
    const Status status = validate_embedding(embeddings.data[index]);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
    if (index == 0U) {
      dimension = embeddings.data[index].dimension;
    } else if (embeddings.data[index].dimension != dimension) {
      return hikoboshi::universal::invalid_argument_status("all-vs-all embedding dimensions differ");
    }
    max_residue_count =
        std::max(max_residue_count, embeddings.data[index].residue_count);
  }
  return hikoboshi::universal::ok_status();
}

Status validate_structure_set(
    hikoboshi::universal::Span<const StructureView> structures,
    std::size_t& max_residue_count) noexcept {
  max_residue_count = 0;
  for (std::size_t index = 0; index < structures.size; ++index) {
    const Status status = validate_structure(structures.data[index]);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
    max_residue_count =
        std::max(max_residue_count, structures.data[index].residue_count);
  }
  return hikoboshi::universal::ok_status();
}

std::uint64_t saturated_residue_count(std::size_t residue_count) noexcept {
  constexpr std::uint64_t kMaxCost =
      std::numeric_limits<std::uint64_t>::max();
  if (residue_count > static_cast<std::size_t>(kMaxCost)) {
    return kMaxCost;
  }
  return static_cast<std::uint64_t>(residue_count);
}

std::uint64_t saturated_pair_cost(std::size_t lhs_residue_count,
                                  std::size_t rhs_residue_count) noexcept {
  constexpr std::uint64_t kMaxCost =
      std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t lhs = saturated_residue_count(lhs_residue_count);
  const std::uint64_t rhs = saturated_residue_count(rhs_residue_count);
  if (lhs != 0U && rhs > kMaxCost / lhs) {
    return kMaxCost;
  }
  return lhs * rhs;
}

std::uint64_t predicted_embedding_pair_cost(
    std::size_t pair_index,
    Span<const EmbeddingView> embeddings,
    const AllVsAllOptions& options) noexcept {
  const PairIndex pair = detail::pair_index_to_ij(
      pair_index, embeddings.size, options.include_self);
  return saturated_pair_cost(embeddings.data[pair.query_index].residue_count,
                             embeddings.data[pair.target_index].residue_count);
}

void build_cost_descending_pair_dispatch_order(
    Span<const EmbeddingView> embeddings,
    const AllVsAllOptions& options,
    std::vector<PairDispatchItem>& dispatch_order) {
  for (std::size_t pair_index = 0; pair_index < dispatch_order.size();
       ++pair_index) {
    dispatch_order[pair_index] = {
        pair_index,
        predicted_embedding_pair_cost(pair_index, embeddings, options)};
  }

  std::sort(dispatch_order.begin(),
            dispatch_order.end(),
            [](const PairDispatchItem& lhs,
               const PairDispatchItem& rhs) noexcept {
              if (lhs.predicted_cost != rhs.predicted_cost) {
                return lhs.predicted_cost > rhs.predicted_cost;
              }
              return lhs.pair_index < rhs.pair_index;
            });
}

std::uint64_t saturated_add_u64(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  if (lhs > kMax - rhs) {
    return kMax;
  }
  return lhs + rhs;
}

std::uint64_t pair_tile_predicted_cost(
    const PairTile& tile,
    Span<const EmbeddingView> embeddings,
    bool include_self) noexcept {
  std::uint64_t total = 0;
  const std::size_t item_count = embeddings.size;
  detail::iterate_pair_tile(
      tile, item_count, include_self,
      [&](std::size_t /*pair_index*/, std::size_t q, std::size_t t) noexcept {
        const std::uint64_t pair_cost =
            detail::saturated_pair_cost_from_residue_counts(
                embeddings.data[q].residue_count,
                embeddings.data[t].residue_count);
        total = saturated_add_u64(total, pair_cost);
      });
  return total;
}

void build_query_tiled_dispatch_tiles(
    Span<const EmbeddingView> embeddings,
    const AllVsAllOptions& options,
    std::size_t query_tile,
    std::size_t target_tile,
    bool sort_cost_descending,
    std::vector<PairTile>& tiles) {
  const std::size_t item_count = embeddings.size;
  const std::size_t reserve_count = detail::worst_case_pair_tile_count(
      item_count, query_tile, target_tile);
  tiles.reserve(reserve_count);
  detail::partition_pair_tiles(item_count, options.include_self, query_tile,
                               target_tile, tiles);
  for (PairTile& tile : tiles) {
    tile.predicted_cost =
        pair_tile_predicted_cost(tile, embeddings, options.include_self);
  }
  if (sort_cost_descending) {
    std::sort(tiles.begin(), tiles.end(),
              [](const PairTile& lhs, const PairTile& rhs) noexcept {
                if (lhs.predicted_cost != rhs.predicted_cost) {
                  return lhs.predicted_cost > rhs.predicted_cost;
                }
                if (lhs.query_begin != rhs.query_begin) {
                  return lhs.query_begin < rhs.query_begin;
                }
                return lhs.target_begin < rhs.target_begin;
              });
  }
}

// Each worker should grab at least this many tiles for work-stealing
// flexibility; otherwise tile dispatch suffers from coarse load
// imbalance (one straggler tile blocks the wall-time tail). Empirically
// 4 keeps thread_scaling within parity gates at small fixtures while
// preserving cache locality at large N.
constexpr std::size_t kQueryTiledTilesPerWorker = 4U;

std::size_t resolve_query_tile_size(
    Span<const EmbeddingView> embeddings,
    std::size_t thread_count) noexcept {
  std::size_t residue_max = 0;
  std::size_t embedding_dimension = 0;
  for (std::size_t index = 0; index < embeddings.size; ++index) {
    const EmbeddingView& view = embeddings.data[index];
    if (view.residue_count > residue_max) {
      residue_max = view.residue_count;
    }
    if (view.dimension > embedding_dimension) {
      embedding_dimension = view.dimension;
    }
  }
  const std::size_t budget_q_tile =
      detail::resolve_query_tiled_plan(embeddings.size, residue_max,
                                       embedding_dimension)
          .query_tile;

  // Cap Q_tile so the worst-case tile count
  // (ceil(item_count / Q_tile) for target_tile=0) leaves at least
  // kQueryTiledTilesPerWorker tiles per worker.
  if (thread_count > 1U && embeddings.size > 0U) {
    const std::size_t target_tile_count =
        thread_count * kQueryTiledTilesPerWorker;
    const std::size_t balance_q_tile =
        embeddings.size >= target_tile_count
            ? embeddings.size / target_tile_count
            : 1U;
    return std::min(budget_q_tile,
                    std::max<std::size_t>(balance_q_tile, 1U));
  }
  return budget_q_tile;
}

detail::PairwiseWorkspacePlan embedding_workspace_plan(
    std::size_t max_query_length,
    std::size_t max_target_length,
    std::size_t dimension,
    bool allocate_soft_sw) noexcept {
  detail::PairwiseWorkspacePlan plan{};
  plan.max_query_length = max_query_length;
  plan.max_target_length = max_target_length;
  plan.embedding_dimension = dimension;
  plan.allocate_mpnn = false;
  plan.allocate_soft_sw = allocate_soft_sw;
  return plan;
}

detail::PairwiseWorkspacePlan structure_workspace_plan(
    std::size_t max_residue_count,
    const hikoboshi::modules::Mpnn64Descriptor& descriptor,
    bool allocate_soft_sw = false) noexcept {
  detail::PairwiseWorkspacePlan plan{};
  plan.max_query_length = max_residue_count;
  plan.max_target_length = 0U;
  plan.embedding_dimension = descriptor.hidden_dimension;
  plan.allocate_mpnn = true;
  plan.allocate_soft_sw = allocate_soft_sw;
  plan.mpnn_descriptor = descriptor;
  return plan;
}

detail::PairwiseWorkspacePlan embedding_workspace_plan(
    std::size_t max_residue_count,
    std::size_t dimension,
    bool allocate_soft_sw = false) noexcept {
  return embedding_workspace_plan(max_residue_count, max_residue_count,
                                  dimension, allocate_soft_sw);
}

Status encode_structure(const StructureView& structure,
                        const hikoboshi::modules::Mpnn64Descriptor& descriptor,
                        const hikoboshi::modules::detail::Mpnn64Weights* weights,
                        PairwiseWorkspace& workspace,
                        float* output) noexcept {
  hikoboshi::modules::Mpnn64ForwardRequest encode_request{};
  encode_request.coordinates = structure.coordinates.data;
  encode_request.atom_sources = structure.atom_sources.data;
  encode_request.residue_count = structure.residue_count;
  encode_request.descriptor = descriptor;
  encode_request.weights = weights;
  encode_request.workspace = workspace.query_mpnn_workspace();

  hikoboshi::modules::Mpnn64ForwardOutput encode_output{};
  encode_output.embeddings = output;
  encode_output.residue_count = structure.residue_count;
  encode_output.hidden_dimension = descriptor.hidden_dimension;
  return hikoboshi::modules::mpnn64_forward_scalar(encode_request,
                                                encode_output);
}

EmbeddingView encoded_embedding_view(const StructureView& structure,
                                     const float* values,
                                     std::size_t dimension) noexcept {
  return {structure.residue_count,
          dimension,
          {values, structure.residue_count * dimension},
          structure.residue_codes,
          structure.residues};
}

bool checked_cache_mul(std::size_t lhs,
                       std::size_t rhs,
                       std::size_t& output) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  output = lhs * rhs;
  return true;
}

class MaxLPaddedEmbeddingCache {
 public:
  Status init(Span<const StructureView> structures,
              std::size_t hidden_dimension) {
    hidden_dim_ = hidden_dimension;
    max_residue_count_ = 0;
    slot_stride_ = 0;
    values_.clear();

    for (std::size_t index = 0; index < structures.size; ++index) {
      max_residue_count_ =
          std::max(max_residue_count_, structures.data[index].residue_count);
    }
    if (hidden_dim_ != 0U &&
        max_residue_count_ >
            std::numeric_limits<std::size_t>::max() / hidden_dim_) {
      return hikoboshi::universal::invalid_argument_status(
          "all-vs-all encoded embedding stride overflows");
    }
    slot_stride_ = max_residue_count_ * hidden_dim_;
    if (slot_stride_ != 0U &&
        structures.size > std::numeric_limits<std::size_t>::max() /
                              slot_stride_) {
      return hikoboshi::universal::invalid_argument_status(
          "all-vs-all encoded embedding cache size overflows");
    }
    const std::size_t value_count = structures.size * slot_stride_;
    if (value_count > values_.max_size()) {
      return hikoboshi::universal::invalid_argument_status(
          "all-vs-all encoded embedding cache size exceeds allocator capacity");
    }

    try {
      values_.resize(value_count);
    } catch (const std::bad_alloc&) {
      return hikoboshi::universal::unavailable_status(
          "all-vs-all structure encoding cache allocation failed");
    }
    return hikoboshi::universal::ok_status();
  }

  float* slot(std::size_t global_index) noexcept {
    return values_.data() + global_index * slot_stride_;
  }

  EmbeddingView view(std::size_t global_index,
                     const StructureView& structure) const noexcept {
    return encoded_embedding_view(structure,
                                  values_.data() + global_index * slot_stride_,
                                  hidden_dim_);
  }

 private:
  std::vector<float> values_;
  std::size_t hidden_dim_ = 0;
  std::size_t max_residue_count_ = 0;
  std::size_t slot_stride_ = 0;
};

class BucketedEmbeddingCache {
 public:
  Status init(Span<const StructureView> structures,
              std::size_t hidden_dimension) {
    hidden_dim_ = hidden_dimension;
    buckets_.clear();
    global_to_bucket_.clear();

    try {
      buckets_.reserve(kDefaultBucketCount + structures.size);
      global_to_bucket_.assign(
          structures.size, {kInvalidLocation, kInvalidLocation});

      for (std::size_t index = 0; index < structures.size; ++index) {
        const std::size_t residue_count =
            structures.data[index].residue_count;
        const std::size_t bucket_residue_count =
            bucket_residue_count_for(residue_count);
        const bool ad_hoc = bucket_residue_count > kLargestDefaultBucket;
        const std::size_t bucket_index =
            ad_hoc ? append_bucket(bucket_residue_count)
                   : find_or_append_bucket(bucket_residue_count);
        Bucket& bucket = buckets_[bucket_index];
        const std::size_t slot_index = bucket.entry_to_global.size();
        bucket.entry_to_global.push_back(index);
        global_to_bucket_[index] = {bucket_index, slot_index};
      }
    } catch (const std::bad_alloc&) {
      return hikoboshi::universal::unavailable_status(
          "all-vs-all bucketed embedding cache allocation failed");
    }

    for (Bucket& bucket : buckets_) {
      std::size_t slot_value_count = 0;
      if (!checked_cache_mul(bucket.bucket_residue_count, hidden_dim_,
                             slot_value_count)) {
        return hikoboshi::universal::invalid_argument_status(
            "all-vs-all encoded embedding stride overflows");
      }

      std::size_t value_count = 0;
      if (!checked_cache_mul(bucket.entry_to_global.size(), slot_value_count,
                             value_count)) {
        return hikoboshi::universal::invalid_argument_status(
            "all-vs-all encoded embedding cache size overflows");
      }
      if (value_count > bucket.values.max_size()) {
        return hikoboshi::universal::invalid_argument_status(
            "all-vs-all encoded embedding cache size exceeds allocator capacity");
      }

      try {
        bucket.values.resize(value_count);
      } catch (const std::bad_alloc&) {
        return hikoboshi::universal::unavailable_status(
            "all-vs-all bucketed embedding cache allocation failed");
      }
    }

    return hikoboshi::universal::ok_status();
  }

  float* slot(std::size_t global_index) noexcept {
    const SlotLocation location = global_to_bucket_[global_index];
    Bucket& bucket = buckets_[location.bucket_index];
    return bucket.values.data() +
           location.slot_index * bucket.bucket_residue_count * hidden_dim_;
  }

  EmbeddingView view(std::size_t global_index,
                     const StructureView& structure) const noexcept {
    const SlotLocation location = global_to_bucket_[global_index];
    const Bucket& bucket = buckets_[location.bucket_index];
    return encoded_embedding_view(
        structure,
        bucket.values.data() +
            location.slot_index * bucket.bucket_residue_count * hidden_dim_,
        hidden_dim_);
  }

 private:
  static constexpr std::size_t kInvalidLocation =
      std::numeric_limits<std::size_t>::max();
  static constexpr std::size_t kDefaultBucketCount = 6;
  static constexpr std::size_t kLargestDefaultBucket = 2048;
  inline static constexpr std::size_t kDefaultBucketSizes[kDefaultBucketCount] = {
      64, 128, 256, 512, 1024, 2048};

  struct SlotLocation {
    std::size_t bucket_index;
    std::size_t slot_index;
  };

  struct Bucket {
    std::size_t bucket_residue_count = 0;
    std::vector<float> values;
    std::vector<std::size_t> entry_to_global;
  };

  static std::size_t bucket_residue_count_for(
      std::size_t residue_count) noexcept {
    for (std::size_t bucket_size : kDefaultBucketSizes) {
      if (residue_count <= bucket_size) {
        return bucket_size;
      }
    }
    return residue_count;
  }

  std::size_t append_bucket(std::size_t bucket_residue_count) {
    Bucket bucket{};
    bucket.bucket_residue_count = bucket_residue_count;
    buckets_.push_back(std::move(bucket));
    return buckets_.size() - 1U;
  }

  std::size_t find_or_append_bucket(std::size_t bucket_residue_count) {
    for (std::size_t index = 0; index < buckets_.size(); ++index) {
      if (buckets_[index].bucket_residue_count == bucket_residue_count) {
        return index;
      }
    }
    return append_bucket(bucket_residue_count);
  }

  std::vector<Bucket> buckets_;
  std::vector<SlotLocation> global_to_bucket_;
  std::size_t hidden_dim_ = 0;
};

#if defined(HIKOBOSHI_ALLVSALL_EMBEDDING_STORAGE_BUCKETED)
using EncodedEmbeddingCache = BucketedEmbeddingCache;
#else
using EncodedEmbeddingCache = MaxLPaddedEmbeddingCache;
#endif

bool has_structure_metadata(Span<const StructureView> structures,
                            std::size_t item_count) noexcept {
  return structures.data != nullptr && structures.size == item_count;
}

Status compute_embedding_pair_record(
    std::size_t pair_index,
    const PairSource& source,
    Span<const EmbeddingView> embeddings,
    Span<const StructureView> structures,
    const AllVsAllOptions& options,
    PairwiseWorkspace& workspace,
    PairwiseResultRecord& record) {
  const PairIndex pair = source.resolve(pair_index);
  PairwiseEmbeddingRequest pairwise_request{};
  pairwise_request.query_embedding = embeddings.data[pair.query_index];
  pairwise_request.target_embedding = embeddings.data[pair.target_index];
  if (has_structure_metadata(structures, embeddings.size)) {
    pairwise_request.query_structure = structures.data[pair.query_index];
    pairwise_request.target_structure = structures.data[pair.target_index];
  }
  pairwise_request.options = options.pairwise;
  pairwise_request.hard_mode = options.hard_mode;
  pairwise_request.soft_mode = options.soft_mode;
  pairwise_request.temperature = options.temperature;

  record.query_index = pair.query_index;
  record.target_index = pair.target_index;
  const Status status =
      run_pairwise_embeddings(pairwise_request, workspace, record.result);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  return hikoboshi::universal::ok_status();
}

// Overload preserving the original symmetric-enumeration call shape. The
// all-vs-all parallel paths and `run_per_worker_pairwise_bundle_pair` keep
// this signature; it forwards to the source-aware overload with the
// `i < j` enumeration source, so their behavior is unchanged.
Status compute_embedding_pair_record(
    std::size_t pair_index,
    Span<const EmbeddingView> embeddings,
    Span<const StructureView> structures,
    const AllVsAllOptions& options,
    PairwiseWorkspace& workspace,
    PairwiseResultRecord& record) {
  return compute_embedding_pair_record(
      pair_index,
      symmetric_pair_source(embeddings.size, options.include_self),
      embeddings, structures, options, workspace, record);
}

Status run_serial_embedding_pairs(
    const PairSource& source,
    Span<const EmbeddingView> embeddings,
    Span<const StructureView> structures,
    const AllVsAllOptions& options,
    std::size_t pair_count,
    const detail::PairwiseWorkspacePlan& workspace_plan,
    AllVsAllWorkerWorkspace* serial_workspace,
    PairwiseResultSink& sink) {
  PairwiseWorkspace local_workspace;
  PairwiseWorkspace& workspace =
      serial_workspace == nullptr ? local_workspace : serial_workspace->pairwise;
  PairwiseResultRecord local_record{};
  PairwiseResultRecord& record =
      serial_workspace == nullptr ? local_record : serial_workspace->serial_record;

  Status status = workspace.prepare(workspace_plan);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  record.result.path.steps.reserve(workspace_plan.max_query_length +
                                   workspace_plan.max_target_length);

  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    status = compute_embedding_pair_record(pair_index, source, embeddings,
                                           structures, options, workspace,
                                           record);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
    status = sink.receive(record);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
  }
  return hikoboshi::universal::ok_status();
}

Status validate_parallel_execution(
    const hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces) noexcept {
  if (pool == nullptr || thread_count <= 1U) {
    return hikoboshi::universal::ok_status();
  }
  if (pool->thread_count() != thread_count) {
    return hikoboshi::universal::failed_precondition_status(
        "all-vs-all parallel thread count does not match pool");
  }
  if (worker_workspaces.data == nullptr ||
      worker_workspaces.size < thread_count) {
    return hikoboshi::universal::failed_precondition_status(
        "all-vs-all parallel workspaces are insufficient");
  }
  return hikoboshi::universal::ok_status();
}

#if !defined(HIKOBOSHI_ALLVSALL_SINK_STREAMING) && \
    !defined(HIKOBOSHI_BENCH_NO_DETERMINISTIC_SINK)
Status validate_parallel_record_budget(std::size_t pair_count) noexcept {
  if (sizeof(PairwiseResultRecord) == 0U ||
      pair_count >
          std::numeric_limits<std::size_t>::max() /
              sizeof(PairwiseResultRecord)) {
    return hikoboshi::universal::invalid_argument_status(
        "all-vs-all parallel record buffer size overflows");
  }
  if (pair_count * sizeof(PairwiseResultRecord) >
      kMaxParallelRecordBytes) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all parallel record buffer exceeds memory budget");
  }
  return hikoboshi::universal::ok_status();
}
#endif

Status prepare_parallel_workspaces(
    Span<AllVsAllWorkerWorkspace> worker_workspaces,
    std::size_t thread_count,
    const detail::PairwiseWorkspacePlan& workspace_plan) {
  for (std::size_t worker = 0; worker < thread_count; ++worker) {
    PerWorkerPairwiseBundleRequest request{};
    request.pairwise_plan = workspace_plan;
    request.prepare_encoder = false;
    request.max_result_step_count =
        workspace_plan.max_query_length + workspace_plan.max_target_length;
    const Status status = prepare_per_worker_pairwise_bundle(
        worker_workspaces.data[worker], request);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
  }
  return hikoboshi::universal::ok_status();
}

void release_parallel_encoder_workspaces(
    Span<AllVsAllWorkerWorkspace> worker_workspaces,
    std::size_t thread_count) {
  if (worker_workspaces.data == nullptr) {
    return;
  }
  const std::size_t count = std::min(thread_count, worker_workspaces.size);
  for (std::size_t worker = 0; worker < count; ++worker) {
    worker_workspaces.data[worker].encoder = PairwiseWorkspace{};
  }
}

bool should_parallel_encode_structures(
    const hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    std::size_t item_count) noexcept {
  return pool != nullptr && thread_count > 1U &&
         item_count >= kParallelEncodingThreshold;
}

void note_pair_list_protein_encoded() noexcept;

template <typename StructureEncodeRequest>
Status encode_structures_serial(
    const StructureEncodeRequest& request,
    std::size_t max_residue_count,
    bool note_encode,
    EncodedEmbeddingCache& cache) {
  const detail::PairwiseWorkspacePlan workspace_plan =
      structure_workspace_plan(max_residue_count, request.descriptor);
  PairwiseWorkspace encode_workspace;
  Status status = encode_workspace.prepare(workspace_plan);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  for (std::size_t index = 0; index < request.structures.size; ++index) {
    const StructureView& structure = request.structures.data[index];
    float* output = cache.slot(index);
    status = encode_structure(structure, request.descriptor, request.weights,
                              encode_workspace, output);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
    if (note_encode) {
      note_pair_list_protein_encoded();
    }
  }
  return hikoboshi::universal::ok_status();
}

template <typename StructureEncodeRequest>
Status encode_structures_parallel(
    const StructureEncodeRequest& request,
    bool note_encode,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces,
    EncodedEmbeddingCache& cache) {
  Status status =
      validate_parallel_execution(pool, thread_count, worker_workspaces);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  Status encode_status = hikoboshi::universal::ok_status();
  try {
    pool->parallel_for(0, request.structures.size,
                       [&](std::size_t worker_id,
                           std::size_t begin,
                           std::size_t end) {
      if (worker_id >= worker_workspaces.size) {
        throw PairwiseStatusFailure{
            hikoboshi::universal::internal_error_status(
                "all-vs-all encoder worker id exceeded workspace count")};
      }
      PairwiseWorkspace& workspace =
          worker_workspaces.data[worker_id].encoder;
      std::size_t worker_max_residue_count = 0;
      for (std::size_t index = begin; index < end; ++index) {
        worker_max_residue_count =
            std::max(worker_max_residue_count,
                     request.structures.data[index].residue_count);
      }
      const detail::PairwiseWorkspacePlan workspace_plan =
          structure_workspace_plan(worker_max_residue_count,
                                   request.descriptor);
      const Status prepare_status = workspace.prepare(workspace_plan);
      if (!hikoboshi::universal::is_ok(prepare_status)) {
        throw PairwiseStatusFailure{prepare_status};
      }
      for (std::size_t index = begin; index < end; ++index) {
        const StructureView& structure = request.structures.data[index];
        float* output = cache.slot(index);
        const Status encode_status =
            encode_structure(structure, request.descriptor, request.weights,
                             workspace, output);
        if (!hikoboshi::universal::is_ok(encode_status)) {
          throw PairwiseStatusFailure{encode_status};
        }
        if (note_encode) {
          note_pair_list_protein_encoded();
        }
      }
    });
  } catch (const PairwiseStatusFailure& failure) {
    encode_status = failure.status;
  } catch (const std::bad_alloc&) {
    encode_status = hikoboshi::universal::unavailable_status(
        "all-vs-all parallel structure encoding allocation failed");
  }
  release_parallel_encoder_workspaces(worker_workspaces, thread_count);
  return encode_status;
}

Status run_parallel_embedding_pairs(
    Span<const EmbeddingView> embeddings,
    Span<const StructureView> structures,
    const AllVsAllOptions& options,
    std::size_t pair_count,
    const detail::PairwiseWorkspacePlan& workspace_plan,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces,
    PairwiseResultSink& sink) {
#if defined(HIKOBOSHI_BENCH_NO_DETERMINISTIC_SINK)
  Status status =
      validate_parallel_execution(pool, thread_count, worker_workspaces);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  std::vector<PairDispatchItem> dispatch_order;
  std::vector<PairTile> dispatch_tiles;
  const bool tile_dispatch =
      kQueryTiledEnabled &&
      pair_count > detail::kQueryTiledPairThreshold;
  try {
    if (tile_dispatch) {
      const std::size_t q_tile =
          resolve_query_tile_size(embeddings, thread_count);
      build_query_tiled_dispatch_tiles(embeddings, options, q_tile,
                                       /*target_tile=*/0U,
                                       /*sort_cost_descending=*/true,
                                       dispatch_tiles);
    } else if (pair_count > kCostAwarePairSchedulingThreshold) {
      dispatch_order.resize(pair_count);
      build_cost_descending_pair_dispatch_order(embeddings,
                                                options,
                                                dispatch_order);
    }
    status =
        prepare_parallel_workspaces(worker_workspaces, thread_count,
                                    workspace_plan);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all parallel preparation allocation failed");
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  try {
    std::atomic<std::size_t> next_dispatch_index{0};
    std::atomic<std::size_t> next_tile_index{0};
    std::mutex sink_mutex;
    const auto compute_and_emit = [&](std::size_t pair_index,
                                      PairwiseWorkspace& workspace,
                                      PairwiseResultRecord& record) {
      Status pair_status =
          compute_embedding_pair_record(pair_index, embeddings, structures,
                                        options, workspace, record);
      if (!hikoboshi::universal::is_ok(pair_status)) {
        throw PairwiseStatusFailure{pair_status};
      }
      std::lock_guard<std::mutex> lock(sink_mutex);
      pair_status = sink.receive(record);
      if (!hikoboshi::universal::is_ok(pair_status)) {
        throw PairwiseStatusFailure{pair_status};
      }
    };

    pool->parallel_for(0, pair_count, [&](std::size_t worker_id,
                                          std::size_t begin,
                                          std::size_t end) {
      if (worker_id >= worker_workspaces.size) {
        throw PairwiseStatusFailure{
            hikoboshi::universal::internal_error_status(
                "all-vs-all worker id exceeded workspace count")};
      }
      AllVsAllWorkerWorkspace& worker_workspace =
          worker_workspaces.data[worker_id];
      PairwiseWorkspace& workspace = worker_workspace.pairwise;
      PairwiseResultRecord& record = worker_workspace.serial_record;
      if (!dispatch_tiles.empty()) {
        for (;;) {
          const std::size_t tile_index =
              next_tile_index.fetch_add(1U, std::memory_order_relaxed);
          if (tile_index >= dispatch_tiles.size()) {
            break;
          }
          detail::iterate_pair_tile(
              dispatch_tiles[tile_index], embeddings.size,
              options.include_self,
              [&](std::size_t pair_index, std::size_t /*q*/,
                  std::size_t /*t*/) {
                compute_and_emit(pair_index, workspace, record);
              });
        }
      } else if (dispatch_order.empty()) {
        for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
          compute_and_emit(pair_index, workspace, record);
        }
      } else {
        for (;;) {
          const std::size_t dispatch_index =
              next_dispatch_index.fetch_add(1U, std::memory_order_relaxed);
          if (dispatch_index >= dispatch_order.size()) {
            break;
          }
          compute_and_emit(dispatch_order[dispatch_index].pair_index,
                           workspace,
                           record);
        }
      }
    });
  } catch (const PairwiseStatusFailure& failure) {
    return failure.status;
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all parallel pair result allocation failed");
  }

  return hikoboshi::universal::ok_status();
#elif defined(HIKOBOSHI_ALLVSALL_SINK_STREAMING)
  Status status =
      validate_parallel_execution(pool, thread_count, worker_workspaces);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  // Streaming sink design history:
  //
  // - Legacy bounded-ring path (kept here behind HIKOBOSHI_STREAMING_SINK_LEGACY=1
  //   until ssf2 validates the replacement): a single ring of K slots drains
  //   one slot at a time in pair_id order. The submit order must stay within K
  //   of the drain front, so dispatch must be pair_id-ascending or it
  //   deadlocks. See bench/STREAMING_SINK_DEADLOCK_DIAGNOSIS.md (p50).
  // - New per-worker-buffers + sequencer path (default; ssf1): each worker
  //   writes into its own SPSC ring; a dedicated sequencer thread emits in
  //   canonical pair_id order. A slow head pair on one worker no longer pins
  //   ring slots for other workers, so utilization is no longer capped at the
  //   in-flight-spread / K ratio. See p58 SCALING_DIAGNOSIS.md.
  //
  // Both paths share the same dispatch order (pair_id-ascending). For the new
  // sequencer path the order is a courtesy that keeps each worker ring sorted
  // (per-worker dispatch is monotonic), not a deadlock-avoidance invariant.
  // The query_tiled scheduler (p48) preserves the courtesy by emitting tiles
  // in ascending (Qi, Tj) lex order with Q_tile=1 forced, so each tile is one
  // query row × T_tile target chunk and tile order matches pair_id order.
  std::vector<PairDispatchItem> dispatch_order;
  std::vector<PairTile> dispatch_tiles;
  const bool tile_dispatch =
      kQueryTiledEnabled &&
      pair_count > detail::kQueryTiledPairThreshold;
  try {
    if (tile_dispatch) {
      build_query_tiled_dispatch_tiles(
          embeddings, options, /*query_tile=*/1U,
          /*target_tile=*/kStreamingQueryTileTargetSize,
          /*sort_cost_descending=*/false, dispatch_tiles);
    }
    status =
        prepare_parallel_workspaces(worker_workspaces, thread_count,
                                    workspace_plan);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all parallel preparation allocation failed");
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  const std::size_t max_step_count =
      workspace_plan.max_query_length + workspace_plan.max_target_length;

  std::atomic<std::size_t> next_dispatch_index{0};
  std::atomic<std::size_t> next_tile_index{0};

  const auto resolve_pair_index = [&](std::size_t dispatch_index) noexcept {
    return dispatch_order.empty()
               ? dispatch_index
               : dispatch_order[dispatch_index].pair_index;
  };

  // Tile-cursor inline iterator. Holds the worker's current tile and the
  // (q, t) coordinate within it; advances pair-by-pair. Allocation-free.
  struct TileCursor {
    PairTile tile{};
    bool active = false;
    std::size_t q = 0;
    std::size_t t = 0;
  };

  const auto reset_cursor_at_query =
      [include_self = options.include_self](TileCursor& cursor,
                                            std::size_t q) noexcept {
        const std::size_t lower_bound =
            include_self ? q
                         : (q == std::numeric_limits<std::size_t>::max()
                                ? q
                                : q + 1U);
        cursor.t = lower_bound > cursor.tile.target_begin
                       ? lower_bound
                       : cursor.tile.target_begin;
      };

  const auto advance_cursor_to_valid = [&](TileCursor& cursor) noexcept {
    while (cursor.active) {
      if (cursor.q >= cursor.tile.query_end) {
        cursor.active = false;
        return;
      }
      if (cursor.t < cursor.tile.target_end) {
        return;
      }
      ++cursor.q;
      if (cursor.q >= cursor.tile.query_end) {
        cursor.active = false;
        return;
      }
      reset_cursor_at_query(cursor, cursor.q);
    }
  };

  const auto load_next_tile = [&](TileCursor& cursor) noexcept {
    while (true) {
      const std::size_t tile_index =
          next_tile_index.fetch_add(1U, std::memory_order_relaxed);
      if (tile_index >= dispatch_tiles.size()) {
        cursor.active = false;
        return false;
      }
      cursor.tile = dispatch_tiles[tile_index];
      cursor.q = cursor.tile.query_begin;
      reset_cursor_at_query(cursor, cursor.q);
      cursor.active = true;
      advance_cursor_to_valid(cursor);
      if (cursor.active) {
        return true;
      }
    }
  };

  // Dispatch fetcher closure: returns the next pair_id this worker should
  // compute (in pair_id-ascending order, modulo the tile/dispatch_order
  // knobs). Each worker constructs its own copy with its own per-worker
  // `cursor`; both streaming paths use the same body.
  auto make_fetch_next_pair_index = [&](TileCursor& cursor) {
    return [&](std::size_t& pair_index_out) -> bool {
      if (tile_dispatch) {
        if (!cursor.active && !load_next_tile(cursor)) {
          return false;
        }
        pair_index_out = detail::ij_to_pair_index(
            cursor.q, cursor.t, embeddings.size, options.include_self);
        ++cursor.t;
        advance_cursor_to_valid(cursor);
        return true;
      }
      const std::size_t dispatch_index =
          next_dispatch_index.fetch_add(1U, std::memory_order_relaxed);
      if (dispatch_index >= pair_count) {
        return false;
      }
      pair_index_out = resolve_pair_index(dispatch_index);
      return true;
    };
  };

  if (detail::StreamingSequencerBuffer::legacy_sink_enabled_from_env()) {
    // Legacy ring-buffer streaming sink path. Kept until ssf2 validates the
    // per-worker-buffers + sequencer replacement; remove after ssf2.
    std::unique_ptr<detail::StreamingRecordBuffer> buffer;
    try {
      buffer = std::make_unique<detail::StreamingRecordBuffer>(
          detail::StreamingRecordBuffer::resolve_slot_count(pair_count),
          max_step_count, sink);
    } catch (const std::bad_alloc&) {
      return hikoboshi::universal::unavailable_status(
          "all-vs-all streaming sink ring buffer allocation failed");
    }

    Status drain_status = hikoboshi::universal::ok_status();

    try {
      pool->parallel_for(
          0, thread_count, [&](std::size_t worker_id,
                               std::size_t /*begin*/, std::size_t /*end*/) {
            if (worker_id >= worker_workspaces.size) {
              buffer->request_abort();
              throw PairwiseStatusFailure{
                  hikoboshi::universal::internal_error_status(
                      "all-vs-all streaming worker id exceeded workspace count")};
            }
            AllVsAllWorkerWorkspace& worker_workspace =
                worker_workspaces.data[worker_id];
            PairwiseWorkspace& workspace = worker_workspace.pairwise;
            PairwiseResultRecord& record = worker_workspace.serial_record;
            TileCursor cursor;
            auto fetch_next_pair_index = make_fetch_next_pair_index(cursor);

            if (worker_id == 0U) {
              // Driver thread: dual-role drainer + opportunistic compute.
              // Worker 0 also computes pairs to preserve speedup at small
              // thread_count. Backpressure is handled by interleaving drain
              // steps with try_submit so worker 0 never blocks on a slot it
              // would itself need to drain.
              std::size_t next_to_flush = 0U;
              bool have_pending_record = false;
              std::size_t pending_pair_index = 0U;
              Status local_drain_status = hikoboshi::universal::ok_status();
              while (next_to_flush < pair_count) {
                if (buffer->aborted()) {
                  break;
                }
                // Drain everything ready without blocking.
                while (next_to_flush < pair_count) {
                  bool drained = false;
                  local_drain_status =
                      buffer->try_drain_step(next_to_flush, drained);
                  if (!hikoboshi::universal::is_ok(local_drain_status)) {
                    break;
                  }
                  if (!drained) {
                    break;
                  }
                  ++next_to_flush;
                }
                if (!hikoboshi::universal::is_ok(local_drain_status)) {
                  break;
                }
                if (next_to_flush >= pair_count) {
                  break;
                }
                if (have_pending_record) {
                  if (buffer->try_submit(pending_pair_index, record)) {
                    have_pending_record = false;
                    continue;
                  }
                  // Slot still busy; advance drain by one slot (blocking).
                  local_drain_status =
                      buffer->drain_step_blocking(next_to_flush);
                  if (!hikoboshi::universal::is_ok(local_drain_status)) {
                    break;
                  }
                  ++next_to_flush;
                  continue;
                }
                std::size_t pair_index = 0U;
                if (fetch_next_pair_index(pair_index)) {
                  const Status pair_status = compute_embedding_pair_record(
                      pair_index, embeddings, structures, options, workspace,
                      record);
                  if (!hikoboshi::universal::is_ok(pair_status)) {
                    buffer->request_abort();
                    throw PairwiseStatusFailure{pair_status};
                  }
                  if (!buffer->try_submit(pair_index, record)) {
                    have_pending_record = true;
                    pending_pair_index = pair_index;
                  }
                } else {
                  // No more compute; advance drain (blocking) until done.
                  local_drain_status =
                      buffer->drain_step_blocking(next_to_flush);
                  if (!hikoboshi::universal::is_ok(local_drain_status)) {
                    break;
                  }
                  ++next_to_flush;
                }
              }
              drain_status = local_drain_status;
              return;
            }
            // Compute-only worker.
            while (!buffer->aborted()) {
              std::size_t pair_index = 0U;
              if (!fetch_next_pair_index(pair_index)) {
                break;
              }
              const Status pair_status = compute_embedding_pair_record(
                  pair_index, embeddings, structures, options, workspace,
                  record);
              if (!hikoboshi::universal::is_ok(pair_status)) {
                buffer->request_abort();
                throw PairwiseStatusFailure{pair_status};
              }
              buffer->submit(pair_index, record);
            }
          });
    } catch (const PairwiseStatusFailure& failure) {
      return failure.status;
    } catch (const std::bad_alloc&) {
      buffer->request_abort();
      return hikoboshi::universal::unavailable_status(
          "all-vs-all streaming pair execution allocation failed");
    }

    return drain_status;
  }

  // Default streaming sink: per-worker buffers + sequencer thread (ssf1).
  // Worker progress is decoupled from drain progress: a slow head pair on
  // any one worker cannot pin slots for other workers, eliminating the
  // ring-rotation backpressure that capped utilization at the in-flight-
  // spread / K ratio in the legacy path. See p58 SCALING_DIAGNOSIS.md.
  detail::StreamingSequencerBuffer::Config sequencer_config;
  sequencer_config.worker_count = thread_count;
  sequencer_config.total_pair_count = pair_count;
  sequencer_config.per_worker_capacity =
      detail::kDefaultStreamingSequencerPerWorkerCapacity;
  sequencer_config.max_result_step_count = max_step_count;
  sequencer_config.max_in_flight_bytes =
      detail::StreamingSequencerBuffer::resolve_max_in_flight_bytes_from_env();

  std::unique_ptr<detail::StreamingSequencerBuffer> sequencer;
  try {
    sequencer = std::make_unique<detail::StreamingSequencerBuffer>(
        sequencer_config, sink);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all streaming sink sequencer allocation failed");
  }

  try {
    pool->parallel_for(
        0, thread_count, [&](std::size_t worker_id,
                             std::size_t /*begin*/, std::size_t /*end*/) {
          if (worker_id >= worker_workspaces.size) {
            sequencer->request_abort();
            throw PairwiseStatusFailure{
                hikoboshi::universal::internal_error_status(
                    "all-vs-all streaming worker id exceeded workspace count")};
          }
          AllVsAllWorkerWorkspace& worker_workspace =
              worker_workspaces.data[worker_id];
          PairwiseWorkspace& workspace = worker_workspace.pairwise;
          PairwiseResultRecord& record = worker_workspace.serial_record;
          TileCursor cursor;
          auto fetch_next_pair_index = make_fetch_next_pair_index(cursor);

          while (!sequencer->aborted()) {
            std::size_t pair_index = 0U;
            if (!fetch_next_pair_index(pair_index)) {
              break;
            }
            const Status pair_status = compute_embedding_pair_record(
                pair_index, embeddings, structures, options, workspace,
                record);
            if (!hikoboshi::universal::is_ok(pair_status)) {
              sequencer->request_abort();
              throw PairwiseStatusFailure{pair_status};
            }
            sequencer->submit(worker_id, pair_index, record);
          }
        });
  } catch (const PairwiseStatusFailure& failure) {
    sequencer->request_abort();
    Status drain_status = sequencer->wait_drain_complete();
    (void)drain_status;
    return failure.status;
  } catch (const std::bad_alloc&) {
    sequencer->request_abort();
    Status drain_status = sequencer->wait_drain_complete();
    (void)drain_status;
    return hikoboshi::universal::unavailable_status(
        "all-vs-all streaming pair execution allocation failed");
  }

  return sequencer->wait_drain_complete();
#else
  Status status =
      validate_parallel_execution(pool, thread_count, worker_workspaces);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = validate_parallel_record_budget(pair_count);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  std::vector<PairwiseResultRecord> records;
  std::vector<PairDispatchItem> dispatch_order;
  std::vector<PairTile> dispatch_tiles;
  const bool tile_dispatch =
      kQueryTiledEnabled &&
      pair_count > detail::kQueryTiledPairThreshold;
  try {
    records.resize(pair_count);
    if (tile_dispatch) {
      const std::size_t q_tile =
          resolve_query_tile_size(embeddings, thread_count);
      build_query_tiled_dispatch_tiles(embeddings, options, q_tile,
                                       /*target_tile=*/0U,
                                       /*sort_cost_descending=*/true,
                                       dispatch_tiles);
    } else if (pair_count > kCostAwarePairSchedulingThreshold) {
      dispatch_order.resize(pair_count);
      build_cost_descending_pair_dispatch_order(embeddings,
                                                options,
                                                dispatch_order);
    }
    status =
        prepare_parallel_workspaces(worker_workspaces, thread_count,
                                    workspace_plan);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all parallel preparation allocation failed");
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  try {
    std::atomic<std::size_t> next_dispatch_index{0};
    std::atomic<std::size_t> next_tile_index{0};
    pool->parallel_for(0, pair_count, [&](std::size_t worker_id,
                                          std::size_t begin,
                                          std::size_t end) {
      if (worker_id >= worker_workspaces.size) {
        throw PairwiseStatusFailure{
            hikoboshi::universal::internal_error_status(
                "all-vs-all worker id exceeded workspace count")};
      }
      PairwiseWorkspace& workspace =
          worker_workspaces.data[worker_id].pairwise;
      if (!dispatch_tiles.empty()) {
        for (;;) {
          const std::size_t tile_index =
              next_tile_index.fetch_add(1U, std::memory_order_relaxed);
          if (tile_index >= dispatch_tiles.size()) {
            break;
          }
          detail::iterate_pair_tile(
              dispatch_tiles[tile_index], embeddings.size,
              options.include_self,
              [&](std::size_t pair_index, std::size_t /*q*/,
                  std::size_t /*t*/) {
                const Status pair_status = compute_embedding_pair_record(
                    pair_index, embeddings, structures, options, workspace,
                    records[pair_index]);
                if (!hikoboshi::universal::is_ok(pair_status)) {
                  throw PairwiseStatusFailure{pair_status};
                }
              });
        }
      } else if (dispatch_order.empty()) {
        for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
          const Status pair_status =
              compute_embedding_pair_record(pair_index, embeddings, structures,
                                            options, workspace,
                                            records[pair_index]);
          if (!hikoboshi::universal::is_ok(pair_status)) {
            throw PairwiseStatusFailure{pair_status};
          }
        }
      } else {
        for (;;) {
          const std::size_t dispatch_index =
              next_dispatch_index.fetch_add(1U, std::memory_order_relaxed);
          if (dispatch_index >= dispatch_order.size()) {
            break;
          }
          const std::size_t pair_index =
              dispatch_order[dispatch_index].pair_index;
          const Status pair_status =
              compute_embedding_pair_record(pair_index, embeddings, structures,
                                            options, workspace,
                                            records[pair_index]);
          if (!hikoboshi::universal::is_ok(pair_status)) {
            throw PairwiseStatusFailure{pair_status};
          }
        }
      }
    });
  } catch (const PairwiseStatusFailure& failure) {
    return failure.status;
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all parallel pair result allocation failed");
  }

  BoundedRecordStagingResult staging_result{};
  return bounded_record_staging(
      {{records.data(), records.size()}, {nullptr, 0}, &sink},
      staging_result);
#endif
}

Status run_parallel_resolved_list_pairs(
    const PairSource& source,
    Span<const EmbeddingView> embeddings,
    Span<const StructureView> structures,
    const AllVsAllOptions& options,
    std::size_t pair_count,
    const detail::PairwiseWorkspacePlan& workspace_plan,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces,
    PairwiseResultSink& sink) {
  Status status =
      validate_parallel_execution(pool, thread_count, worker_workspaces);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  std::vector<PairwiseResultRecord> records;
  try {
    records.resize(pair_count);
    status =
        prepare_parallel_workspaces(worker_workspaces, thread_count,
                                    workspace_plan);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "pair-list parallel preparation allocation failed");
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  try {
    pool->parallel_for(0, pair_count, [&](std::size_t worker_id,
                                          std::size_t begin,
                                          std::size_t end) {
      if (worker_id >= worker_workspaces.size) {
        throw PairwiseStatusFailure{
            hikoboshi::universal::internal_error_status(
                "pair-list worker id exceeded workspace count")};
      }
      PairwiseWorkspace& workspace =
          worker_workspaces.data[worker_id].pairwise;
      for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
        const Status pair_status =
            compute_embedding_pair_record(pair_index, source, embeddings,
                                          structures, options, workspace,
                                          records[pair_index]);
        if (!hikoboshi::universal::is_ok(pair_status)) {
          throw PairwiseStatusFailure{pair_status};
        }
      }
    });
  } catch (const PairwiseStatusFailure& failure) {
    return failure.status;
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "pair-list parallel pair result allocation failed");
  }

  BoundedRecordStagingResult staging_result{};
  return bounded_record_staging(
      {{records.data(), records.size()}, {nullptr, 0}, &sink},
      staging_result);
}

// Shared per-pair execution dispatcher for all-vs-all and pair-list (npc1b).
//
// `source` is the only behavioral difference between the two callers (see
// `PairSource`). The cost-aware tiling + parallel scheduling path remains
// symmetric-only. A pair-list (`ResolvedList`) source may still compute pairs
// on the pool; completed records are staged by original pair index and drained
// to `sink` in caller input order.
Status dispatch_pairs(
    const PairSource& source,
    Span<const EmbeddingView> embeddings,
    Span<const StructureView> structures,
    const AllVsAllOptions& options,
    std::size_t pair_count,
    const detail::PairwiseWorkspacePlan& workspace_plan,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces,
    PairwiseResultSink& sink) {
  const bool parallel_eligible = pool != nullptr && thread_count > 1U &&
                                 pair_count >= kParallelPairThreshold;
  const bool run_symmetric_parallel =
      source.symmetric() && parallel_eligible;
  const bool run_resolved_list_parallel =
      !source.symmetric() && parallel_eligible;
  if (!run_symmetric_parallel && !run_resolved_list_parallel) {
    AllVsAllWorkerWorkspace* serial_workspace =
        worker_workspaces.data != nullptr && worker_workspaces.size >= 1U
            ? &worker_workspaces.data[0]
            : nullptr;
    return run_serial_embedding_pairs(source, embeddings, structures, options,
                                      pair_count, workspace_plan,
                                      serial_workspace, sink);
  }
  if (run_symmetric_parallel) {
    return run_parallel_embedding_pairs(embeddings, structures, options,
                                        pair_count, workspace_plan, pool,
                                        thread_count, worker_workspaces, sink);
  }
  return run_parallel_resolved_list_pairs(
      source, embeddings, structures, options, pair_count, workspace_plan,
      pool, thread_count, worker_workspaces, sink);
}

// Reject a pair-list index pair that references an item outside the
// resolved source span. The api/engine layer resolves caller IDs to
// indices, but the algorithm layer revalidates so a malformed direct
// caller cannot read out of bounds inside `dispatch_pairs`.
Status validate_pair_list_indices(
    Span<const std::pair<std::size_t, std::size_t>> pairs,
    std::size_t item_count) noexcept {
  for (std::size_t k = 0; k < pairs.size; ++k) {
    if (pairs.data[k].first >= item_count ||
        pairs.data[k].second >= item_count) {
      return hikoboshi::universal::invalid_argument_status(
          "pair-list pair index is out of range for the source span");
    }
  }
  return hikoboshi::universal::ok_status();
}

struct PairListAxisMaxima {
  std::size_t max_query_length = 0;
  std::size_t max_target_length = 0;
};

void apply_pair_list_workspace_hints(std::size_t max_query_length,
                                     std::size_t max_target_length,
                                     PairListAxisMaxima& maxima) noexcept {
  maxima.max_query_length =
      std::max(maxima.max_query_length, max_query_length);
  maxima.max_target_length =
      std::max(maxima.max_target_length, max_target_length);
}

template <typename LengthAt>
PairListAxisMaxima pair_list_axis_maxima(
    Span<const std::pair<std::size_t, std::size_t>> pairs,
    LengthAt&& length_at) noexcept {
  PairListAxisMaxima maxima{};
  for (std::size_t index = 0; index < pairs.size; ++index) {
    const std::pair<std::size_t, std::size_t>& pair = pairs.data[index];
    maxima.max_query_length =
        std::max(maxima.max_query_length, length_at(pair.first));
    maxima.max_target_length =
        std::max(maxima.max_target_length, length_at(pair.second));
  }
  return maxima;
}

// Debug-only encode-once instrumentation for the pair-list routes (npc1b
// packet section E). `run_pair_list_structures` / `run_pair_list_sequences`
// bump this counter once per protein encoded; the encode-once invariant
// holds iff the post-invocation count equals the unique-protein count, and
// `pair_list_cache_hit` asserts exactly that. The counter is compiled only
// into debug (`!NDEBUG`) builds; in release builds it does not exist and
// `note_pair_list_protein_encoded` expands to nothing.
#ifndef NDEBUG
// Atomic so the parallel sequence/pair-list encode can bump it without a
// data race. The counter is only read after the encode parallel_for joins,
// so relaxed ordering is sufficient and the post-join count is exact.
std::atomic<std::size_t> g_pair_list_encode_count{0};
#endif

void note_pair_list_protein_encoded() noexcept {
#ifndef NDEBUG
  g_pair_list_encode_count.fetch_add(1U, std::memory_order_relaxed);
#endif
}

// Sequence-route encode constants shared by all-vs-all and pair-list. Each
// raw AA token span is wrapped as [<cls>, aa..., <eos>] for the ESM2 forward
// pass and the CLS/EOS rows are stripped from the residue-level
// EmbeddingView, matching the PyTorch training pipeline (fe2). Without this
// wrap the per-residue embeddings drift from the PyTorch reference.
constexpr std::int32_t kEsm2ClsTokenId = 26;
constexpr std::int32_t kEsm2EosTokenId = 27;
constexpr std::size_t kEsm2SpecialTokenOverhead = 2U;

bool should_parallel_encode_sequences(
    const hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    std::size_t item_count) noexcept {
  return pool != nullptr && thread_count > 1U &&
         item_count >= kParallelEncodingThreshold;
}

// Encode one sequence into `encoded_storage[index]` and point
// `encoded_embeddings[index]` at its residue rows. `wrapped_scratch` is the
// caller's token buffer; in the parallel path it MUST be worker-local so no
// two threads share the wrap buffer. `note_encode` bumps the pair-list
// encode-once counter (only the pair-list routes pass true).
Status encode_one_sequence(
    std::size_t index,
    const AllVsAllSequenceEntry& entry,
    const hikoboshi::universal::WeightsView& weights_view,
    const hikoboshi::modules::Esm2Descriptor& descriptor,
    std::size_t hidden,
    bool note_encode,
    std::vector<std::int32_t>& wrapped_scratch,
    std::vector<std::vector<float>>& encoded_storage,
    std::vector<EmbeddingView>& encoded_embeddings) {
  const std::size_t wrapped_token_count =
      entry.token_ids.size + kEsm2SpecialTokenOverhead;
  try {
    wrapped_scratch.assign(wrapped_token_count, 0);
    encoded_storage[index].assign(wrapped_token_count * hidden, 0.0F);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "sequence-route per-sequence embedding allocation failed");
  }
  wrapped_scratch[0] = kEsm2ClsTokenId;
  std::copy(entry.token_ids.data,
            entry.token_ids.data + entry.token_ids.size,
            wrapped_scratch.begin() + 1);
  wrapped_scratch[wrapped_token_count - 1] = kEsm2EosTokenId;
  const Status status = hikoboshi::algorithms::detail::encode_esm2_sequence(
      weights_view, descriptor,
      hikoboshi::universal::Span<const std::int32_t>{
          wrapped_scratch.data(), wrapped_token_count},
      encoded_storage[index].data());
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  if (note_encode) {
    note_pair_list_protein_encoded();
  }
  EmbeddingView view{};
  view.residue_count = entry.token_ids.size;
  view.dimension = hidden;
  // Skip the leading CLS row; the trailing EOS row is excluded via the
  // residue-count-derived span length.
  view.values = {encoded_storage[index].data() + hidden,
                 entry.token_ids.size * hidden};
  encoded_embeddings[index] = view;
  return hikoboshi::universal::ok_status();
}

Status encode_sequences_serial(
    Span<const AllVsAllSequenceEntry> sequences,
    const hikoboshi::universal::WeightsView& weights_view,
    const hikoboshi::modules::Esm2Descriptor& descriptor,
    std::size_t hidden,
    bool note_encode,
    std::vector<std::vector<float>>& encoded_storage,
    std::vector<EmbeddingView>& encoded_embeddings) {
  std::vector<std::int32_t> wrapped_scratch;
  for (std::size_t index = 0; index < sequences.size; ++index) {
    const Status status = encode_one_sequence(
        index, sequences.data[index], weights_view, descriptor, hidden,
        note_encode, wrapped_scratch, encoded_storage, encoded_embeddings);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
  }
  return hikoboshi::universal::ok_status();
}

// Mirror of `encode_structures_parallel` for the ESM2 sequence route.
// `encode_esm2_sequence` self-allocates its internal forward-pass workspace,
// so no shared encoder workspace is required; each worker only needs a
// thread-local token wrap buffer, and writes into the independent
// `encoded_storage[index]` slot. `worker_workspaces` is validated for parity
// with the structure route (the engine provides one bundle per worker for
// the downstream align stage). A serial fallback is kept in the run
// functions for `pool == nullptr || thread_count <= 1`.
Status encode_sequences_parallel(
    Span<const AllVsAllSequenceEntry> sequences,
    const hikoboshi::universal::WeightsView& weights_view,
    const hikoboshi::modules::Esm2Descriptor& descriptor,
    std::size_t hidden,
    bool note_encode,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces,
    std::vector<std::vector<float>>& encoded_storage,
    std::vector<EmbeddingView>& encoded_embeddings) {
  const Status validate =
      validate_parallel_execution(pool, thread_count, worker_workspaces);
  if (!hikoboshi::universal::is_ok(validate)) {
    return validate;
  }
  try {
    pool->parallel_for(0, sequences.size,
                       [&](std::size_t /*worker_id*/,
                           std::size_t begin,
                           std::size_t end) {
      // Worker-local token scratch — never shared across threads.
      std::vector<std::int32_t> wrapped_scratch;
      for (std::size_t index = begin; index < end; ++index) {
        const Status status = encode_one_sequence(
            index, sequences.data[index], weights_view, descriptor, hidden,
            note_encode, wrapped_scratch, encoded_storage,
            encoded_embeddings);
        if (!hikoboshi::universal::is_ok(status)) {
          throw PairwiseStatusFailure{status};
        }
      }
    });
  } catch (const PairwiseStatusFailure& failure) {
    return failure.status;
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all parallel sequence encoding allocation failed");
  }
  return hikoboshi::universal::ok_status();
}

}  // namespace

#ifndef NDEBUG
void pair_list_debug_reset_encode_count() noexcept {
  g_pair_list_encode_count.store(0U, std::memory_order_relaxed);
}

std::size_t pair_list_debug_encode_count() noexcept {
  return g_pair_list_encode_count.load(std::memory_order_relaxed);
}
#endif

Status prepare_per_worker_pairwise_bundle(
    detail::PerWorkerPairwiseBundle& bundle,
    const PerWorkerPairwiseBundleRequest& request) {
  try {
    if (request.prepare_pairwise) {
      const Status status = bundle.pairwise.prepare(request.pairwise_plan);
      if (!hikoboshi::universal::is_ok(status)) {
        return status;
      }
      const std::size_t reserve_count =
          request.max_result_step_count != 0U
              ? request.max_result_step_count
              : request.pairwise_plan.max_query_length +
                    request.pairwise_plan.max_target_length;
      bundle.serial_record.result.path.steps.reserve(reserve_count);
    }
    if (request.prepare_encoder) {
      const Status status = bundle.encoder.prepare(request.encoder_plan);
      if (!hikoboshi::universal::is_ok(status)) {
        return status;
      }
    }
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all per-worker bundle preparation allocation failed");
  }
  return hikoboshi::universal::ok_status();
}

Status run_per_worker_pairwise_bundle_pair(
    const PerWorkerPairwiseBundlePairRequest& request,
    detail::PerWorkerPairwiseBundle& bundle,
    PairwiseResultRecord& record) {
  if (request.options == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "all-vs-all per-worker pair request requires options");
  }
  return compute_embedding_pair_record(request.pair_index, request.embeddings,
                                       request.structures, *request.options,
                                       bundle.pairwise, record);
}

Status bounded_record_staging(const BoundedRecordStagingRequest& request,
                              BoundedRecordStagingResult& result) {
  result = {};
  if (request.sink == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "bounded record staging requires a sink");
  }
  if (request.ordered_records.size != 0U &&
      request.ordered_records.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "bounded record staging input records are invalid");
  }
  if (request.output_buffer.size != 0U &&
      request.output_buffer.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "bounded record staging output buffer is invalid");
  }

  const auto flush = [&](std::size_t count) -> Status {
    for (std::size_t index = 0; index < count; ++index) {
      const Status status = request.sink->receive(
          *request.output_buffer.data[index]);
      if (!hikoboshi::universal::is_ok(status)) {
        return status;
      }
      ++result.emitted_count;
    }
    if (count != 0U) {
      ++result.flush_count;
    }
    return hikoboshi::universal::ok_status();
  };

  if (request.output_buffer.size == 0U) {
    for (std::size_t index = 0; index < request.ordered_records.size; ++index) {
      const Status status =
          request.sink->receive(request.ordered_records.data[index]);
      if (!hikoboshi::universal::is_ok(status)) {
        return status;
      }
      ++result.emitted_count;
    }
    return hikoboshi::universal::ok_status();
  }

  std::size_t staged_count = 0;
  for (std::size_t index = 0; index < request.ordered_records.size; ++index) {
    request.output_buffer.data[staged_count] =
        &request.ordered_records.data[index];
    ++staged_count;
    if (staged_count == request.output_buffer.size) {
      const Status status = flush(staged_count);
      if (!hikoboshi::universal::is_ok(status)) {
        return status;
      }
      staged_count = 0;
    }
  }
  return flush(staged_count);
}

Status run_all_vs_all_embeddings(const AllVsAllEmbeddingRequest& request,
                                 PairwiseResultSink& sink) {
  return run_all_vs_all_embeddings(request, sink, nullptr, 1U, {nullptr, 0});
}

Status run_all_vs_all_embeddings(
    const AllVsAllEmbeddingRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces) {
  const std::size_t pair_count = detail::symmetric_pair_count(
      request.embeddings.size, request.options.include_self);
  if (pair_count == 0U) {
    return hikoboshi::universal::ok_status();
  }
  if (request.embeddings.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all embedding input span is invalid");
  }

  std::size_t max_residue_count = 0;
  std::size_t dimension = 0;
  Status status = validate_embedding_set(request.embeddings, max_residue_count,
                                         dimension);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = validate_gap_options(request.options.pairwise);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  return dispatch_pairs(
      symmetric_pair_source(request.embeddings.size,
                            request.options.include_self),
      request.embeddings, {nullptr, 0}, request.options, pair_count,
      embedding_workspace_plan(max_residue_count, dimension,
                               request.options.soft_mode),
      pool, thread_count, worker_workspaces, sink);
}

Status run_all_vs_all_structures(const AllVsAllStructureRequest& request,
                                 PairwiseResultSink& sink) {
  return run_all_vs_all_structures(request, sink, nullptr, 1U, {nullptr, 0});
}

Status run_all_vs_all_structures(
    const AllVsAllStructureRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces) {
  const std::size_t pair_count = detail::symmetric_pair_count(
      request.structures.size, request.options.include_self);
  if (pair_count == 0U) {
    return hikoboshi::universal::ok_status();
  }
  if (request.structures.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status("all-vs-all structure input span is invalid");
  }
  if (request.weights == nullptr) {
    return hikoboshi::universal::failed_precondition_status("all-vs-all structure request requires MPNN weights");
  }

  std::size_t max_residue_count = 0;
  Status status =
      validate_structure_set(request.structures, max_residue_count);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = validate_gap_options(request.options.pairwise);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  const std::size_t embedding_dimension = request.descriptor.hidden_dimension;
  EncodedEmbeddingCache encoded_cache;
  status = encoded_cache.init(request.structures, embedding_dimension);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  std::vector<EmbeddingView> encoded_embeddings;
  try {
    encoded_embeddings.resize(request.structures.size);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all encoded embedding view allocation failed");
  }

  if (should_parallel_encode_structures(pool, thread_count,
                                        request.structures.size)) {
    status = encode_structures_parallel(request, /*note_encode=*/false, pool,
                                        thread_count, worker_workspaces,
                                        encoded_cache);
  } else {
    status = encode_structures_serial(request, max_residue_count,
                                      /*note_encode=*/false,
                                      encoded_cache);
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  for (std::size_t index = 0; index < request.structures.size; ++index) {
    encoded_embeddings[index] =
        encoded_cache.view(index, request.structures.data[index]);
  }

  return dispatch_pairs(
      symmetric_pair_source(encoded_embeddings.size(),
                            request.options.include_self),
      {encoded_embeddings.data(), encoded_embeddings.size()},
      request.structures, request.options, pair_count,
      embedding_workspace_plan(max_residue_count, embedding_dimension,
                               request.options.soft_mode),
      pool, thread_count, worker_workspaces, sink);
}

Status run_all_vs_all_sequences(const AllVsAllSequenceRequest& request,
                                PairwiseResultSink& sink) {
  return run_all_vs_all_sequences(request, sink, nullptr, 1U, {nullptr, 0});
}

Status run_all_vs_all_sequences(
    const AllVsAllSequenceRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces) {
  if (request.weights_view == nullptr) {
    return hikoboshi::universal::failed_precondition_status(
        "all-vs-all sequence request requires an ESM2 weights view");
  }
  if (request.sequences.size != 0U && request.sequences.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "all-vs-all sequence input span is invalid");
  }
  const std::size_t pair_count = detail::symmetric_pair_count(
      request.sequences.size, request.options.include_self);
  if (pair_count == 0U) {
    return hikoboshi::universal::ok_status();
  }

  std::size_t max_residue_count = 0;
  for (std::size_t index = 0; index < request.sequences.size; ++index) {
    const AllVsAllSequenceEntry& entry = request.sequences.data[index];
    if (entry.token_ids.data == nullptr || entry.token_ids.size == 0) {
      return hikoboshi::universal::invalid_argument_status(
          "all-vs-all sequence entries must have non-empty token spans");
    }
    max_residue_count = std::max(max_residue_count, entry.token_ids.size);
  }

  const Status gap_status = validate_gap_options(request.options.pairwise);
  if (!hikoboshi::universal::is_ok(gap_status)) {
    return gap_status;
  }

  const std::size_t hidden = request.descriptor.hidden_dimension;
  if (hidden == 0) {
    return hikoboshi::universal::invalid_argument_status(
        "all-vs-all sequence request descriptor has zero hidden dimension");
  }

  // Per-sequence embedding cache. Hikoboshi 0.1.0 sequence-route uses a
  // simple linear cache (one [seq_len, hidden] block per sequence); the
  // bucketed embedding storage refinement and per-worker encoder pool
  // tracked in the post-0.1 perf wave are intentionally out of scope. The
  // encode parallelizes over `pool` exactly as the structure route does:
  // each sequence is independent, so the worker fans out per-sequence into
  // its own `encoded_storage` slot with thread-local token scratch.
  std::vector<std::vector<float>> encoded_storage;
  std::vector<EmbeddingView> encoded_embeddings;
  try {
    encoded_storage.resize(request.sequences.size);
    encoded_embeddings.resize(request.sequences.size);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "all-vs-all sequence embedding cache allocation failed");
  }

  Status status = hikoboshi::universal::ok_status();
  if (should_parallel_encode_sequences(pool, thread_count,
                                       request.sequences.size)) {
    status = encode_sequences_parallel(
        request.sequences, *request.weights_view, request.descriptor, hidden,
        /*note_encode=*/false, pool, thread_count, worker_workspaces,
        encoded_storage, encoded_embeddings);
  } else {
    status = encode_sequences_serial(
        request.sequences, *request.weights_view, request.descriptor, hidden,
        /*note_encode=*/false, encoded_storage, encoded_embeddings);
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  return dispatch_pairs(
      symmetric_pair_source(encoded_embeddings.size(),
                            request.options.include_self),
      {encoded_embeddings.data(), encoded_embeddings.size()}, {nullptr, 0},
      request.options, pair_count,
      embedding_workspace_plan(max_residue_count, hidden,
                               request.options.soft_mode),
      pool, thread_count, worker_workspaces, sink);
}

// Pair-list algorithm-layer entries (npc1b). npc1a declared these as
// `Unimplemented` stubs; npc1b implements the encode-once + per-pair
// dispatch pipeline. Each `run_pair_list_*` mirrors the matching
// `run_all_vs_all_*` encode path, then drives the shared `dispatch_pairs`
// helper with a `ResolvedList` source over `request.pairs` instead of the
// symmetric enumeration, so records are emitted one-per-pair in caller
// input order.
//
// The api/engine layer owns the string-ID dedup: by the time a request
// reaches this layer, `embeddings` / `structures` / `sequences` is already
// the unique protein set and `pairs` indexes it. These routes therefore
// encode every source-span element exactly once, which is the encode-once
// invariant `pair_list_cache_hit` checks via the debug encode counter.

Status run_pair_list_embeddings(const PairListEmbeddingRequest& request,
                                PairwiseResultSink& sink) {
  const std::size_t pair_count = request.pairs.size;
  if (pair_count == 0U) {
    return hikoboshi::universal::ok_status();
  }
  if (request.pairs.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pair-list embedding pair span is invalid");
  }
  if (request.embeddings.size != 0U && request.embeddings.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pair-list embedding input span is invalid");
  }

  std::size_t max_residue_count = 0;
  std::size_t dimension = 0;
  Status status = validate_embedding_set(request.embeddings, max_residue_count,
                                         dimension);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = validate_gap_options(request.options.pairwise);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = validate_pair_list_indices(request.pairs, request.embeddings.size);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  PairListAxisMaxima axis_maxima = pair_list_axis_maxima(
      request.pairs,
      [&](std::size_t item_index) noexcept {
        return request.embeddings.data[item_index].residue_count;
      });
  apply_pair_list_workspace_hints(request.max_query_length,
                                  request.max_target_length, axis_maxima);

  // Embedding inputs are already encoded, so there is no encode-once step
  // on this route; pair-list collapses straight to per-pair dispatch.
  return dispatch_pairs(
      resolved_list_pair_source(request.pairs), request.embeddings,
      {nullptr, 0}, request.options, pair_count,
      embedding_workspace_plan(axis_maxima.max_query_length,
                               axis_maxima.max_target_length, dimension,
                               request.options.soft_mode),
      nullptr, 1U, {nullptr, 0}, sink);
}

Status run_pair_list_structures(const PairListStructureRequest& request,
                                PairwiseResultSink& sink) {
  return run_pair_list_structures(request, sink, nullptr, 1U, {nullptr, 0});
}

Status run_pair_list_structures(
    const PairListStructureRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces) {
  const std::size_t pair_count = request.pairs.size;
  if (pair_count == 0U) {
    return hikoboshi::universal::ok_status();
  }
  if (request.pairs.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pair-list structure pair span is invalid");
  }
  if (request.structures.size != 0U && request.structures.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pair-list structure input span is invalid");
  }
  if (request.weights == nullptr) {
    return hikoboshi::universal::failed_precondition_status(
        "pair-list structure request requires MPNN weights");
  }

  std::size_t max_residue_count = 0;
  Status status =
      validate_structure_set(request.structures, max_residue_count);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = validate_gap_options(request.options.pairwise);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = validate_pair_list_indices(request.pairs, request.structures.size);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  PairListAxisMaxima axis_maxima = pair_list_axis_maxima(
      request.pairs,
      [&](std::size_t item_index) noexcept {
        return request.structures.data[item_index].residue_count;
      });
  apply_pair_list_workspace_hints(request.max_query_length,
                                  request.max_target_length, axis_maxima);

  const std::size_t embedding_dimension = request.descriptor.hidden_dimension;
  EncodedEmbeddingCache encoded_cache;
  status = encoded_cache.init(request.structures, embedding_dimension);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  // Encode-once: the engine has already deduped the pair-list ID set into
  // `request.structures`, so encoding the whole span is exactly one MPNN
  // forward pass per unique protein. The encode and per-pair dispatch
  // parallelize over `pool` when eligible; dispatch stages records by input
  // pair index before draining them to the sink in caller order.
  std::vector<EmbeddingView> encoded_embeddings;
  try {
    encoded_embeddings.resize(request.structures.size);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "pair-list encoded embedding view allocation failed");
  }
  if (should_parallel_encode_structures(pool, thread_count,
                                        request.structures.size)) {
    status = encode_structures_parallel(request, /*note_encode=*/true, pool,
                                        thread_count, worker_workspaces,
                                        encoded_cache);
  } else {
    status = encode_structures_serial(request, max_residue_count,
                                      /*note_encode=*/true,
                                      encoded_cache);
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  for (std::size_t index = 0; index < request.structures.size; ++index) {
    encoded_embeddings[index] =
        encoded_cache.view(index, request.structures.data[index]);
  }

  return dispatch_pairs(
      resolved_list_pair_source(request.pairs),
      {encoded_embeddings.data(), encoded_embeddings.size()},
      request.structures, request.options, pair_count,
      embedding_workspace_plan(axis_maxima.max_query_length,
                               axis_maxima.max_target_length,
                               embedding_dimension,
                               request.options.soft_mode),
      pool, thread_count, worker_workspaces, sink);
}

Status run_pair_list_sequences(const PairListSequenceRequest& request,
                               PairwiseResultSink& sink) {
  return run_pair_list_sequences(request, sink, nullptr, 1U, {nullptr, 0});
}

Status run_pair_list_sequences(
    const PairListSequenceRequest& request,
    PairwiseResultSink& sink,
    hikoboshi::universal::detail::ThreadPool* pool,
    std::size_t thread_count,
    Span<AllVsAllWorkerWorkspace> worker_workspaces) {
  const std::size_t pair_count = request.pairs.size;
  if (pair_count == 0U) {
    return hikoboshi::universal::ok_status();
  }
  if (request.pairs.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pair-list sequence pair span is invalid");
  }
  if (request.weights_view == nullptr) {
    return hikoboshi::universal::failed_precondition_status(
        "pair-list sequence request requires an ESM2 weights view");
  }
  if (request.sequences.size != 0U && request.sequences.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pair-list sequence input span is invalid");
  }

  Status status =
      validate_pair_list_indices(request.pairs, request.sequences.size);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  std::size_t max_residue_count = 0;
  for (std::size_t index = 0; index < request.sequences.size; ++index) {
    const AllVsAllSequenceEntry& entry = request.sequences.data[index];
    if (entry.token_ids.data == nullptr || entry.token_ids.size == 0) {
      return hikoboshi::universal::invalid_argument_status(
          "pair-list sequence entries must have non-empty token spans");
    }
    max_residue_count = std::max(max_residue_count, entry.token_ids.size);
  }

  status = validate_gap_options(request.options.pairwise);
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  PairListAxisMaxima axis_maxima = pair_list_axis_maxima(
      request.pairs,
      [&](std::size_t item_index) noexcept {
        return request.sequences.data[item_index].token_ids.size;
      });
  apply_pair_list_workspace_hints(request.max_query_length,
                                  request.max_target_length, axis_maxima);

  const std::size_t hidden = request.descriptor.hidden_dimension;
  if (hidden == 0) {
    return hikoboshi::universal::invalid_argument_status(
        "pair-list sequence request descriptor has zero hidden dimension");
  }

  std::vector<std::vector<float>> encoded_storage;
  std::vector<EmbeddingView> encoded_embeddings;
  try {
    encoded_storage.resize(request.sequences.size);
    encoded_embeddings.resize(request.sequences.size);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "pair-list sequence embedding cache allocation failed");
  }

  // Encode-once: one ESM2 forward pass per unique sequence. The
  // `[<cls>, aa..., <eos>]` wrap and the CLS/EOS-row strip mirror
  // `run_all_vs_all_sequences` exactly (shared `encode_one_sequence`), so
  // pair-list per-residue embeddings are bit-identical to the all-vs-all and
  // pairwise sequence routes — the contract `pair_list_vs_pairwise_parity`
  // checks. The encode parallelizes over `pool` (each unique sequence is
  // encoded exactly once, on one worker, bumping the encode-once counter
  // once). The per-pair dispatch also parallelizes over `pool` when eligible,
  // staging records by input pair index before draining them to the sink in
  // caller order.
  if (should_parallel_encode_sequences(pool, thread_count,
                                       request.sequences.size)) {
    status = encode_sequences_parallel(
        request.sequences, *request.weights_view, request.descriptor, hidden,
        /*note_encode=*/true, pool, thread_count, worker_workspaces,
        encoded_storage, encoded_embeddings);
  } else {
    status = encode_sequences_serial(
        request.sequences, *request.weights_view, request.descriptor, hidden,
        /*note_encode=*/true, encoded_storage, encoded_embeddings);
  }
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  return dispatch_pairs(
      resolved_list_pair_source(request.pairs),
      {encoded_embeddings.data(), encoded_embeddings.size()}, {nullptr, 0},
      request.options, pair_count,
      embedding_workspace_plan(axis_maxima.max_query_length,
                               axis_maxima.max_target_length, hidden,
                               request.options.soft_mode),
      pool, thread_count, worker_workspaces, sink);
}

}  // namespace hikoboshi::algorithms

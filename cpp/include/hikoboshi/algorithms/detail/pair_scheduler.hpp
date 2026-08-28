#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_PAIR_SCHEDULER_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_PAIR_SCHEDULER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hikoboshi::algorithms::detail {

struct PairIndex {
  std::size_t query_index = 0;
  std::size_t target_index = 0;
};

struct PairRange {
  std::size_t begin = 0;
  std::size_t end = 0;
};

std::size_t symmetric_pair_count(std::size_t item_count,
                                 bool include_self) noexcept;

PairIndex pair_index_to_ij(std::size_t pair_index,
                           std::size_t item_count,
                           bool include_self) noexcept;

PairRange partition_pair_range(std::size_t pair_count,
                               std::size_t partition_index,
                               std::size_t partition_count) noexcept;

// Inverse of pair_index_to_ij for upper-triangular pairs.
// Returns the linear pair_index for a (query, target) coordinate pair.
// Assumes (query <= target) when include_self is true, (query < target)
// otherwise.
std::size_t ij_to_pair_index(std::size_t query_index,
                             std::size_t target_index,
                             std::size_t item_count,
                             bool include_self) noexcept;

// Tile-aware (Q × T) partition of the upper-triangular pair set.
//
// Each tile owns a contiguous slab of query rows [query_begin, query_end)
// and a contiguous slab of target columns [target_begin, target_end).
// A worker that executes a tile keeps the Q_tile = (query_end -
// query_begin) query embeddings hot in its private L1/L2 cache while
// iterating each (q, t) pair within the tile, paying the target
// embedding load cost only once per (target_begin, target_end) sweep
// per query.
//
// `predicted_cost` is the saturated sum of L_q × L_t over upper-
// triangular (q, t) pairs in the tile, used to dispatch heavier tiles
// first so cost spread across workers stays balanced (preserves p41's
// load-balance benefit at tile granularity).
struct PairTile {
  std::size_t query_begin = 0;
  std::size_t query_end = 0;
  std::size_t target_begin = 0;
  std::size_t target_end = 0;
  std::uint64_t predicted_cost = 0;
};

// Default tile dimensions.
//
// Q_tile is sized so that the active query embedding chunk stays inside
// per-thread L1+L2 budget on AMD EPYC 9655 (1 MB L2 per core, ~256 KB
// budget after pairwise workspaces). For Hikoboshi-MPNN-64 (hidden_dim
// = 64, sizeof(float) = 4) one residue costs 256 bytes, so an
// embedding of length L costs 256 * L bytes. Q_tile is the smallest
// integer with Q_tile * L_max * 256 <= 256 KB, clamped to
// [kMinQueryTileSize, kMaxQueryTileSize].
//
// T_tile is the contiguous run of targets a worker iterates before
// releasing the query chunk. Setting T_tile = 0 means "all targets
// reachable from the query slab" (one tile per query slab).
inline constexpr std::size_t kMinQueryTileSize = 1U;
inline constexpr std::size_t kMaxQueryTileSize = 16U;
inline constexpr std::size_t kDefaultTargetTileSize = 0U;
inline constexpr std::size_t kPerThreadEmbeddingBudgetBytes =
    256U * 1024U;
inline constexpr std::size_t kDefaultEmbeddingDimension = 64U;
// Below this pair count, the worker count and tile-per-worker floor
// dominate; query_tiled falls back to per-pair dispatch in the same
// shape as cost_aware so small workloads pay no extra cost.
inline constexpr std::size_t kQueryTiledPairThreshold = 256U;

struct QueryTiledPlan {
  std::size_t query_tile = 1U;
  std::size_t target_tile = 0U;  // 0 means "to the end of the row".
};

// Pick default Q_tile / T_tile from the workload. `embedding_dimension`
// = 0 falls back to kDefaultEmbeddingDimension. The plan is bucket-
// agnostic; p43's bucketed embedding cache handles per-query padding,
// so query reads from the cache are dominated by max-residue size of
// queries actually in the slab, not by the global L_max.
QueryTiledPlan resolve_query_tiled_plan(
    std::size_t item_count,
    std::size_t residue_count_max,
    std::size_t embedding_dimension) noexcept;

// Build tiles covering the upper-triangular pair set of `item_count`
// items. Output is appended to `tiles` (already sized for the worst-
// case tile count by the caller). Returns the number of non-empty
// tiles emitted.
//
// Tiles are emitted in ascending (Qi, Tj) lexicographic order; the
// caller is responsible for sorting them by descending predicted_cost
// before dispatch when load balance is desired.
std::size_t partition_pair_tiles(std::size_t item_count,
                                 bool include_self,
                                 std::size_t query_tile,
                                 std::size_t target_tile,
                                 std::vector<PairTile>& tiles);

// Worst-case tile count for `partition_pair_tiles` given the planning
// parameters. Useful for one-shot reservation of the tile vector.
std::size_t worst_case_pair_tile_count(std::size_t item_count,
                                       std::size_t query_tile,
                                       std::size_t target_tile) noexcept;

// Saturated cost of an (q, t) pair from residue counts.
std::uint64_t saturated_pair_cost_from_residue_counts(
    std::size_t lhs_residue_count,
    std::size_t rhs_residue_count) noexcept;

// Iterate every upper-triangular (q, t) pair belonging to `tile`, in
// q-major then t-ascending order. The visitor receives the
// pair_index and the (q, t) coordinates. The traversal preserves the
// ordering invariant so that records emitted in tile-iteration order
// match the natural pair_index order **within** the tile.
template <class Visit>
void iterate_pair_tile(const PairTile& tile,
                       std::size_t item_count,
                       bool include_self,
                       Visit&& visit) {
  const std::size_t query_end =
      tile.query_end <= item_count ? tile.query_end : item_count;
  const std::size_t target_end =
      tile.target_end <= item_count ? tile.target_end : item_count;
  for (std::size_t q = tile.query_begin; q < query_end; ++q) {
    const std::size_t t_first =
        include_self ? (q > tile.target_begin ? q : tile.target_begin)
                     : (q + 1U > tile.target_begin ? q + 1U
                                                   : tile.target_begin);
    if (t_first >= target_end) {
      continue;
    }
    for (std::size_t t = t_first; t < target_end; ++t) {
      const std::size_t pair_index =
          ij_to_pair_index(q, t, item_count, include_self);
      visit(pair_index, q, t);
    }
  }
}

}  // namespace hikoboshi::algorithms::detail

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_PAIR_SCHEDULER_HPP

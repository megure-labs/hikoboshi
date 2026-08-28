#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace hikoboshi::algorithms::detail {
namespace {

std::size_t half_product(std::size_t lhs, std::size_t rhs) noexcept {
  if ((lhs % 2U) == 0U) {
    return (lhs / 2U) * rhs;
  }
  return lhs * (rhs / 2U);
}

std::size_t prefix_pair_count(std::size_t row,
                              std::size_t item_count,
                              bool include_self) noexcept {
  if (row == 0 || item_count == 0) {
    return 0;
  }
  if (include_self) {
    return row * item_count - half_product(row, row - 1U);
  }
  return row * (item_count - 1U) - half_product(row, row - 1U);
}

std::size_t ceil_div(std::size_t numerator, std::size_t denominator) noexcept {
  if (denominator == 0U) {
    return 0U;
  }
  return (numerator + denominator - 1U) / denominator;
}

std::size_t saturated_residue_count(std::size_t residue_count) noexcept {
  constexpr std::uint64_t kMaxCost = std::numeric_limits<std::uint64_t>::max();
  if (residue_count > static_cast<std::size_t>(kMaxCost)) {
    return static_cast<std::size_t>(kMaxCost);
  }
  return residue_count;
}

}  // namespace

std::size_t symmetric_pair_count(std::size_t item_count,
                                 bool include_self) noexcept {
  if (include_self) {
    return half_product(item_count, item_count + 1U);
  }
  if (item_count < 2U) {
    return 0;
  }
  return half_product(item_count, item_count - 1U);
}

PairIndex pair_index_to_ij(std::size_t pair_index,
                           std::size_t item_count,
                           bool include_self) noexcept {
  const std::size_t pair_count =
      symmetric_pair_count(item_count, include_self);
  if (pair_index >= pair_count) {
    return {item_count, item_count};
  }

  std::size_t low = 0;
  std::size_t high = include_self ? item_count : item_count - 1U;
  while (low + 1U < high) {
    const std::size_t middle = low + (high - low) / 2U;
    if (prefix_pair_count(middle, item_count, include_self) <= pair_index) {
      low = middle;
    } else {
      high = middle;
    }
  }

  const std::size_t row = low;
  const std::size_t offset =
      pair_index - prefix_pair_count(row, item_count, include_self);
  const std::size_t target = include_self ? row + offset : row + 1U + offset;
  return {row, target};
}

std::size_t ij_to_pair_index(std::size_t query_index,
                             std::size_t target_index,
                             std::size_t item_count,
                             bool include_self) noexcept {
  const std::size_t row_offset =
      prefix_pair_count(query_index, item_count, include_self);
  const std::size_t column_offset =
      include_self ? target_index - query_index
                   : target_index - query_index - 1U;
  return row_offset + column_offset;
}

PairRange partition_pair_range(std::size_t pair_count,
                               std::size_t partition_index,
                               std::size_t partition_count) noexcept {
  if (partition_count == 0U || partition_index >= partition_count) {
    return {pair_count, pair_count};
  }
  const std::size_t base = pair_count / partition_count;
  const std::size_t remainder = pair_count % partition_count;
  const std::size_t begin =
      partition_index * base +
      (partition_index < remainder ? partition_index : remainder);
  const std::size_t size = base + (partition_index < remainder ? 1U : 0U);
  return {begin, begin + size};
}

std::uint64_t saturated_pair_cost_from_residue_counts(
    std::size_t lhs_residue_count,
    std::size_t rhs_residue_count) noexcept {
  constexpr std::uint64_t kMaxCost = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t lhs =
      static_cast<std::uint64_t>(saturated_residue_count(lhs_residue_count));
  const std::uint64_t rhs =
      static_cast<std::uint64_t>(saturated_residue_count(rhs_residue_count));
  if (lhs != 0U && rhs > kMaxCost / lhs) {
    return kMaxCost;
  }
  return lhs * rhs;
}

QueryTiledPlan resolve_query_tiled_plan(
    std::size_t item_count,
    std::size_t residue_count_max,
    std::size_t embedding_dimension) noexcept {
  QueryTiledPlan plan{};
  plan.query_tile = kMinQueryTileSize;
  plan.target_tile = kDefaultTargetTileSize;

  if (item_count == 0U) {
    return plan;
  }

  const std::size_t hidden_dim = embedding_dimension == 0U
                                     ? kDefaultEmbeddingDimension
                                     : embedding_dimension;
  // Clamp residue length to a sane minimum so the integer division below
  // does not promote Q_tile beyond the documented cap. Length-1 inputs
  // are still safe because the cap is applied afterwards.
  const std::size_t residue_max = residue_count_max == 0U ? 1U
                                                          : residue_count_max;
  const std::size_t bytes_per_residue = hidden_dim * sizeof(float);
  const std::size_t bytes_per_embedding = residue_max * bytes_per_residue;

  std::size_t q_tile = kMinQueryTileSize;
  if (bytes_per_embedding > 0U) {
    const std::size_t budget_tile =
        kPerThreadEmbeddingBudgetBytes / bytes_per_embedding;
    if (budget_tile > q_tile) {
      q_tile = budget_tile;
    }
  }
  if (q_tile > kMaxQueryTileSize) {
    q_tile = kMaxQueryTileSize;
  }
  if (q_tile == 0U) {
    q_tile = 1U;
  }
  plan.query_tile = q_tile;
  return plan;
}

std::size_t worst_case_pair_tile_count(std::size_t item_count,
                                       std::size_t query_tile,
                                       std::size_t target_tile) noexcept {
  if (item_count == 0U || query_tile == 0U) {
    return 0U;
  }
  const std::size_t q_chunks = ceil_div(item_count, query_tile);
  if (target_tile == 0U) {
    return q_chunks;
  }
  const std::size_t t_chunks = ceil_div(item_count, target_tile);
  return q_chunks * t_chunks;
}

std::size_t partition_pair_tiles(std::size_t item_count,
                                 bool include_self,
                                 std::size_t query_tile,
                                 std::size_t target_tile,
                                 std::vector<PairTile>& tiles) {
  tiles.clear();
  if (item_count == 0U || query_tile == 0U) {
    return 0U;
  }

  const std::size_t q_tile = query_tile;
  const std::size_t t_tile = target_tile;
  const std::size_t q_chunks = ceil_div(item_count, q_tile);

  std::size_t emitted = 0U;
  for (std::size_t qi = 0U; qi < q_chunks; ++qi) {
    const std::size_t q_begin = qi * q_tile;
    const std::size_t q_end = std::min(q_begin + q_tile, item_count);
    if (q_begin >= q_end) {
      continue;
    }

    // First reachable target column is q_begin (include_self) or
    // q_begin + 1 (no self-pairs). Any tile that is entirely below
    // the diagonal owns no upper-triangular pairs and is skipped.
    const std::size_t first_reachable_target =
        include_self ? q_begin
                     : q_begin + 1U;
    if (first_reachable_target >= item_count) {
      continue;
    }

    const std::size_t t_chunk_count =
        t_tile == 0U
            ? 1U
            : ceil_div(item_count - first_reachable_target, t_tile);
    for (std::size_t tj = 0U; tj < t_chunk_count; ++tj) {
      PairTile tile{};
      if (t_tile == 0U) {
        tile.target_begin = first_reachable_target;
        tile.target_end = item_count;
      } else {
        const std::size_t base = first_reachable_target + tj * t_tile;
        tile.target_begin = base;
        tile.target_end = std::min(base + t_tile, item_count);
      }
      if (tile.target_begin >= tile.target_end) {
        continue;
      }
      tile.query_begin = q_begin;
      tile.query_end = q_end;
      tiles.push_back(tile);
      ++emitted;
    }
  }
  return emitted;
}

}  // namespace hikoboshi::algorithms::detail

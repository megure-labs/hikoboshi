#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_ALL_VS_ALL_WORKSPACE_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_ALL_VS_ALL_WORKSPACE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <hikoboshi/algorithms/detail/pairwise_workspace.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/modules/mpnn.hpp>

namespace hikoboshi::algorithms {

struct PairwiseResultRecord {
  std::size_t query_index = 0;
  std::size_t target_index = 0;
  PairwiseResult result{};
};

namespace detail {

inline constexpr std::size_t kAllVsAllParallelEncodingThreshold = 4;
inline constexpr std::size_t kAllVsAllPairsPerAutoWorkerFloor = 2;

struct alignas(64) PerWorkerPairwiseBundle {
  PairwiseWorkspace pairwise;
  PairwiseWorkspace encoder;
  PairwiseResultRecord serial_record;
};

using AllVsAllWorkerWorkspace = PerWorkerPairwiseBundle;

static_assert(alignof(PerWorkerPairwiseBundle) == 64,
              "all-vs-all worker workspaces must be cache-line aligned");
static_assert(sizeof(PerWorkerPairwiseBundle) % 64 == 0,
              "all-vs-all worker workspaces must occupy whole cache lines");

[[nodiscard]] bool estimate_pairwise_workspace_storage_bytes(
    const PairwiseWorkspacePlan& plan,
    std::size_t& bytes) noexcept;

[[nodiscard]] bool estimate_all_vs_all_structure_encoder_workspace_bytes(
    std::size_t max_residue_count,
    const hikoboshi::modules::Mpnn64Descriptor& descriptor,
    std::size_t& bytes) noexcept;

[[nodiscard]] std::size_t select_all_vs_all_phase1_thread_count_for_budget(
    std::size_t requested_thread_count,
    std::size_t item_count,
    std::size_t workspace_bytes_per_thread,
    std::size_t workspace_budget_bytes) noexcept;

[[nodiscard]] inline std::size_t resolve_all_vs_all_auto_thread_count(
    std::uint32_t requested_thread_count,
    std::size_t pair_count,
    std::size_t hardware_thread_count) noexcept {
  if (requested_thread_count != 0U) {
    return static_cast<std::size_t>(requested_thread_count);
  }
  if (pair_count == 0U) {
    return 1U;
  }
  const std::size_t hardware =
      hardware_thread_count == 0U ? 1U : hardware_thread_count;
  const std::size_t pair_cap =
      (pair_count + kAllVsAllPairsPerAutoWorkerFloor - 1U) /
      kAllVsAllPairsPerAutoWorkerFloor;
  return std::min(hardware, pair_cap);
}

}  // namespace detail
}  // namespace hikoboshi::algorithms

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_ALL_VS_ALL_WORKSPACE_HPP

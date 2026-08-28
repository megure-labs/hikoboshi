#ifndef HIKOBOSHI_MODULES_DETAIL_MPNN_WORKSPACE_HPP
#define HIKOBOSHI_MODULES_DETAIL_MPNN_WORKSPACE_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::modules::detail {

struct Mpnn64MemoryPlan {
  std::size_t max_residue_count;
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
  std::size_t rbf_count;
  std::size_t layer_count;
};

struct Mpnn64Workspace {
  Mpnn64MemoryPlan plan;
  hikoboshi::universal::Span<float> ca_coordinates;
  // Reused as the per-edge [positional, RBF] input buffer for the real
  // Hikoboshi-MPNN-64 edge embedding path.
  hikoboshi::universal::Span<float> residue_features;
  hikoboshi::universal::Span<std::int32_t> neighbor_indices;
  hikoboshi::universal::Span<float> neighbor_squared_distances;
  hikoboshi::universal::Span<float> rbf_features;
  hikoboshi::universal::Span<float> residue_state;
  hikoboshi::universal::Span<float> gathered_state;
  hikoboshi::universal::Span<float> edge_state;
  hikoboshi::universal::Span<float> message_state;
  hikoboshi::universal::Span<float> projected_message_state;
  hikoboshi::universal::Span<float> residue_scratch;
  hikoboshi::universal::Span<float> ffn_hidden;
};

inline constexpr std::size_t mpnn64_ca_coordinate_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return plan.max_residue_count * 3;
}

inline constexpr std::size_t mpnn64_neighbor_slot_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return plan.max_residue_count * plan.neighbor_count;
}

inline constexpr std::size_t kMpnn64AtomPairCount = 25;
inline constexpr std::size_t kMpnn64PositionalFeatureCount = 16;
inline constexpr std::size_t kMpnn64PositionalClassCount = 66;
inline constexpr std::int32_t kMpnn64MaxRelativePosition = 32;

inline constexpr std::size_t mpnn64_edge_rbf_dimension(
    const Mpnn64MemoryPlan& plan) noexcept {
  return kMpnn64AtomPairCount * plan.rbf_count;
}

inline constexpr std::size_t mpnn64_edge_feature_dimension(
    const Mpnn64MemoryPlan& plan) noexcept {
  return kMpnn64PositionalFeatureCount + mpnn64_edge_rbf_dimension(plan);
}

inline constexpr std::size_t mpnn64_atom_pair_distance_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return mpnn64_neighbor_slot_count(plan) * kMpnn64AtomPairCount;
}

inline constexpr std::size_t mpnn64_neighbor_positional_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return mpnn64_neighbor_slot_count(plan) * kMpnn64PositionalFeatureCount;
}

inline constexpr std::size_t mpnn64_neighbor_edge_feature_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return mpnn64_neighbor_slot_count(plan) * mpnn64_edge_feature_dimension(plan);
}

inline constexpr std::size_t mpnn64_residue_feature_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return mpnn64_neighbor_edge_feature_count(plan);
}

inline constexpr std::size_t mpnn64_neighbor_hidden_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return mpnn64_neighbor_slot_count(plan) * plan.hidden_dimension;
}

inline constexpr std::size_t mpnn64_neighbor_rbf_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return mpnn64_neighbor_slot_count(plan) * mpnn64_edge_rbf_dimension(plan);
}

inline constexpr std::size_t mpnn64_residue_hidden_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return plan.max_residue_count * plan.hidden_dimension;
}

inline constexpr std::size_t mpnn64_ffn_hidden_count(
    const Mpnn64MemoryPlan& plan) noexcept {
  return plan.max_residue_count * 4 * plan.hidden_dimension;
}

}  // namespace hikoboshi::modules::detail

#endif  // HIKOBOSHI_MODULES_DETAIL_MPNN_WORKSPACE_HPP

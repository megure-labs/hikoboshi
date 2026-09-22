#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_MPNN_WORKSPACE_STORAGE_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_MPNN_WORKSPACE_STORAGE_HPP

#include <algorithm>
#include <vector>

#include <hikoboshi/modules/detail/mpnn_workspace.hpp>

namespace hikoboshi::algorithms::detail {

// Scalar MPNN-64 forward first materializes RBF features, copies them into
// residue_features, and emits any RBF diagnostic tensor. Only then does edge
// embedding / message passing write the node, edge and MLP buffers. Their
// disjoint lifetimes allow those buffers to share the RBF allocation. All
// message-passing spans remain mutually disjoint, including for small K.
// RBF values are scratch: they are not retained after edge embedding begins.
inline std::size_t mpnn64_phase_storage_count(
    const modules::detail::Mpnn64MemoryPlan& plan) noexcept {
  namespace pmd = modules::detail;
  return std::max(pmd::mpnn64_neighbor_rbf_count(plan),
                  4 * pmd::mpnn64_neighbor_hidden_count(plan) +
                      2 * pmd::mpnn64_residue_hidden_count(plan) +
                      pmd::mpnn64_ffn_hidden_count(plan));
}

class Mpnn64WorkspaceStorage {
 public:
  modules::detail::Mpnn64Workspace workspace{};

  void prepare(const modules::detail::Mpnn64MemoryPlan& plan) {
    namespace pmd = modules::detail;
    grow(ca_coordinates_, pmd::mpnn64_ca_coordinate_count(plan));
    grow(residue_features_, pmd::mpnn64_residue_feature_count(plan));
    grow(neighbor_indices_, pmd::mpnn64_neighbor_slot_count(plan));
    grow(neighbor_squared_distances_, pmd::mpnn64_neighbor_slot_count(plan));
    grow(phase_storage_, mpnn64_phase_storage_count(plan));

    workspace.plan = plan;
    workspace.ca_coordinates = span(ca_coordinates_);
    workspace.residue_features = span(residue_features_);
    workspace.neighbor_indices = span(neighbor_indices_);
    workspace.neighbor_squared_distances = span(neighbor_squared_distances_);
    workspace.rbf_features = slice(0, pmd::mpnn64_neighbor_rbf_count(plan));
    std::size_t offset = 0;
    const auto take = [&](std::size_t count) {
      auto result = slice(offset, count);
      offset += count;
      return result;
    };
    workspace.edge_state = take(pmd::mpnn64_neighbor_hidden_count(plan));
    workspace.message_state = take(pmd::mpnn64_neighbor_hidden_count(plan));
    workspace.projected_message_state = take(pmd::mpnn64_neighbor_hidden_count(plan));
    workspace.gathered_state = take(pmd::mpnn64_neighbor_hidden_count(plan));
    workspace.residue_state = take(pmd::mpnn64_residue_hidden_count(plan));
    workspace.residue_scratch = take(pmd::mpnn64_residue_hidden_count(plan));
    workspace.ffn_hidden = take(pmd::mpnn64_ffn_hidden_count(plan));
  }

 private:
  template <typename T>
  static void grow(std::vector<T>& values, std::size_t count) {
    if (values.size() < count) values.resize(count);
  }
  template <typename T>
  static universal::Span<T> span(std::vector<T>& values) noexcept {
    return {values.data(), values.size()};
  }
  universal::Span<float> slice(std::size_t offset, std::size_t count) noexcept {
    return {count == 0 ? nullptr : phase_storage_.data() + offset, count};
  }

  std::vector<float> ca_coordinates_;
  std::vector<float> residue_features_;
  std::vector<std::int32_t> neighbor_indices_;
  std::vector<float> neighbor_squared_distances_;
  std::vector<float> phase_storage_;
};

}  // namespace hikoboshi::algorithms::detail

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_MPNN_WORKSPACE_STORAGE_HPP

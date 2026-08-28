#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_PAIRWISE_WORKSPACE_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_PAIRWISE_WORKSPACE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hikoboshi/algorithms/detail/path_builder.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::algorithms::detail {

enum class SwTraceDirection : std::uint8_t {
  Stop = 0,
  Match = 1,
  InsertQuery = 2,
  DeleteTarget = 3,
};

struct PairwiseWorkspacePlan {
  std::size_t max_query_length = 0;
  std::size_t max_target_length = 0;
  std::size_t embedding_dimension = 0;
  bool allocate_mpnn = false;
  bool allocate_soft_sw = false;
  hikoboshi::modules::Mpnn64Descriptor mpnn_descriptor{};
};

class PairwiseWorkspace {
 public:
  struct OwnedMpnnWorkspace {
    hikoboshi::modules::detail::Mpnn64Workspace workspace{};
    std::vector<float> ca_coordinates;
    std::vector<float> residue_features;
    std::vector<std::int32_t> neighbor_indices;
    std::vector<float> neighbor_squared_distances;
    std::vector<float> rbf_features;
    std::vector<float> residue_state;
    std::vector<float> gathered_state;
    std::vector<float> edge_state;
    std::vector<float> message_state;
    std::vector<float> projected_message_state;
    std::vector<float> residue_scratch;
    std::vector<float> ffn_hidden;
  };

  // Soft Smith-Waterman backward-pass scratch. Forward DP cells reuse the
  // existing match_workspace_/insert_workspace_/delete_workspace_ buffers;
  // only the betas are extra. Each is sized (max_query+1)*(max_target+1).
  struct OwnedSoftSwWorkspace {
    std::vector<float> match_grad;
    std::vector<float> insert_grad;
    std::vector<float> delete_grad;
  };

  [[nodiscard]] hikoboshi::universal::Status prepare(
      const PairwiseWorkspacePlan& plan);

  const PairwiseWorkspacePlan& plan() const noexcept;
  bool has_mpnn_workspaces() const noexcept;
  bool has_soft_sw_workspaces() const noexcept;
  bool can_align(std::size_t query_length,
                 std::size_t target_length,
                 std::size_t embedding_dimension) const noexcept;

  float* query_embedding_data() noexcept;
  float* target_embedding_data() noexcept;
  float* similarity_data() noexcept;
  float* match_workspace_data() noexcept;
  float* insert_workspace_data() noexcept;
  float* delete_workspace_data() noexcept;
  float* match_grad_workspace_data() noexcept;
  float* insert_grad_workspace_data() noexcept;
  float* delete_grad_workspace_data() noexcept;
  float* posteriors_data() noexcept;
  SwTraceDirection* trace_match_data() noexcept;
  SwTraceDirection* trace_insert_data() noexcept;
  SwTraceDirection* trace_delete_data() noexcept;

  std::size_t sw_workspace_cells() const noexcept;
  std::size_t score_matrix_capacity() const noexcept;
  std::size_t posteriors_capacity() const noexcept;

  hikoboshi::modules::detail::Mpnn64Workspace* query_mpnn_workspace() noexcept;
  hikoboshi::modules::detail::Mpnn64Workspace* target_mpnn_workspace() noexcept;
  hikoboshi::universal::Span<hikoboshi::universal::AlignmentStep>
  traceback_step_scratch() noexcept;
  std::size_t traceback_step_capacity() const noexcept;
  PathBuilder& path_builder() noexcept;

 private:
  void deactivate_mpnn_workspaces() noexcept;
  void deactivate_soft_sw_workspaces() noexcept;

  PairwiseWorkspacePlan plan_{};
  bool prepared_ = false;
  bool has_mpnn_workspaces_ = false;
  bool has_soft_sw_workspaces_ = false;
  std::vector<float> query_embeddings_;
  std::vector<float> target_embeddings_;
  std::vector<float> similarity_;
  std::vector<float> match_workspace_;
  std::vector<float> insert_workspace_;
  std::vector<float> delete_workspace_;
  std::vector<SwTraceDirection> trace_match_;
  std::vector<SwTraceDirection> trace_insert_;
  std::vector<SwTraceDirection> trace_delete_;
  OwnedMpnnWorkspace query_mpnn_;
  OwnedMpnnWorkspace target_mpnn_;
  OwnedSoftSwWorkspace soft_sw_;
  std::vector<float> posteriors_;
  PathBuilder path_builder_;
};

}  // namespace hikoboshi::algorithms::detail

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_PAIRWISE_WORKSPACE_HPP

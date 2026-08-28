#include <hikoboshi/algorithms/detail/pairwise_workspace.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace hikoboshi::algorithms::detail {
namespace {

using hikoboshi::universal::AlignmentPath;
using hikoboshi::universal::AlignmentStep;
using hikoboshi::universal::Span;
using hikoboshi::universal::Status;

void assign_span(std::vector<float>& storage, Span<float>& span) noexcept {
  span = {storage.data(), storage.size()};
}

void assign_span(std::vector<std::int32_t>& storage,
                 Span<std::int32_t>& span) noexcept {
  span = {storage.data(), storage.size()};
}

template <typename T>
void grow_to(std::vector<T>& storage, std::size_t required_size) {
  if (storage.size() < required_size) {
    storage.resize(required_size);
  }
}

void resize_mpnn_workspace(
    PairwiseWorkspace::OwnedMpnnWorkspace& owned,
    const hikoboshi::modules::detail::Mpnn64MemoryPlan& plan) {
  namespace pmd = hikoboshi::modules::detail;
  owned.workspace.plan = plan;
  grow_to(owned.ca_coordinates, pmd::mpnn64_ca_coordinate_count(plan));
  grow_to(owned.residue_features, pmd::mpnn64_residue_feature_count(plan));
  grow_to(owned.neighbor_indices, pmd::mpnn64_neighbor_slot_count(plan));
  grow_to(owned.neighbor_squared_distances,
          pmd::mpnn64_neighbor_slot_count(plan));
  grow_to(owned.rbf_features, pmd::mpnn64_neighbor_rbf_count(plan));
  grow_to(owned.residue_state, pmd::mpnn64_residue_hidden_count(plan));
  grow_to(owned.gathered_state, pmd::mpnn64_neighbor_hidden_count(plan));
  grow_to(owned.edge_state, pmd::mpnn64_neighbor_hidden_count(plan));
  grow_to(owned.message_state, pmd::mpnn64_neighbor_hidden_count(plan));
  grow_to(owned.projected_message_state,
          pmd::mpnn64_neighbor_hidden_count(plan));
  grow_to(owned.residue_scratch, pmd::mpnn64_residue_hidden_count(plan));
  grow_to(owned.ffn_hidden, pmd::mpnn64_ffn_hidden_count(plan));

  assign_span(owned.ca_coordinates, owned.workspace.ca_coordinates);
  assign_span(owned.residue_features, owned.workspace.residue_features);
  assign_span(owned.neighbor_indices, owned.workspace.neighbor_indices);
  assign_span(owned.neighbor_squared_distances,
              owned.workspace.neighbor_squared_distances);
  assign_span(owned.rbf_features, owned.workspace.rbf_features);
  assign_span(owned.residue_state, owned.workspace.residue_state);
  assign_span(owned.gathered_state, owned.workspace.gathered_state);
  assign_span(owned.edge_state, owned.workspace.edge_state);
  assign_span(owned.message_state, owned.workspace.message_state);
  assign_span(owned.projected_message_state,
              owned.workspace.projected_message_state);
  assign_span(owned.residue_scratch, owned.workspace.residue_scratch);
  assign_span(owned.ffn_hidden, owned.workspace.ffn_hidden);
}

void deactivate_owned_mpnn(PairwiseWorkspace::OwnedMpnnWorkspace& owned) noexcept {
  owned.workspace = {};
}

void resize_soft_sw_workspace(
    PairwiseWorkspace::OwnedSoftSwWorkspace& owned,
    std::size_t cells) {
  grow_to(owned.match_grad, cells);
  grow_to(owned.insert_grad, cells);
  grow_to(owned.delete_grad, cells);
}

}  // namespace

void PathBuilder::prepare(std::size_t max_step_count) {
  grow_to(reverse_steps_, max_step_count);
  reset();
}

void PathBuilder::reset() noexcept {
  step_count_ = 0;
  aligned_pairs_ = 0;
  query_start_ = hikoboshi::universal::kAlignmentGapSentinel;
  query_end_ = hikoboshi::universal::kAlignmentGapSentinel;
  target_start_ = hikoboshi::universal::kAlignmentGapSentinel;
  target_end_ = hikoboshi::universal::kAlignmentGapSentinel;
}

bool PathBuilder::push_reverse(AlignmentStep step) noexcept {
  if (step_count_ >= reverse_steps_.size()) {
    return false;
  }
  reverse_steps_[step_count_] = step;
  ++step_count_;
  return true;
}

void PathBuilder::set_span(std::int32_t query_start,
                           std::int32_t query_end,
                           std::int32_t target_start,
                           std::int32_t target_end,
                           std::size_t aligned_pairs) noexcept {
  query_start_ = query_start;
  query_end_ = query_end;
  target_start_ = target_start;
  target_end_ = target_end;
  aligned_pairs_ = aligned_pairs;
}

void PathBuilder::write_ordered_to(AlignmentPath& path) const {
  path.steps.resize(step_count_);
  std::reverse_copy(reverse_steps_.begin(),
                    reverse_steps_.begin() + step_count_,
                    path.steps.begin());
  path.aligned_pairs = aligned_pairs_;
  path.query_start = query_start_;
  path.query_end = query_end_;
  path.target_start = target_start_;
  path.target_end = target_end_;
}

Span<AlignmentStep> PathBuilder::scratch() noexcept {
  return {reverse_steps_.data(), reverse_steps_.size()};
}

std::size_t PathBuilder::capacity() const noexcept {
  return reverse_steps_.size();
}

std::size_t PathBuilder::size() const noexcept {
  return step_count_;
}

Status PairwiseWorkspace::prepare(const PairwiseWorkspacePlan& plan) {
  if (plan.embedding_dimension == 0) {
    return hikoboshi::universal::invalid_argument_status("pairwise workspace embedding dimension must be non-zero");
  }
  if (plan.allocate_mpnn) {
    if (plan.mpnn_descriptor.hidden_dimension == 0 ||
        plan.mpnn_descriptor.neighbor_count == 0 ||
        plan.mpnn_descriptor.rbf_count == 0) {
      return hikoboshi::universal::invalid_argument_status("pairwise workspace MPNN descriptor is incomplete");
    }
    if (plan.mpnn_descriptor.hidden_dimension != plan.embedding_dimension) {
      return hikoboshi::universal::invalid_argument_status("pairwise workspace MPNN hidden dimension mismatch");
    }
  }

  if (!prepared_ || plan_.embedding_dimension != plan.embedding_dimension) {
    plan_ = {};
    plan_.max_query_length = plan.max_query_length;
    plan_.max_target_length = plan.max_target_length;
    plan_.embedding_dimension = plan.embedding_dimension;
  } else {
    plan_.max_query_length =
        std::max(plan_.max_query_length, plan.max_query_length);
    plan_.max_target_length =
        std::max(plan_.max_target_length, plan.max_target_length);
  }
  plan_.allocate_mpnn = plan.allocate_mpnn;
  plan_.allocate_soft_sw = plan.allocate_soft_sw;
  plan_.mpnn_descriptor =
      plan.allocate_mpnn ? plan.mpnn_descriptor
                         : hikoboshi::modules::Mpnn64Descriptor{};

  const std::size_t max_query = plan_.max_query_length;
  const std::size_t max_target = plan_.max_target_length;
  const std::size_t dim = plan.embedding_dimension;
  grow_to(query_embeddings_, max_query * dim);
  grow_to(target_embeddings_, max_target * dim);
  grow_to(similarity_, max_query * max_target);

  const std::size_t sw_cells = (max_query + 1) * (max_target + 1);
  grow_to(match_workspace_, sw_cells);
  grow_to(insert_workspace_, sw_cells);
  grow_to(delete_workspace_, sw_cells);
  grow_to(trace_match_, max_query * max_target);
  grow_to(trace_insert_, max_query * max_target);
  grow_to(trace_delete_, max_query * max_target);
  path_builder_.prepare(max_query + max_target);

  if (plan.allocate_soft_sw) {
    resize_soft_sw_workspace(soft_sw_, sw_cells);
    grow_to(posteriors_, max_query * max_target);
    has_soft_sw_workspaces_ = true;
  } else {
    deactivate_soft_sw_workspaces();
  }

  if (plan.allocate_mpnn) {
    hikoboshi::modules::detail::Mpnn64MemoryPlan query_plan{};
    query_plan.max_residue_count = plan_.max_query_length;
    query_plan.hidden_dimension = plan.mpnn_descriptor.hidden_dimension;
    query_plan.neighbor_count = plan.mpnn_descriptor.neighbor_count;
    query_plan.rbf_count = plan.mpnn_descriptor.rbf_count;
    query_plan.layer_count = plan.mpnn_descriptor.layer_count;

    hikoboshi::modules::detail::Mpnn64MemoryPlan target_plan = query_plan;
    target_plan.max_residue_count = plan_.max_target_length;

    resize_mpnn_workspace(query_mpnn_, query_plan);
    resize_mpnn_workspace(target_mpnn_, target_plan);
    has_mpnn_workspaces_ = true;
  } else {
    deactivate_mpnn_workspaces();
  }

  prepared_ = true;
  return hikoboshi::universal::ok_status();
}

const PairwiseWorkspacePlan& PairwiseWorkspace::plan() const noexcept {
  return plan_;
}

bool PairwiseWorkspace::has_mpnn_workspaces() const noexcept {
  return has_mpnn_workspaces_;
}

bool PairwiseWorkspace::has_soft_sw_workspaces() const noexcept {
  return has_soft_sw_workspaces_;
}

bool PairwiseWorkspace::can_align(std::size_t query_length,
                                  std::size_t target_length,
                                  std::size_t embedding_dimension) const noexcept {
  return prepared_ && query_length <= plan_.max_query_length &&
         target_length <= plan_.max_target_length &&
         embedding_dimension == plan_.embedding_dimension;
}

float* PairwiseWorkspace::query_embedding_data() noexcept {
  return query_embeddings_.data();
}

float* PairwiseWorkspace::target_embedding_data() noexcept {
  return target_embeddings_.data();
}

float* PairwiseWorkspace::similarity_data() noexcept {
  return similarity_.data();
}

float* PairwiseWorkspace::match_workspace_data() noexcept {
  return match_workspace_.data();
}

float* PairwiseWorkspace::insert_workspace_data() noexcept {
  return insert_workspace_.data();
}

float* PairwiseWorkspace::delete_workspace_data() noexcept {
  return delete_workspace_.data();
}

float* PairwiseWorkspace::match_grad_workspace_data() noexcept {
  return has_soft_sw_workspaces_ ? soft_sw_.match_grad.data() : nullptr;
}

float* PairwiseWorkspace::insert_grad_workspace_data() noexcept {
  return has_soft_sw_workspaces_ ? soft_sw_.insert_grad.data() : nullptr;
}

float* PairwiseWorkspace::delete_grad_workspace_data() noexcept {
  return has_soft_sw_workspaces_ ? soft_sw_.delete_grad.data() : nullptr;
}

float* PairwiseWorkspace::posteriors_data() noexcept {
  return has_soft_sw_workspaces_ ? posteriors_.data() : nullptr;
}

SwTraceDirection* PairwiseWorkspace::trace_match_data() noexcept {
  return trace_match_.data();
}

SwTraceDirection* PairwiseWorkspace::trace_insert_data() noexcept {
  return trace_insert_.data();
}

SwTraceDirection* PairwiseWorkspace::trace_delete_data() noexcept {
  return trace_delete_.data();
}

std::size_t PairwiseWorkspace::sw_workspace_cells() const noexcept {
  return match_workspace_.size();
}

std::size_t PairwiseWorkspace::score_matrix_capacity() const noexcept {
  return similarity_.size();
}

std::size_t PairwiseWorkspace::posteriors_capacity() const noexcept {
  return posteriors_.size();
}

hikoboshi::modules::detail::Mpnn64Workspace*
PairwiseWorkspace::query_mpnn_workspace() noexcept {
  return has_mpnn_workspaces_ ? &query_mpnn_.workspace : nullptr;
}

hikoboshi::modules::detail::Mpnn64Workspace*
PairwiseWorkspace::target_mpnn_workspace() noexcept {
  return has_mpnn_workspaces_ ? &target_mpnn_.workspace : nullptr;
}

Span<AlignmentStep> PairwiseWorkspace::traceback_step_scratch() noexcept {
  return path_builder_.scratch();
}

std::size_t PairwiseWorkspace::traceback_step_capacity() const noexcept {
  return path_builder_.capacity();
}

PathBuilder& PairwiseWorkspace::path_builder() noexcept {
  return path_builder_;
}

void PairwiseWorkspace::deactivate_mpnn_workspaces() noexcept {
  deactivate_owned_mpnn(query_mpnn_);
  deactivate_owned_mpnn(target_mpnn_);
  has_mpnn_workspaces_ = false;
}

void PairwiseWorkspace::deactivate_soft_sw_workspaces() noexcept {
  has_soft_sw_workspaces_ = false;
}

}  // namespace hikoboshi::algorithms::detail

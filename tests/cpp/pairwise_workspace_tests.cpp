#define HIKOBOSHI_TEST_ALLOC_COUNTER_IMPLEMENTATION
#include "support/test_alloc_counter.hpp"

#include <hikoboshi/algorithms/detail/resolved_alignment.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_md = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "pairwise_workspace_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double a, double b, double tolerance = 1.0e-6) {
  return std::fabs(a - b) <= tolerance;
}

static_assert(std::is_standard_layout<hiko_ad::ResolvedAlignmentProblem>::value,
              "ResolvedAlignmentProblem must stay a plain contract record");
static_assert(std::is_trivially_copyable<hiko_u::ScoreMatrixView>::value,
              "ScoreMatrixView must stay trivially copyable");
static_assert(std::is_trivially_copyable<hiko_u::AffineGapModel>::value,
              "AffineGapModel must stay trivially copyable");

void require_ok(hiko_u::Status status, const char* message) {
  if (status.code != hiko_u::StatusCode::Ok) {
    std::fprintf(stderr, "pairwise_workspace_tests: %s: %s\n", message,
                 status.detail == nullptr ? "" : status.detail);
    std::exit(1);
  }
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values,
                                  const std::vector<char>& codes,
                                  std::size_t residue_count,
                                  std::size_t dimension) {
  return {residue_count,
          dimension,
          {values.data(), values.size()},
          {codes.data(), codes.size()},
          {nullptr, 0}};
}

void test_resolved_alignment_problem_contract() {
  const float scores[2 * 2] = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };

  const hiko_ad::ResolvedAlignmentProblem problem{
      {scores, 2, 2, 2},
      hiko_u::kRawDotV1ScoreSemantics,
      hiko_u::kHardSwDefaultAffineGapModel,
      hiko_ad::kHardLocalAffineSwV1Algorithm,
      hiko_ad::kPublicPairwiseTracebackPolicy,
      {},
      {},
  };

  if (!hiko_ad::is_supported_0_1_alignment_problem(problem)) {
    fail("resolved alignment problem must encode the narrow Hikoboshi 0.1 contract");
  }
  if (problem.scores.values[1 * problem.scores.row_stride + 1] != 1.0F) {
    fail("resolved score matrix must be row-major query by target");
  }
  if (problem.gaps.gap_open != hiko::kPairwiseDefaultGapOpen ||
      problem.gaps.gap_extension != hiko::kPairwiseDefaultGapExtension) {
    fail("resolved affine gap model must match pairwise hard-SW defaults");
  }
  if (!hiko_ad::is_hard_sw_default_affine_gap_model(problem.gaps)) {
    fail("resolved default affine gap model must be identifiable");
  }
  hiko_ad::ResolvedAlignmentProblem custom_gap_problem = problem;
  custom_gap_problem.gaps.gap_open = -2.0F;
  custom_gap_problem.gaps.gap_extension = -0.2F;
  if (!hiko_ad::is_supported_0_1_alignment_problem(custom_gap_problem)) {
    fail("resolved alignment problem must allow finite affine gap overrides");
  }
  if (!hiko_ad::is_public_traceback_required(
          hiko_ad::kPublicAllVsAllTracebackPolicy)) {
    fail("public all-vs-all traceback policy must be required");
  }
}

void test_embedding_only_workspace_skips_mpnn_scratch() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 3;
  plan.max_target_length = 3;
  plan.embedding_dimension = 3;
  plan.allocate_mpnn = false;
  require_ok(workspace.prepare(plan), "workspace prepare");
  if (workspace.has_mpnn_workspaces()) {
    fail("embedding-only workspace must not allocate MPNN scratch");
  }
}

void test_mpnn_workspace_ffn_capacity_matches_memory_plan() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 2;
  plan.max_target_length = 3;
  plan.embedding_dimension = 64;
  plan.allocate_mpnn = true;
  plan.mpnn_descriptor.hidden_dimension = 64;
  plan.mpnn_descriptor.neighbor_count = 2;
  plan.mpnn_descriptor.rbf_count = 16;
  plan.mpnn_descriptor.layer_count = 3;
  require_ok(workspace.prepare(plan), "workspace prepare with MPNN");

  const hiko_md::Mpnn64Workspace* query = workspace.query_mpnn_workspace();
  const hiko_md::Mpnn64Workspace* target = workspace.target_mpnn_workspace();
  if (query == nullptr || target == nullptr) {
    fail("allocated MPNN workspace must expose query and target workspaces");
  }
  if (query->ffn_hidden.size != hiko_md::mpnn64_ffn_hidden_count(query->plan)) {
    fail("query MPNN FFN scratch capacity must match memory plan");
  }
  if (target->ffn_hidden.size != hiko_md::mpnn64_ffn_hidden_count(target->plan)) {
    fail("target MPNN FFN scratch capacity must match memory plan");
  }
  if (query->ffn_hidden.size <= hiko_md::mpnn64_residue_hidden_count(query->plan) ||
      target->ffn_hidden.size <= hiko_md::mpnn64_residue_hidden_count(target->plan)) {
    fail("MPNN FFN scratch must be larger than residue-hidden scratch");
  }
}

void test_workspace_prepare_reuses_same_shape_capacity() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 4;
  plan.max_target_length = 5;
  plan.embedding_dimension = 3;
  plan.allocate_mpnn = false;
  require_ok(workspace.prepare(plan), "workspace prepare warmup");

  const float* query_storage = workspace.query_embedding_data();
  const float* similarity_storage = workspace.similarity_data();
  const auto traceback_scratch = workspace.traceback_step_scratch();
  if (workspace.score_matrix_capacity() < plan.max_query_length *
                                            plan.max_target_length) {
    fail("workspace score-matrix buffer must cover max query x max target");
  }

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t iteration = 0; iteration < 20; ++iteration) {
    require_ok(workspace.prepare(plan), "same-shape workspace prepare");
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0) {
    fail("same-shape workspace prepare must not allocate after warmup");
  }
  if (workspace.query_embedding_data() != query_storage ||
      workspace.similarity_data() != similarity_storage ||
      workspace.traceback_step_scratch().data != traceback_scratch.data) {
    fail("same-shape workspace prepare must keep warmed storage");
  }
}

void test_workspace_prepare_does_not_shrink_capacity() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan large{};
  large.max_query_length = 4;
  large.max_target_length = 5;
  large.embedding_dimension = 3;
  large.allocate_mpnn = false;
  require_ok(workspace.prepare(large), "large workspace prepare");

  const float* query_storage = workspace.query_embedding_data();
  const float* target_storage = workspace.target_embedding_data();
  const float* similarity_storage = workspace.similarity_data();
  const auto traceback_scratch = workspace.traceback_step_scratch();

  hiko_ad::PairwiseWorkspacePlan small = large;
  small.max_query_length = 2;
  small.max_target_length = 2;

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  require_ok(workspace.prepare(small), "smaller workspace prepare");
  require_ok(workspace.prepare(large), "regrown workspace prepare");
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0) {
    fail("workspace prepare must reuse larger warmed capacity");
  }
  if (workspace.plan().max_query_length != large.max_query_length ||
      workspace.plan().max_target_length != large.max_target_length ||
      workspace.traceback_step_capacity() !=
          large.max_query_length + large.max_target_length) {
    fail("workspace prepare must keep capacity after smaller requests");
  }
  if (workspace.query_embedding_data() != query_storage ||
      workspace.target_embedding_data() != target_storage ||
      workspace.similarity_data() != similarity_storage ||
      workspace.traceback_step_scratch().data != traceback_scratch.data) {
    fail("workspace prepare must not move warmed buffers for smaller requests");
  }
}

void test_mpnn_workspace_prepare_reuses_capacity_after_warmup() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 3;
  plan.max_target_length = 4;
  plan.embedding_dimension = 64;
  plan.allocate_mpnn = true;
  plan.mpnn_descriptor.hidden_dimension = 64;
  plan.mpnn_descriptor.neighbor_count = 2;
  plan.mpnn_descriptor.rbf_count = 16;
  plan.mpnn_descriptor.layer_count = 3;
  require_ok(workspace.prepare(plan), "MPNN workspace warmup");

  const hiko_md::Mpnn64Workspace* query = workspace.query_mpnn_workspace();
  const hiko_md::Mpnn64Workspace* target = workspace.target_mpnn_workspace();
  if (query == nullptr || target == nullptr) {
    fail("MPNN warmup must expose query and target workspaces");
  }
  const float* query_ca = query->ca_coordinates.data;
  const float* target_ca = target->ca_coordinates.data;
  const float* query_ffn = query->ffn_hidden.data;
  const float* target_ffn = target->ffn_hidden.data;

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t iteration = 0; iteration < 20; ++iteration) {
    require_ok(workspace.prepare(plan), "same-shape MPNN workspace prepare");
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0) {
    fail("same-shape MPNN workspace prepare must not allocate after warmup");
  }
  query = workspace.query_mpnn_workspace();
  target = workspace.target_mpnn_workspace();
  if (query == nullptr || target == nullptr ||
      query->ca_coordinates.data != query_ca ||
      target->ca_coordinates.data != target_ca ||
      query->ffn_hidden.data != query_ffn ||
      target->ffn_hidden.data != target_ffn) {
    fail("same-shape MPNN workspace prepare must keep warmed storage");
  }
}

hiko_ad::ResolvedAlignmentProblem fixed_score_problem(const float* scores,
                                                   std::size_t query_length,
                                                   std::size_t target_length) {
  return {{scores, query_length, target_length, target_length},
          hiko_u::kRawDotV1ScoreSemantics,
          hiko_u::kHardSwDefaultAffineGapModel,
          hiko_ad::kHardLocalAffineSwV1Algorithm,
          hiko_ad::kPublicPairwiseTracebackPolicy,
          {},
          {}};
}

void require_identity_path(const hiko_u::AlignmentPath& path,
                           std::size_t length) {
  if (path.aligned_pairs != length || path.steps.size() != length) {
    fail("identity traceback path shape mismatch");
  }
  for (std::size_t index = 0; index < length; ++index) {
    const hiko_u::AlignmentStep& step = path.steps[index];
    if (step.query_index != static_cast<std::int32_t>(index) ||
        step.target_index != static_cast<std::int32_t>(index) ||
        !nearly_equal(step.residue_score, 1.0F)) {
      fail("identity traceback step mismatch");
    }
  }
}

void test_workspace_traceback_scratch_capacity_and_reuse() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 2;
  plan.max_target_length = 2;
  plan.embedding_dimension = 2;
  plan.allocate_mpnn = false;
  require_ok(workspace.prepare(plan), "workspace prepare");

  const auto initial_scratch = workspace.traceback_step_scratch();
  if (initial_scratch.data == nullptr ||
      initial_scratch.size != plan.max_query_length + plan.max_target_length ||
      workspace.traceback_step_capacity() != initial_scratch.size) {
    fail("workspace traceback scratch must be prepared to max query + target");
  }

  const float scores[2 * 2] = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  hiko_u::AlignmentPath path{};
  path.steps.reserve(initial_scratch.size);
  double score = 0.0;
  require_ok(hiko_ad::run_resolved_alignment_problem(
                 fixed_score_problem(scores, 2, 2), workspace, path, score),
             "first resolved alignment");
  require_identity_path(path, 2);
  if (workspace.traceback_step_scratch().data != initial_scratch.data) {
    fail("workspace traceback scratch must be reused after first traceback");
  }

  path.steps.clear();
  require_ok(hiko_ad::run_resolved_alignment_problem(
                 fixed_score_problem(scores, 2, 2), workspace, path, score),
             "second resolved alignment");
  require_identity_path(path, 2);
  if (workspace.traceback_step_scratch().data != initial_scratch.data) {
    fail("workspace traceback scratch must be reused across alignments");
  }
}

void test_path_builder_no_allocation_after_prepare() {
  hiko_ad::PathBuilder builder;
  builder.prepare(4);

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  builder.reset();
  if (!builder.push_reverse({2, 2, 1.0F}) ||
      !builder.push_reverse({1, 1, 1.0F}) ||
      !builder.push_reverse({0, 0, 1.0F})) {
    fail("path builder prepared capacity should accept reverse steps");
  }
  builder.set_span(0, 2, 0, 2, 3);
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0) {
    fail("path builder push/reset must not allocate after prepare");
  }
}

void test_prepared_sw_traceback_region_does_not_allocate() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 3;
  plan.max_target_length = 3;
  plan.embedding_dimension = 3;
  plan.allocate_mpnn = false;
  require_ok(workspace.prepare(plan), "workspace prepare");

  const float scores[3 * 3] = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  hiko_u::AlignmentPath path{};
  path.steps.reserve(plan.max_query_length + plan.max_target_length);
  double score = 0.0;

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t iteration = 0; iteration < 20; ++iteration) {
    require_ok(hiko_ad::run_resolved_alignment_problem(
                   fixed_score_problem(scores, 3, 3), workspace, path, score),
               "prepared resolved alignment");
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0) {
    fail("prepared SW plus traceback region must not allocate");
  }
  if (!nearly_equal(score, 3.0)) {
    fail("prepared SW allocation regression must preserve raw score");
  }
  require_identity_path(path, 3);
}

void test_pairwise_embedding_pipeline_score_and_path() {
  hiko_ad::PairwiseWorkspace workspace;
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 3;
  plan.max_target_length = 3;
  plan.embedding_dimension = 3;
  plan.allocate_mpnn = false;
  require_ok(workspace.prepare(plan), "workspace prepare");

  const std::vector<float> query_values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<float> target_values = query_values;
  const std::vector<char> query_codes = {'A', 'C', 'D'};
  const std::vector<char> target_codes = {'A', 'C', 'D'};

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = embedding_view(query_values, query_codes, 3, 3);
  request.target_embedding = embedding_view(target_values, target_codes, 3, 3);

  hiko::PairwiseResult result{};
  require_ok(hiko::run_pairwise_embeddings(request, workspace, result),
             "run_pairwise_embeddings");

  if (!nearly_equal(result.raw_sw_score, 3.0)) {
    fail("identity one-hot embeddings must produce raw SW score 3");
  }
  if (result.path.aligned_pairs != 3 || result.path.steps.size() != 3) {
    fail("identity one-hot embeddings must produce three aligned pairs");
  }
  for (std::size_t i = 0; i < result.path.steps.size(); ++i) {
    const auto& step = result.path.steps[i];
    if (step.query_index != static_cast<std::int32_t>(i) ||
        step.target_index != static_cast<std::int32_t>(i)) {
      fail("pairwise alignment path must be ordered by local traceback");
    }
  }
  if (!result.metrics.coverage_query.valid ||
      !nearly_equal(result.metrics.coverage_query.value, 1.0) ||
      !result.metrics.coverage_target.valid ||
      !nearly_equal(result.metrics.coverage_target.value, 1.0) ||
      !result.metrics.identity.valid ||
      !nearly_equal(result.metrics.identity.value, 1.0)) {
    fail("pairwise result assembly must attach coverage and identity metrics");
  }
  if (result.metrics.rmsd.valid ||
      result.metrics.rmsd.reason != hiko_u::MetricInvalidReason::MissingStructureMetadata) {
    fail("embedding-only structural metrics must be invalid with missing metadata");
  }
}

}  // namespace

int main() {
  test_resolved_alignment_problem_contract();
  test_embedding_only_workspace_skips_mpnn_scratch();
  test_mpnn_workspace_ffn_capacity_matches_memory_plan();
  test_workspace_prepare_reuses_same_shape_capacity();
  test_workspace_prepare_does_not_shrink_capacity();
  test_mpnn_workspace_prepare_reuses_capacity_after_warmup();
  test_workspace_traceback_scratch_capacity_and_reuse();
  test_path_builder_no_allocation_after_prepare();
  test_prepared_sw_traceback_region_does_not_allocate();
  test_pairwise_embedding_pipeline_score_and_path();
  return 0;
}

#include <hikoboshi/algorithms/detail/resolved_alignment.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "resolved_alignment_problem_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double a, double b, double tolerance = 1.0e-6) {
  return std::fabs(a - b) <= tolerance;
}

void require_ok(hiko_u::Status status, const char* message) {
  if (status.code != hiko_u::StatusCode::Ok) {
    std::fprintf(stderr, "resolved_alignment_problem_tests: %s: %s\n",
                 message, status.detail == nullptr ? "" : status.detail);
    std::exit(1);
  }
}

void require_invalid(hiko_u::Status status, const char* message) {
  if (status.code != hiko_u::StatusCode::InvalidArgument ||
      status.detail == nullptr || status.detail[0] == '\0') {
    std::fprintf(stderr, "resolved_alignment_problem_tests: %s: %s\n",
                 message, status.detail == nullptr ? "" : status.detail);
    std::exit(1);
  }
}

hiko_ad::PairwiseWorkspacePlan workspace_plan() {
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = 3;
  plan.max_target_length = 3;
  plan.embedding_dimension = 3;
  plan.allocate_mpnn = false;
  return plan;
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values) {
  return {3, 3, {values.data(), values.size()}, {nullptr, 0}, {nullptr, 0}};
}

hiko_ad::ResolvedAlignmentProblem valid_problem(const float* scores) {
  return {{scores, 3, 3, 3},
          hiko_u::kRawDotV1ScoreSemantics,
          hiko_u::kHardSwDefaultAffineGapModel,
          hiko_ad::kHardLocalAffineSwV1Algorithm,
          hiko_ad::kPublicPairwiseTracebackPolicy,
          {},
          {}};
}

void require_same_path(const hiko_u::AlignmentPath& actual,
                       const hiko_u::AlignmentPath& expected) {
  if (actual.aligned_pairs != expected.aligned_pairs ||
      actual.query_start != expected.query_start ||
      actual.query_end != expected.query_end ||
      actual.target_start != expected.target_start ||
      actual.target_end != expected.target_end ||
      actual.steps.size() != expected.steps.size()) {
    fail("direct resolved path metadata differs from embedding path");
  }

  for (std::size_t index = 0; index < actual.steps.size(); ++index) {
    const hiko_u::AlignmentStep& a = actual.steps[index];
    const hiko_u::AlignmentStep& e = expected.steps[index];
    if (a.query_index != e.query_index ||
        a.target_index != e.target_index ||
        !nearly_equal(a.residue_score, e.residue_score)) {
      fail("direct resolved path steps differ from embedding path");
    }
  }
}

void test_direct_score_matrix_matches_embedding_route() {
  const std::vector<float> one_hot = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<float> fixed_scores = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };

  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(workspace_plan()), "workspace prepare");

  hiko::PairwiseEmbeddingRequest embedding_request{};
  embedding_request.query_embedding = embedding_view(one_hot);
  embedding_request.target_embedding = embedding_view(one_hot);
  hiko::PairwiseResult embedding_result{};
  require_ok(hiko::run_pairwise_embeddings(embedding_request, workspace,
                                          embedding_result),
             "embedding pairwise");

  hiko_u::AlignmentPath direct_path{};
  double direct_score = 0.0;
  require_ok(hiko_ad::run_resolved_alignment_problem(
                 valid_problem(fixed_scores.data()), workspace, direct_path,
                 direct_score),
             "direct resolved alignment");

  if (!nearly_equal(direct_score, embedding_result.raw_sw_score) ||
      !nearly_equal(direct_score,
                    embedding_result.metrics.raw_sw_score)) {
    fail("direct resolved raw score differs from embedding path");
  }
  require_same_path(direct_path, embedding_result.path);
}

void test_rejects_unsupported_score_semantics() {
  const float scores[3 * 3] = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(workspace_plan()), "workspace prepare");

  hiko_ad::ResolvedAlignmentProblem problem = valid_problem(scores);
  problem.semantics.method = hiko_u::ScoreMethod::CosineV1;

  hiko_u::AlignmentPath path{};
  double score = 0.0;
  require_invalid(hiko_ad::run_resolved_alignment_problem(problem, workspace, path,
                                                       score),
                  "unsupported score semantics must be rejected");
}

void test_rejects_non_hard_sw_algorithm() {
  const float scores[3 * 3] = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(workspace_plan()), "workspace prepare");

  hiko_ad::ResolvedAlignmentProblem problem = valid_problem(scores);
  problem.algorithm = hiko_u::AlignmentAlgorithmId::GlobalAffineSwV1;

  hiko_u::AlignmentPath path{};
  double score = 0.0;
  require_invalid(hiko_ad::run_resolved_alignment_problem(problem, workspace, path,
                                                       score),
                  "non-hard-SW algorithm must be rejected");
}

void test_rejects_non_finite_gap_values() {
  const float scores[3 * 3] = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(workspace_plan()), "workspace prepare");

  hiko_ad::ResolvedAlignmentProblem problem = valid_problem(scores);
  problem.gaps.gap_open = std::numeric_limits<float>::infinity();

  hiko_u::AlignmentPath path{};
  double score = 0.0;
  require_invalid(hiko_ad::run_resolved_alignment_problem(problem, workspace, path,
                                                       score),
                  "non-finite gap values must be rejected");
}

}  // namespace

int main() {
  test_direct_score_matrix_matches_embedding_route();
  test_rejects_unsupported_score_semantics();
  test_rejects_non_hard_sw_algorithm();
  test_rejects_non_finite_gap_values();
  return 0;
}

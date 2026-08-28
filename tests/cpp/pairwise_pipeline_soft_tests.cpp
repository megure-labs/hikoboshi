#define HIKOBOSHI_TEST_ALLOC_COUNTER_IMPLEMENTATION
#include "support/test_alloc_counter.hpp"

#include <hikoboshi/algorithms/pairwise.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "pairwise_pipeline_soft_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double a, double b, double tolerance = 1.0e-5) {
  return std::fabs(a - b) <= tolerance;
}

struct EmbeddingFixture {
  std::size_t length = 0;
  std::size_t dimension = 0;
  std::vector<float> query_values;
  std::vector<float> target_values;

  hiko_u::EmbeddingView query_view() const noexcept {
    return {length, dimension, {query_values.data(), query_values.size()}, {}, {}};
  }
  hiko_u::EmbeddingView target_view() const noexcept {
    return {length, dimension, {target_values.data(), target_values.size()}, {}, {}};
  }
};

// Builds a query/target embedding pair where the dot product is dominated by
// the diagonal: q[i] . t[j] is highest at i == j.
EmbeddingFixture make_diagonal_dominant_fixture(std::size_t length,
                                                std::size_t dimension,
                                                unsigned seed) {
  EmbeddingFixture fixture;
  fixture.length = length;
  fixture.dimension = dimension;
  fixture.query_values.assign(length * dimension, 0.0F);
  fixture.target_values.assign(length * dimension, 0.0F);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> noise(-0.05F, 0.05F);

  for (std::size_t i = 0; i < length; ++i) {
    const std::size_t channel = i % dimension;
    for (std::size_t d = 0; d < dimension; ++d) {
      const float n = noise(rng);
      const float anchor = d == channel ? 1.0F : 0.0F;
      fixture.query_values[i * dimension + d] = anchor + n;
      fixture.target_values[i * dimension + d] = anchor + noise(rng);
    }
  }
  return fixture;
}

EmbeddingFixture make_random_fixture(std::size_t length,
                                     std::size_t dimension,
                                     unsigned seed) {
  EmbeddingFixture fixture;
  fixture.length = length;
  fixture.dimension = dimension;
  fixture.query_values.assign(length * dimension, 0.0F);
  fixture.target_values.assign(length * dimension, 0.0F);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  for (float& v : fixture.query_values) v = distribution(rng);
  for (float& v : fixture.target_values) v = distribution(rng);
  return fixture;
}

hiko::detail::PairwiseWorkspace make_prepared_workspace(
    std::size_t length, std::size_t dimension, bool allocate_soft_sw = true) {
  hiko::detail::PairwiseWorkspace workspace;
  hiko::detail::PairwiseWorkspacePlan plan{};
  plan.max_query_length = length;
  plan.max_target_length = length;
  plan.embedding_dimension = dimension;
  plan.allocate_soft_sw = allocate_soft_sw;
  const auto status = workspace.prepare(plan);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("workspace prepare must return Ok");
  }
  return workspace;
}

bool paths_equal(const hiko_u::AlignmentPath& lhs,
                 const hiko_u::AlignmentPath& rhs) {
  if (lhs.aligned_pairs != rhs.aligned_pairs ||
      lhs.query_start != rhs.query_start ||
      lhs.query_end != rhs.query_end ||
      lhs.target_start != rhs.target_start ||
      lhs.target_end != rhs.target_end ||
      lhs.steps.size() != rhs.steps.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.steps.size(); ++index) {
    const hiko_u::AlignmentStep& a = lhs.steps[index];
    const hiko_u::AlignmentStep& b = rhs.steps[index];
    if (a.query_index != b.query_index ||
        a.target_index != b.target_index ||
        a.residue_score != b.residue_score) {
      return false;
    }
  }
  return true;
}

void test_default_request_runs_hard_mode_unchanged() {
  // When soft_mode == false (default), the driver must take the existing
  // hard local affine SW path and produce the same outputs as before.
  constexpr std::size_t length = 12;
  constexpr std::size_t dimension = 8;
  EmbeddingFixture fixture =
      make_diagonal_dominant_fixture(length, dimension, /*seed=*/0xC0FFEEU);

  hiko::detail::PairwiseWorkspace workspace_hard;
  hiko::detail::PairwiseWorkspacePlan hard_plan{};
  hard_plan.max_query_length = length;
  hard_plan.max_target_length = length;
  hard_plan.embedding_dimension = dimension;
  hard_plan.allocate_soft_sw = false;
  if (workspace_hard.prepare(hard_plan).code != hiko_u::StatusCode::Ok) {
    fail("hard-only workspace prepare must succeed");
  }

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = fixture.query_view();
  request.target_embedding = fixture.target_view();
  hiko::PairwiseResult result{};
  if (hiko::run_pairwise_embeddings(request, workspace_hard, result).code !=
      hiko_u::StatusCode::Ok) {
    fail("default (hard) request must succeed");
  }
  if (result.path.aligned_pairs == 0 || result.metrics.raw_sw_score <= 0.0) {
    fail("hard-mode default fixture must produce a non-empty alignment");
  }
}

void test_low_temperature_soft_path_matches_hard_path() {
  // At T = 1e-4 the soft posteriors should peak on the same residue pairs as
  // the hard-SW argmax path on a diagonally-dominant fixture, so the soft
  // and hard paths should share the same matched (i, j) cells.
  constexpr std::size_t length = 10;
  constexpr std::size_t dimension = 8;
  EmbeddingFixture fixture =
      make_diagonal_dominant_fixture(length, dimension, /*seed=*/0xBADBEEFU);
  hiko::detail::PairwiseWorkspace workspace =
      make_prepared_workspace(length, dimension);

  hiko::PairwiseEmbeddingRequest hard_request{};
  hard_request.query_embedding = fixture.query_view();
  hard_request.target_embedding = fixture.target_view();
  hiko::PairwiseResult hard_result{};
  if (hiko::run_pairwise_embeddings(hard_request, workspace, hard_result).code !=
      hiko_u::StatusCode::Ok) {
    fail("hard run must succeed under shared workspace");
  }
  if (hard_result.path.aligned_pairs == 0) {
    fail("hard fixture must produce a non-empty path before soft comparison");
  }

  hiko::PairwiseEmbeddingRequest soft_request = hard_request;
  soft_request.soft_mode = true;
  soft_request.temperature = 1e-4F;
  hiko::PairwiseResult soft_result{};
  if (hiko::run_pairwise_embeddings(soft_request, workspace, soft_result).code !=
      hiko_u::StatusCode::Ok) {
    fail("soft run at T=1e-4 must succeed");
  }
  if (!std::isfinite(soft_result.metrics.raw_sw_score)) {
    fail("soft raw_sw_score (log_partition) must be finite");
  }

  std::vector<bool> hard_mask(length * length, false);
  for (const auto& step : hard_result.path.steps) {
    if (step.query_index >= 0 && step.target_index >= 0) {
      hard_mask[static_cast<std::size_t>(step.query_index) * length +
                static_cast<std::size_t>(step.target_index)] = true;
    }
  }
  std::vector<bool> soft_mask(length * length, false);
  for (const auto& step : soft_result.path.steps) {
    if (step.query_index < 0 || step.target_index < 0) {
      fail("soft step indices must be non-negative match cells");
    }
    soft_mask[static_cast<std::size_t>(step.query_index) * length +
              static_cast<std::size_t>(step.target_index)] = true;
  }

  // At T=1e-4 the soft path should agree with hard on the dominant matched
  // cells, modulo a few-cell tolerance for ties / boundary effects.
  std::size_t mismatched = 0;
  for (std::size_t i = 0; i < length * length; ++i) {
    if (hard_mask[i] != soft_mask[i]) {
      ++mismatched;
    }
  }
  const std::size_t tolerance = std::max<std::size_t>(2, length / 4);
  if (mismatched > tolerance) {
    fail("low-T soft path must agree with hard path within few-cell tolerance");
  }

  // Soft posteriors-derived span should bracket hard span; with sharp
  // posteriors the start/end residues should match.
  if (soft_result.path.query_start != hard_result.path.query_start ||
      soft_result.path.query_end != hard_result.path.query_end ||
      soft_result.path.target_start != hard_result.path.target_start ||
      soft_result.path.target_end != hard_result.path.target_end) {
    fail("low-T soft path span must match hard path span on dominant fixture");
  }
}

void test_unit_temperature_soft_path_is_finite_and_bounded() {
  // At T=1.0 the soft path may differ from hard, but the partition must be
  // finite and posteriors-derived residue scores must be sane.
  constexpr std::size_t length = 16;
  constexpr std::size_t dimension = 8;
  EmbeddingFixture fixture =
      make_random_fixture(length, dimension, /*seed=*/0x5EED1234U);
  hiko::detail::PairwiseWorkspace workspace =
      make_prepared_workspace(length, dimension);

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = fixture.query_view();
  request.target_embedding = fixture.target_view();
  request.soft_mode = true;
  request.temperature = 1.0F;
  hiko::PairwiseResult result{};
  if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
      hiko_u::StatusCode::Ok) {
    fail("soft run at T=1.0 must succeed");
  }
  if (!std::isfinite(result.metrics.raw_sw_score)) {
    fail("soft raw_sw_score (log_partition) must be finite at T=1.0");
  }

  // Path metadata must be consistent with the path step set.
  if (result.path.aligned_pairs != result.path.steps.size()) {
    fail("aligned_pairs must equal steps.size() in soft mode");
  }
  for (const auto& step : result.path.steps) {
    if (step.query_index < 0 || step.target_index < 0) {
      fail("soft mode must not emit gap steps");
    }
    if (step.query_index < result.path.query_start ||
        step.query_index > result.path.query_end ||
        step.target_index < result.path.target_start ||
        step.target_index > result.path.target_end) {
      fail("soft step indices must lie within reported path span");
    }
  }
}

void test_soft_request_without_workspace_returns_failed_precondition() {
  constexpr std::size_t length = 6;
  constexpr std::size_t dimension = 8;
  EmbeddingFixture fixture =
      make_diagonal_dominant_fixture(length, dimension, /*seed=*/0xDEADU);

  hiko::detail::PairwiseWorkspace workspace;
  hiko::detail::PairwiseWorkspacePlan plan{};
  plan.max_query_length = length;
  plan.max_target_length = length;
  plan.embedding_dimension = dimension;
  plan.allocate_soft_sw = false;
  if (workspace.prepare(plan).code != hiko_u::StatusCode::Ok) {
    fail("hard-only workspace prepare must succeed");
  }

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = fixture.query_view();
  request.target_embedding = fixture.target_view();
  request.soft_mode = true;
  request.temperature = 1.0F;
  hiko::PairwiseResult result{};
  if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
      hiko_u::StatusCode::FailedPrecondition) {
    fail("soft request without soft workspace must return FailedPrecondition");
  }
}

void test_soft_request_with_invalid_temperature_is_rejected() {
  constexpr std::size_t length = 6;
  constexpr std::size_t dimension = 8;
  EmbeddingFixture fixture =
      make_diagonal_dominant_fixture(length, dimension, /*seed=*/0x4BU);
  hiko::detail::PairwiseWorkspace workspace =
      make_prepared_workspace(length, dimension);

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = fixture.query_view();
  request.target_embedding = fixture.target_view();
  request.soft_mode = true;
  hiko::PairwiseResult result{};

  request.temperature = 0.0F;
  if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("temperature=0 must return InvalidArgument");
  }
  request.temperature = -1.0F;
  if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("negative temperature must return InvalidArgument");
  }
  request.temperature = std::nanf("");
  if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("NaN temperature must return InvalidArgument");
  }
}

void test_hard_and_soft_share_workspace_across_calls() {
  // The same workspace prepared with allocate_soft_sw=true must serve both
  // hard and soft requests interchangeably without re-preparing.
  constexpr std::size_t length = 8;
  constexpr std::size_t dimension = 8;
  EmbeddingFixture fixture =
      make_diagonal_dominant_fixture(length, dimension, /*seed=*/0x9999U);
  hiko::detail::PairwiseWorkspace workspace =
      make_prepared_workspace(length, dimension);

  hiko::PairwiseEmbeddingRequest hard_request{};
  hard_request.query_embedding = fixture.query_view();
  hard_request.target_embedding = fixture.target_view();
  hiko::PairwiseResult hard_first{};
  if (hiko::run_pairwise_embeddings(hard_request, workspace, hard_first).code !=
      hiko_u::StatusCode::Ok) {
    fail("first hard run on shared workspace must succeed");
  }

  hiko::PairwiseEmbeddingRequest soft_request = hard_request;
  soft_request.soft_mode = true;
  soft_request.temperature = 0.5F;
  hiko::PairwiseResult soft_run{};
  if (hiko::run_pairwise_embeddings(soft_request, workspace, soft_run).code !=
      hiko_u::StatusCode::Ok) {
    fail("intermediate soft run on shared workspace must succeed");
  }

  hiko::PairwiseResult hard_second{};
  if (hiko::run_pairwise_embeddings(hard_request, workspace, hard_second).code !=
      hiko_u::StatusCode::Ok) {
    fail("second hard run on shared workspace must succeed");
  }
  if (!nearly_equal(hard_second.metrics.raw_sw_score,
                    hard_first.metrics.raw_sw_score, 1e-6)) {
    fail("hard mode determinism must survive interleaved soft mode runs");
  }
  if (hard_second.path.aligned_pairs != hard_first.path.aligned_pairs ||
      hard_second.path.query_start != hard_first.path.query_start ||
      hard_second.path.query_end != hard_first.path.query_end ||
      hard_second.path.target_start != hard_first.path.target_start ||
      hard_second.path.target_end != hard_first.path.target_end) {
    fail("hard mode path must be identical before and after soft run");
  }
}

void test_both_mode_exact_parity_with_separate_runs() {
  // `Both` must keep the hard traceback/score bit-identical to a hard-only
  // run and the soft score bit-identical to a soft-only run at T=1.0.
  constexpr std::size_t length = 18;
  constexpr std::size_t dimension = 16;
  EmbeddingFixture fixture =
      make_diagonal_dominant_fixture(length, dimension, /*seed=*/0xB01B5EEDU);

  hiko::PairwiseEmbeddingRequest hard_request{};
  hard_request.query_embedding = fixture.query_view();
  hard_request.target_embedding = fixture.target_view();
  hard_request.hard_mode = true;

  hiko::PairwiseEmbeddingRequest soft_request = hard_request;
  soft_request.hard_mode = false;
  soft_request.soft_mode = true;
  soft_request.temperature = 1.0F;

  hiko::PairwiseEmbeddingRequest both_request = hard_request;
  both_request.soft_mode = true;
  both_request.temperature = 1.0F;

  hiko::detail::PairwiseWorkspace hard_workspace =
      make_prepared_workspace(length, dimension, false);
  hiko::detail::PairwiseWorkspace soft_workspace =
      make_prepared_workspace(length, dimension, true);
  hiko::detail::PairwiseWorkspace both_workspace =
      make_prepared_workspace(length, dimension, true);

  hiko::PairwiseResult hard_result{};
  hiko::PairwiseResult soft_result{};
  hiko::PairwiseResult both_result{};
  if (hiko::run_pairwise_embeddings(hard_request, hard_workspace,
                                   hard_result).code != hiko_u::StatusCode::Ok ||
      hiko::run_pairwise_embeddings(soft_request, soft_workspace,
                                   soft_result).code != hiko_u::StatusCode::Ok ||
      hiko::run_pairwise_embeddings(both_request, both_workspace,
                                   both_result).code != hiko_u::StatusCode::Ok) {
    fail("hard, soft, and both parity runs must all return Ok");
  }

  if (both_result.raw_sw_score != hard_result.raw_sw_score ||
      both_result.metrics.raw_sw_score != hard_result.metrics.raw_sw_score) {
    fail("both-mode hard score must exactly match hard-only score");
  }
  if (!paths_equal(both_result.path, hard_result.path)) {
    fail("both-mode hard path must exactly match hard-only path");
  }
  if (!soft_result.metrics.soft_sw_score.valid ||
      soft_result.raw_sw_score != soft_result.metrics.soft_sw_score.value) {
    fail("soft-only run must mirror soft score into raw score");
  }
  if (!both_result.metrics.soft_sw_score.valid ||
      both_result.metrics.soft_sw_score.value != soft_result.raw_sw_score) {
    fail("both-mode soft score must exactly match soft-only score at T=1");
  }
}

void test_both_mode_prepared_workspace_has_no_discarded_soft_path_allocation() {
  // The `both` path reports the hard traceback as primary output; the soft
  // pass only needs its forward/backward score here. After warmup and result
  // path reservation, a repeated `both` run should not allocate a discarded
  // soft AlignmentPath.
  constexpr std::size_t length = 14;
  constexpr std::size_t dimension = 12;
  EmbeddingFixture fixture =
      make_diagonal_dominant_fixture(length, dimension, /*seed=*/0xA110CU);
  hiko::detail::PairwiseWorkspace workspace =
      make_prepared_workspace(length, dimension, true);

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = fixture.query_view();
  request.target_embedding = fixture.target_view();
  request.hard_mode = true;
  request.soft_mode = true;
  request.temperature = 1e-3F;

  hiko::PairwiseResult result{};
  result.path.steps.reserve(length + length);
  if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
      hiko_u::StatusCode::Ok) {
    fail("warmup both-mode run must succeed");
  }
  if (!result.metrics.soft_sw_score.valid) {
    fail("warmup both-mode run must populate soft score");
  }

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t iteration = 0; iteration < 8; ++iteration) {
    if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
        hiko_u::StatusCode::Ok) {
      fail("allocation-counted both-mode run must succeed");
    }
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0) {
    fail("prepared both-mode run must not allocate a discarded soft path");
  }
}

std::uint64_t time_request_microseconds(
    const hiko::PairwiseEmbeddingRequest& request,
    std::size_t length,
    std::size_t dimension,
    bool allocate_soft_sw,
    std::size_t repetitions) {
  hiko::detail::PairwiseWorkspace workspace =
      make_prepared_workspace(length, dimension, allocate_soft_sw);
  hiko::PairwiseResult result{};
  result.path.steps.reserve(length + length);
  if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
      hiko_u::StatusCode::Ok) {
    fail("timing warmup run must succeed");
  }

  const auto start = std::chrono::steady_clock::now();
  double score_sink = 0.0;
  for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
    if (hiko::run_pairwise_embeddings(request, workspace, result).code !=
        hiko_u::StatusCode::Ok) {
      fail("timed pairwise run must succeed");
    }
    score_sink += result.raw_sw_score;
    if (result.metrics.soft_sw_score.valid) {
      score_sink += result.metrics.soft_sw_score.value;
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  if (!std::isfinite(score_sink)) {
    fail("timing score sink must stay finite");
  }
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count());
}

void report_both_mode_micro_timing() {
  // Report-only micro-check for the packet log. The dimensionality makes the
  // raw-dot score-matrix build a visible cost, but CI/load variance keeps this
  // intentionally out of the pass/fail contract.
  constexpr std::size_t length = 96;
  constexpr std::size_t dimension = 256;
  constexpr std::size_t repetitions = 12;
  EmbeddingFixture fixture =
      make_random_fixture(length, dimension, /*seed=*/0x71A11C5U);

  hiko::PairwiseEmbeddingRequest hard_request{};
  hard_request.query_embedding = fixture.query_view();
  hard_request.target_embedding = fixture.target_view();
  hard_request.hard_mode = true;

  hiko::PairwiseEmbeddingRequest soft_request = hard_request;
  soft_request.hard_mode = false;
  soft_request.soft_mode = true;
  soft_request.temperature = 1.0F;

  hiko::PairwiseEmbeddingRequest both_request = hard_request;
  both_request.soft_mode = true;
  both_request.temperature = 1.0F;

  const std::uint64_t hard_us = time_request_microseconds(
      hard_request, length, dimension, false, repetitions);
  const std::uint64_t soft_us = time_request_microseconds(
      soft_request, length, dimension, true, repetitions);
  const std::uint64_t both_us = time_request_microseconds(
      both_request, length, dimension, true, repetitions);
  const std::uint64_t separate_us = hard_us + soft_us;

  std::fprintf(stderr,
               "pairwise_pipeline_soft_tests: micro-timing reps=%zu L=%zu "
               "D=%zu hard+soft=%llu us both=%llu us saved=%lld us\n",
               repetitions, length, dimension,
               static_cast<unsigned long long>(separate_us),
               static_cast<unsigned long long>(both_us),
               static_cast<long long>(separate_us) -
                   static_cast<long long>(both_us));
}

}  // namespace

int main() {
  test_default_request_runs_hard_mode_unchanged();
  test_low_temperature_soft_path_matches_hard_path();
  test_unit_temperature_soft_path_is_finite_and_bounded();
  test_soft_request_without_workspace_returns_failed_precondition();
  test_soft_request_with_invalid_temperature_is_rejected();
  test_hard_and_soft_share_workspace_across_calls();
  test_both_mode_exact_parity_with_separate_runs();
  test_both_mode_prepared_workspace_has_no_discarded_soft_path_allocation();
  report_both_mode_micro_timing();
  return 0;
}

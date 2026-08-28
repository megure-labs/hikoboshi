#include <hikoboshi/primitives/alignment/smith_waterman.hpp>
#include <hikoboshi/primitives/alignment/traceback.hpp>
#include <hikoboshi/universal/alignment_path.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko = hikoboshi::primitives::alignment;
namespace hiko_u = hikoboshi::universal;

namespace {

bool nearly_equal(float a, float b, float tolerance = 1e-5F) {
  return std::fabs(a - b) <= tolerance;
}

void fail(const char* tag) {
  std::fprintf(stderr, "primitive_alignment_goldens: %s\n", tag);
  std::exit(1);
}

struct AlignmentRun {
  hiko::SmithWatermanScalarOutput output;
  std::vector<hiko::TraceDirection> trace_match;
  std::vector<hiko::TraceDirection> trace_insert;
  std::vector<hiko::TraceDirection> trace_delete;
  std::vector<float> match_workspace;
  std::vector<float> insert_workspace;
  std::vector<float> delete_workspace;
  std::vector<hiko_u::AlignmentStep> traceback_scratch;
  hiko_u::AlignmentPath path;
};

void write_traceback_output(const hiko::TracebackScalarOutput& output,
                            hiko_u::AlignmentPath& path) {
  path.steps.clear();
  if (output.step_count != 0) {
    path.steps.assign(output.steps.data,
                      output.steps.data + output.step_count);
  }
  path.aligned_pairs = output.aligned_pairs;
  path.query_start = output.query_start;
  path.query_end = output.query_end;
  path.target_start = output.target_start;
  path.target_end = output.target_end;
}

AlignmentRun run_alignment(const std::vector<float>& scores,
                           std::size_t lq,
                           std::size_t lt,
                           float gap_open,
                           float gap_ext) {
  AlignmentRun run;
  run.trace_match.assign(lq * lt, hiko::TraceDirection::Stop);
  run.trace_insert.assign(lq * lt, hiko::TraceDirection::Stop);
  run.trace_delete.assign(lq * lt, hiko::TraceDirection::Stop);
  const std::size_t workspace_cells = (lq + 1) * (lt + 1);
  run.match_workspace.assign(workspace_cells, 0.0F);
  run.insert_workspace.assign(workspace_cells, 0.0F);
  run.delete_workspace.assign(workspace_cells, 0.0F);

  hiko::SmithWatermanScalarRequest sw_request{};
  sw_request.scores = scores.data();
  sw_request.query_length = lq;
  sw_request.target_length = lt;
  sw_request.gap_open = gap_open;
  sw_request.gap_extension = gap_ext;
  sw_request.match_workspace = run.match_workspace.data();
  sw_request.insert_workspace = run.insert_workspace.data();
  sw_request.delete_workspace = run.delete_workspace.data();
  sw_request.workspace_cells = workspace_cells;
  run.output.trace_match = run.trace_match.data();
  run.output.trace_insert = run.trace_insert.data();
  run.output.trace_delete = run.trace_delete.data();
  hiko::smith_waterman_scalar(sw_request, run.output);

  run.traceback_scratch.resize(lq + lt);
  hiko::TracebackScalarRequest tb_request{};
  tb_request.trace_match = run.trace_match.data();
  tb_request.trace_insert = run.trace_insert.data();
  tb_request.trace_delete = run.trace_delete.data();
  tb_request.scores = scores.data();
  tb_request.query_length = lq;
  tb_request.target_length = lt;
  tb_request.best_query_index = run.output.best_query_index;
  tb_request.best_target_index = run.output.best_target_index;
  tb_request.best_state = run.output.best_state;
  hiko::TracebackScalarOutput tb_output{};
  tb_output.steps = {run.traceback_scratch.data(),
                     run.traceback_scratch.size()};
  hiko::traceback_scalar(tb_request, tb_output);
  write_traceback_output(tb_output, run.path);
  return run;
}

void test_charter_default_constants() {
  if (!nearly_equal(hiko::kHardSwGapOpenDefault, -1.4F) ||
      !nearly_equal(hiko::kHardSwGapExtensionDefault, -0.15F)) {
    fail("hard SW default gap parameters must match charter constants");
  }
}

void test_zero_score_matrix_returns_zero_score() {
  std::vector<float> scores(3 * 3, 0.0F);
  const AlignmentRun run = run_alignment(scores, 3, 3,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  if (run.output.best_score != 0.0F) {
    fail("zero score matrix must yield raw SW score 0");
  }
  if (!run.path.steps.empty() || run.path.aligned_pairs != 0) {
    fail("zero score matrix must yield empty alignment path");
  }
}

void test_perfect_match_diagonal() {
  // Identity diagonal: scores[i,i] = 1, off-diagonal = -1 (forces matches).
  const std::size_t lq = 3;
  const std::size_t lt = 3;
  std::vector<float> scores(lq * lt, -1.0F);
  for (std::size_t i = 0; i < lq; ++i) {
    scores[i * lt + i] = 1.0F;
  }
  const AlignmentRun run = run_alignment(scores, lq, lt,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  if (!nearly_equal(run.output.best_score, 3.0F)) {
    fail("identity diagonal must accumulate 3 matches as raw score 3");
  }
  if (run.path.aligned_pairs != 3) {
    fail("identity diagonal must produce 3 aligned pairs");
  }
  if (run.path.steps.size() != 3) {
    fail("identity diagonal path must have 3 ordered steps");
  }
  for (std::size_t s = 0; s < run.path.steps.size(); ++s) {
    const auto& step = run.path.steps[s];
    if (step.query_index != static_cast<std::int32_t>(s) ||
        step.target_index != static_cast<std::int32_t>(s)) {
      fail("identity diagonal traceback must walk q==t in order");
    }
    if (!nearly_equal(step.residue_score, 1.0F)) {
      fail("identity diagonal step residue_score must be 1");
    }
  }
  if (run.path.query_start != 0 || run.path.target_start != 0 ||
      run.path.query_end != 2 || run.path.target_end != 2) {
    fail("identity diagonal local span endpoints must cover full diagonal");
  }
}

void test_strict_inequality_restart_wins_ties() {
  // Single positive cell with everything else zero; strict `>` keeps the
  // earliest row-major cell on ties. Since prev_match starts at 0 and
  // restart also yields 0, restart wins so the score for this cell is
  // exactly s (no extension into zero predecessors).
  const std::size_t lq = 2;
  const std::size_t lt = 2;
  std::vector<float> scores = {
      1.0F, 1.0F,
      1.0F, 1.0F,
  };
  const AlignmentRun run = run_alignment(scores, lq, lt,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  // The optimal is to walk the diagonal: M[1,1] = 1, M[2,2] = 1 + max(M[1,1], 0) = 2.
  if (!nearly_equal(run.output.best_score, 2.0F)) {
    fail("uniform score matrix must accumulate diagonal of length 2");
  }
  // Earliest row-major cell with score 2 is (1,1) i.e. q==t==1.
  if (run.output.best_query_index != 1 || run.output.best_target_index != 1) {
    fail("strict tie behavior must keep earliest best-cell row-major");
  }
}

void test_restart_on_zero_ties_keeps_stop_predecessor() {
  std::vector<float> scores(2 * 2, 0.0F);
  const AlignmentRun run = run_alignment(scores, 2, 2,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  for (hiko::TraceDirection predecessor : run.trace_match) {
    if (predecessor != hiko::TraceDirection::Stop) {
      fail("zero-valued M predecessor ties must keep restart/Stop");
    }
  }
  if (run.output.best_state != hiko::TraceDirection::Stop) {
    fail("zero-valued best-state ties must keep Stop");
  }
}

void test_match_predecessor_priority_survives_ties() {
  // At q1/t2, the diagonal M and D predecessor scores at q0/t1 are both 0.6.
  // Strict `>` keeps the earlier M predecessor.
  const std::size_t lq_md = 2;
  const std::size_t lt_md = 3;
  std::vector<float> scores_md = {
       2.0F,  0.6F, -3.0F,
      -3.0F, -3.0F,  1.0F,
  };
  const AlignmentRun run_md = run_alignment(scores_md, lq_md, lt_md,
                                            hiko::kHardSwGapOpenDefault,
                                            hiko::kHardSwGapExtensionDefault);
  if (run_md.trace_match[1 * lt_md + 2] != hiko::TraceDirection::Match) {
    fail("M/D predecessor ties must keep M priority");
  }

  // At q2/t1, the diagonal M and I predecessor scores at q1/t0 are both 0.6.
  const std::size_t lq_mi = 3;
  const std::size_t lt_mi = 2;
  std::vector<float> scores_mi = {
       2.0F, -3.0F,
       0.6F, -3.0F,
      -3.0F,  1.0F,
  };
  const AlignmentRun run_mi = run_alignment(scores_mi, lq_mi, lt_mi,
                                            hiko::kHardSwGapOpenDefault,
                                            hiko::kHardSwGapExtensionDefault);
  if (run_mi.trace_match[2 * lt_mi + 1] != hiko::TraceDirection::Match) {
    fail("M/I predecessor ties must keep M priority");
  }
}

void test_gap_open_from_match_survives_extension_ties() {
  // At q2/t0, opening from M and extending I both produce 0.45.
  const std::size_t lq_insert = 3;
  const std::size_t lt_insert = 1;
  std::vector<float> scores_insert = {
      2.0F,
      1.85F,
     -5.0F,
  };
  const AlignmentRun run_insert = run_alignment(
      scores_insert, lq_insert, lt_insert, hiko::kHardSwGapOpenDefault,
      hiko::kHardSwGapExtensionDefault);
  if (run_insert.trace_insert[2 * lt_insert] != hiko::TraceDirection::Match) {
    fail("I open/extend ties must keep gap-open-from-M");
  }

  // Transposed case for D at q0/t2.
  const std::size_t lq_delete = 1;
  const std::size_t lt_delete = 3;
  std::vector<float> scores_delete = {
      2.0F, 1.85F, -5.0F,
  };
  const AlignmentRun run_delete = run_alignment(
      scores_delete, lq_delete, lt_delete, hiko::kHardSwGapOpenDefault,
      hiko::kHardSwGapExtensionDefault);
  if (run_delete.trace_delete[2] != hiko::TraceDirection::Match) {
    fail("D open/extend ties must keep gap-open-from-M");
  }
}

void test_best_scan_covers_all_states_with_priority() {
  // Positive gap penalties are not product defaults; this crafted primitive
  // fixture makes terminal gap states observable in the best-state scan.
  std::vector<float> gap_best_scores = {
       2.0F, -10.0F,
  };
  const AlignmentRun gap_best = run_alignment(gap_best_scores, 1, 2,
                                             1.0F, 0.0F);
  if (!nearly_equal(gap_best.output.best_score, 3.0F) ||
      gap_best.output.best_query_index != 0 ||
      gap_best.output.best_target_index != 1 ||
      gap_best.output.best_state != hiko::TraceDirection::DeleteTarget) {
    fail("best scan must consider D state, not only M");
  }
  if (gap_best.path.steps.size() != 2 ||
      gap_best.path.steps[0].query_index != 0 ||
      gap_best.path.steps[0].target_index != 0 ||
      gap_best.path.steps[1].query_index != hiko_u::kAlignmentGapSentinel ||
      gap_best.path.steps[1].target_index != 1) {
    fail("traceback must start from terminal D state when D wins");
  }

  std::vector<float> insert_best_scores = {
       2.0F,
      -10.0F,
  };
  const AlignmentRun insert_best = run_alignment(insert_best_scores, 2, 1,
                                                 1.0F, 0.0F);
  if (!nearly_equal(insert_best.output.best_score, 3.0F) ||
      insert_best.output.best_query_index != 1 ||
      insert_best.output.best_target_index != 0 ||
      insert_best.output.best_state != hiko::TraceDirection::InsertQuery) {
    fail("best scan must consider I state, not only M");
  }
  if (insert_best.path.steps.size() != 2 ||
      insert_best.path.steps[0].query_index != 0 ||
      insert_best.path.steps[0].target_index != 0 ||
      insert_best.path.steps[1].query_index != 1 ||
      insert_best.path.steps[1].target_index != hiko_u::kAlignmentGapSentinel) {
    fail("traceback must start from terminal I state when I wins");
  }

  std::vector<float> same_cell_tie_scores = {
      2.0F, 3.0F,
  };
  const AlignmentRun same_cell_tie = run_alignment(same_cell_tie_scores, 1, 2,
                                                   1.0F, 0.0F);
  if (!nearly_equal(same_cell_tie.output.best_score, 3.0F) ||
      same_cell_tie.output.best_query_index != 0 ||
      same_cell_tie.output.best_target_index != 1 ||
      same_cell_tie.output.best_state != hiko::TraceDirection::Match) {
    fail("same-cell best-state ties must keep M before I/D");
  }
}

void test_gap_convention_length_one() {
  // 2x3 with a single target-side gap of length 1.
  // scores layout (q rows, t cols):
  //   q0: [ 2, -1, -1 ]
  //   q1: [-1, -1,  2 ]
  // Match scores chosen so the gap path beats the local restart:
  //   path:  q0/t0 match (2), gap-target at t1 (open),
  //          q1/t2 match (2). Total = 2 + gap_open + 2 = 2.6.
  //   restart only: best single match = 2 (loses).
  const std::size_t lq = 2;
  const std::size_t lt = 3;
  std::vector<float> scores = {
       2.0F, -1.0F, -1.0F,
      -1.0F, -1.0F,  2.0F,
  };
  const AlignmentRun run = run_alignment(scores, lq, lt,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  if (!nearly_equal(run.output.best_score, 2.6F, 1e-4F)) {
    fail("single gap of length 1 must cost gap_open exactly");
  }
  // Path order: query 0 / target 0 (match), gap (-1, 1), query 1 / target 2 (match).
  if (run.path.steps.size() != 3 || run.path.aligned_pairs != 2) {
    fail("gap-of-length-1 path must have 3 ordered steps and 2 matches");
  }
  if (run.path.steps[0].query_index != 0 || run.path.steps[0].target_index != 0) {
    fail("gap-of-length-1 first step must be q0/t0 match");
  }
  // Middle step is a delete-target gap (query gap sentinel, target index 1).
  if (run.path.steps[1].query_index != hiko_u::kAlignmentGapSentinel ||
      run.path.steps[1].target_index != 1) {
    fail("gap-of-length-1 middle step must use -1 query sentinel at t1");
  }
  if (!nearly_equal(run.path.steps[1].residue_score, 0.0F)) {
    fail("gap step residue_score must be 0.0");
  }
  if (run.path.steps[2].query_index != 1 || run.path.steps[2].target_index != 2) {
    fail("gap-of-length-1 last step must be q1/t2 match");
  }
}

void test_gap_convention_length_two() {
  // 2x4 with a single target-side gap of length 2.
  // scores layout:
  //   q0: [ 2, -1, -1, -1 ]
  //   q1: [-1, -1, -1,  2 ]
  // Score = 2 + gap_open + gap_ext + 2 = 4 - 1.4 - 0.15 = 2.45.
  const std::size_t lq = 2;
  const std::size_t lt = 4;
  std::vector<float> scores = {
       2.0F, -1.0F, -1.0F, -1.0F,
      -1.0F, -1.0F, -1.0F,  2.0F,
  };
  const AlignmentRun run = run_alignment(scores, lq, lt,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  if (!nearly_equal(run.output.best_score, 2.45F, 1e-4F)) {
    fail("gap of length 2 must apply gap_open + gap_ext exactly");
  }
  // Two gap steps in the middle (target indices 1 and 2).
  if (run.path.steps.size() != 4 || run.path.aligned_pairs != 2) {
    fail("gap-of-length-2 path must have 4 ordered steps and 2 matches");
  }
  if (run.path.steps[1].query_index != hiko_u::kAlignmentGapSentinel ||
      run.path.steps[1].target_index != 1) {
    fail("first internal gap step must mark target index 1");
  }
  if (run.path.steps[2].query_index != hiko_u::kAlignmentGapSentinel ||
      run.path.steps[2].target_index != 2) {
    fail("second internal gap step must mark target index 2");
  }
}

void test_gap_convention_length_three() {
  // Length-3 gap: 2 + gap_open + 2 * gap_ext + 2 = 4 - 1.7 = 2.3.
  const std::size_t lq = 2;
  const std::size_t lt = 5;
  std::vector<float> scores = {
       2.0F, -1.0F, -1.0F, -1.0F, -1.0F,
      -1.0F, -1.0F, -1.0F, -1.0F,  2.0F,
  };
  const AlignmentRun run = run_alignment(scores, lq, lt,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  if (!nearly_equal(run.output.best_score, 2.3F, 1e-4F)) {
    fail("gap of length 3 must apply gap_open + 2*gap_ext");
  }
  if (run.path.steps.size() != 5 || run.path.aligned_pairs != 2) {
    fail("gap-of-length-3 path must have 5 ordered steps");
  }
}

void test_negative_only_matrix_yields_zero_score() {
  std::vector<float> scores(2 * 2, -1.0F);
  const AlignmentRun run = run_alignment(scores, 2, 2,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  if (run.output.best_score != 0.0F) {
    fail("strictly negative scores must produce raw score 0 (local SW floor)");
  }
  if (!run.path.steps.empty() || run.path.aligned_pairs != 0) {
    fail("strictly negative scores must produce an empty alignment path");
  }
}

void test_query_insert_uses_query_sentinel_on_target() {
  // Force a query-side insertion (extra query residue between two matches).
  // scores layout:
  //   q0: [ 2, -1 ]
  //   q1: [-1, -1 ]
  //   q2: [-1,  2 ]
  // Optimal is q0->t0, gap query (q1 -> -1), q2->t1. The middle step has
  // target_index == kAlignmentGapSentinel and a real query_index = 1.
  const std::size_t lq = 3;
  const std::size_t lt = 2;
  std::vector<float> scores = {
       2.0F, -1.0F,
      -1.0F, -1.0F,
      -1.0F,  2.0F,
  };
  const AlignmentRun run = run_alignment(scores, lq, lt,
                                         hiko::kHardSwGapOpenDefault,
                                         hiko::kHardSwGapExtensionDefault);
  if (run.path.steps.size() != 3 || run.path.aligned_pairs != 2) {
    fail("query-side gap path must have 3 ordered steps and 2 matches");
  }
  if (run.path.steps[1].query_index != 1 ||
      run.path.steps[1].target_index != hiko_u::kAlignmentGapSentinel) {
    fail("query-side gap middle step must use -1 target sentinel");
  }
}

}  // namespace

int main() {
  test_charter_default_constants();
  test_zero_score_matrix_returns_zero_score();
  test_perfect_match_diagonal();
  test_strict_inequality_restart_wins_ties();
  test_restart_on_zero_ties_keeps_stop_predecessor();
  test_match_predecessor_priority_survives_ties();
  test_gap_open_from_match_survives_extension_ties();
  test_best_scan_covers_all_states_with_priority();
  test_gap_convention_length_one();
  test_gap_convention_length_two();
  test_gap_convention_length_three();
  test_negative_only_matrix_yields_zero_score();
  test_query_insert_uses_query_sentinel_on_target();
  return 0;
}

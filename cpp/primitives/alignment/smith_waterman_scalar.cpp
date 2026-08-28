#include <hikoboshi/primitives/alignment/smith_waterman.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace hikoboshi::primitives::alignment {

namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

void clear_tracebacks(TraceDirection* trace_match,
                      TraceDirection* trace_insert,
                      TraceDirection* trace_delete,
                      std::size_t cell_count) {
  for (std::size_t i = 0; i < cell_count; ++i) {
    trace_match[i] = TraceDirection::Stop;
    trace_insert[i] = TraceDirection::Stop;
    trace_delete[i] = TraceDirection::Stop;
  }
}

}  // namespace

void smith_waterman_scalar(const SmithWatermanScalarRequest& request,
                           SmithWatermanScalarOutput& output) {
  const std::size_t lq = request.query_length;
  const std::size_t lt = request.target_length;
  const std::size_t trace_cells = lq * lt;

  output.best_score = 0.0F;
  output.best_query_index = -1;
  output.best_target_index = -1;
  output.best_state = TraceDirection::Stop;

  if (lq == 0 || lt == 0) {
    clear_tracebacks(output.trace_match, output.trace_insert, output.trace_delete,
                     trace_cells);
    return;
  }

  const float gap_open = request.gap_open;
  const float gap_extension = request.gap_extension;

  // DP cells use one-based indexing internally; layout (lq + 1) x (lt + 1).
  const std::size_t row_stride = lt + 1;
  const std::size_t workspace_cells = (lq + 1) * row_stride;
  if (request.match_workspace == nullptr || request.insert_workspace == nullptr ||
      request.delete_workspace == nullptr ||
      request.workspace_cells < workspace_cells) {
    clear_tracebacks(output.trace_match, output.trace_insert, output.trace_delete,
                     trace_cells);
    return;
  }

  float* match = request.match_workspace;
  float* insert_q = request.insert_workspace;
  float* delete_t = request.delete_workspace;

  // Boundary: row 0 and column 0 of M are 0; I/D are -inf. Interior
  // cells are overwritten before being read, so the hot path avoids clearing
  // the full caller-owned workspace each run.
  for (std::size_t j = 0; j <= lt; ++j) {
    match[j] = 0.0F;
    insert_q[j] = kNegativeInfinity;
    delete_t[j] = kNegativeInfinity;
  }

  float best_score = 0.0F;
  std::int32_t best_i = -1;
  std::int32_t best_j = -1;
  TraceDirection best_state = TraceDirection::Stop;
  for (std::size_t i = 1; i <= lq; ++i) {
    const std::size_t row = i * row_stride;
    const std::size_t prev_row = (i - 1) * row_stride;
    match[row] = 0.0F;
    insert_q[row] = kNegativeInfinity;
    delete_t[row] = kNegativeInfinity;

    for (std::size_t j = 1; j <= lt; ++j) {
      const std::size_t trace_index = (i - 1) * lt + (j - 1);
      const float s = request.scores[trace_index];

      // M predecessor: pick best of M, I, D, 0 (restart). Strict `>` keeps
      // the earlier predecessor (restart wins ties per HARD_SW_CHARTER).
      const std::size_t diag = prev_row + (j - 1);
      const float prev_match = match[diag];
      const float prev_insert = insert_q[diag];
      const float prev_delete = delete_t[diag];

      TraceDirection match_pred = TraceDirection::Stop;
      float best_pred = 0.0F;
      if (prev_match > best_pred) {
        best_pred = prev_match;
        match_pred = TraceDirection::Match;
      }
      if (prev_insert > best_pred) {
        best_pred = prev_insert;
        match_pred = TraceDirection::InsertQuery;
      }
      if (prev_delete > best_pred) {
        best_pred = prev_delete;
        match_pred = TraceDirection::DeleteTarget;
      }
      const float new_match = s + best_pred;
      const std::size_t cell = row + j;
      match[cell] = new_match;
      output.trace_match[trace_index] = match_pred;

      // I predecessor: gap open from M wins ties over gap extension.
      const float open_from_match = match[prev_row + j] + gap_open;
      const float extend_insert = insert_q[prev_row + j] + gap_extension;
      TraceDirection insert_pred = TraceDirection::Match;
      float new_insert = open_from_match;
      if (extend_insert > new_insert) {
        new_insert = extend_insert;
        insert_pred = TraceDirection::InsertQuery;
      }
      insert_q[cell] = new_insert;
      output.trace_insert[trace_index] = insert_pred;

      // D predecessor: same convention; gap open from M wins ties.
      const float open_from_match_d = match[row + (j - 1)] + gap_open;
      const float extend_delete = delete_t[row + (j - 1)] + gap_extension;
      TraceDirection delete_pred = TraceDirection::Match;
      float new_delete = open_from_match_d;
      if (extend_delete > new_delete) {
        new_delete = extend_delete;
        delete_pred = TraceDirection::DeleteTarget;
      }
      delete_t[cell] = new_delete;
      output.trace_delete[trace_index] = delete_pred;

      // Best-cell scan is fused with the recurrence. Row-major iteration and
      // strict `>` preserve earliest-cell ties; checking M, I, then D preserves
      // chartered state priority within a cell.
      if (new_match > best_score) {
        best_score = new_match;
        best_i = static_cast<std::int32_t>(i - 1);
        best_j = static_cast<std::int32_t>(j - 1);
        best_state = TraceDirection::Match;
      }
      if (new_insert > best_score) {
        best_score = new_insert;
        best_i = static_cast<std::int32_t>(i - 1);
        best_j = static_cast<std::int32_t>(j - 1);
        best_state = TraceDirection::InsertQuery;
      }
      if (new_delete > best_score) {
        best_score = new_delete;
        best_i = static_cast<std::int32_t>(i - 1);
        best_j = static_cast<std::int32_t>(j - 1);
        best_state = TraceDirection::DeleteTarget;
      }
    }
  }

  output.best_score = best_score;
  output.best_query_index = best_i;
  output.best_target_index = best_j;
  output.best_state = best_state;
}

}  // namespace hikoboshi::primitives::alignment

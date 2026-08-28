#include <hikoboshi/primitives/alignment/traceback.hpp>

#include <cstddef>
#include <cstdint>

#include <hikoboshi/primitives/alignment/smith_waterman.hpp>

namespace hikoboshi::primitives::alignment {
namespace {

using hikoboshi::universal::AlignmentStep;
using hikoboshi::universal::kAlignmentGapSentinel;

}  // namespace

void traceback_scalar(const TracebackScalarRequest& request,
                      TracebackScalarOutput& output) noexcept {
  output.step_count = 0;
  output.aligned_pairs = 0;
  output.query_start = kAlignmentGapSentinel;
  output.query_end = kAlignmentGapSentinel;
  output.target_start = kAlignmentGapSentinel;
  output.target_end = kAlignmentGapSentinel;

  AlignmentStep* steps = output.steps.data;
  const std::size_t capacity = output.steps.size;
  if (request.best_query_index < 0 || request.best_target_index < 0 ||
      request.best_state == TraceDirection::Stop || steps == nullptr) {
    return;
  }

  const std::size_t lt = request.target_length;
  const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(lt);
  std::int32_t i = request.best_query_index;
  std::int32_t j = request.best_target_index;
  std::ptrdiff_t cell =
      static_cast<std::ptrdiff_t>(i) * stride + static_cast<std::ptrdiff_t>(j);
  std::size_t count = 0;
  std::size_t aligned_pairs = 0;
  std::int32_t query_start = kAlignmentGapSentinel;
  std::int32_t target_start = kAlignmentGapSentinel;
  const std::int32_t query_end = i;
  const std::int32_t target_end = j;

  TraceDirection state = request.best_state;
  while (i >= 0 && j >= 0) {
    if (count == capacity) {
      return;
    }
    const std::size_t trace_index = static_cast<std::size_t>(cell);
    AlignmentStep& step = steps[count];

    if (state == TraceDirection::Match) {
      step.query_index = i;
      step.target_index = j;
      step.residue_score = request.scores[trace_index];
      ++count;
      ++aligned_pairs;
      query_start = i;
      target_start = j;
      const TraceDirection predecessor = request.trace_match[trace_index];
      if (predecessor == TraceDirection::Stop) {
        break;
      }
      state = predecessor;
      --i;
      --j;
      cell -= stride + 1;
    } else if (state == TraceDirection::InsertQuery) {
      step.query_index = i;
      step.target_index = kAlignmentGapSentinel;
      step.residue_score = 0.0F;
      ++count;
      const TraceDirection predecessor = request.trace_insert[trace_index];
      state = predecessor == TraceDirection::InsertQuery
                  ? TraceDirection::InsertQuery
                  : TraceDirection::Match;
      --i;
      cell -= stride;
    } else {
      step.query_index = kAlignmentGapSentinel;
      step.target_index = j;
      step.residue_score = 0.0F;
      ++count;
      const TraceDirection predecessor = request.trace_delete[trace_index];
      state = predecessor == TraceDirection::DeleteTarget
                  ? TraceDirection::DeleteTarget
                  : TraceDirection::Match;
      --j;
      --cell;
    }
  }

  output.step_count = count;
  output.aligned_pairs = aligned_pairs;
  output.query_start = query_start;
  output.query_end = query_end;
  output.target_start = target_start;
  output.target_end = target_end;
  std::size_t left = 0;
  std::size_t right = count;
  while (left < right) {
    --right;
    if (left >= right) {
      break;
    }
    const AlignmentStep temp = steps[left];
    steps[left] = steps[right];
    steps[right] = temp;
    ++left;
  }
}

}  // namespace hikoboshi::primitives::alignment

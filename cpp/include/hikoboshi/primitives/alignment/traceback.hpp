#ifndef HIKOBOSHI_PRIMITIVES_ALIGNMENT_TRACEBACK_HPP
#define HIKOBOSHI_PRIMITIVES_ALIGNMENT_TRACEBACK_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/primitives/alignment/smith_waterman.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::primitives::alignment {

struct TracebackScalarRequest {
  const TraceDirection* trace_match;
  const TraceDirection* trace_insert;
  const TraceDirection* trace_delete;
  const float* scores;
  std::size_t query_length;
  std::size_t target_length;
  std::int32_t best_query_index;
  std::int32_t best_target_index;
  TraceDirection best_state = TraceDirection::Match;
};

struct TracebackScalarOutput {
  hikoboshi::universal::Span<hikoboshi::universal::AlignmentStep> steps;
  std::size_t step_count = 0;
  std::size_t aligned_pairs = 0;
  std::int32_t query_start = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t query_end = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t target_start = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t target_end = hikoboshi::universal::kAlignmentGapSentinel;
};

void traceback_scalar(const TracebackScalarRequest& request,
                      TracebackScalarOutput& output) noexcept;

}  // namespace hikoboshi::primitives::alignment

#endif  // HIKOBOSHI_PRIMITIVES_ALIGNMENT_TRACEBACK_HPP

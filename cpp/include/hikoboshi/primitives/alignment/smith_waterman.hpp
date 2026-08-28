#ifndef HIKOBOSHI_PRIMITIVES_ALIGNMENT_SMITH_WATERMAN_HPP
#define HIKOBOSHI_PRIMITIVES_ALIGNMENT_SMITH_WATERMAN_HPP

#include <cstddef>
#include <cstdint>

namespace hikoboshi::primitives::alignment {

inline constexpr float kHardSwGapOpenDefault = -1.4F;
inline constexpr float kHardSwGapExtensionDefault = -0.15F;

enum class TraceDirection : std::uint8_t {
  Stop = 0,
  Match = 1,
  InsertQuery = 2,
  DeleteTarget = 3,
};

struct SmithWatermanScalarRequest {
  const float* scores;
  std::size_t query_length;
  std::size_t target_length;
  float gap_open;
  float gap_extension;
  float* match_workspace;
  float* insert_workspace;
  float* delete_workspace;
  std::size_t workspace_cells;
};

struct SmithWatermanScalarOutput {
  TraceDirection* trace_match;
  TraceDirection* trace_insert;
  TraceDirection* trace_delete;
  float best_score;
  std::int32_t best_query_index;
  std::int32_t best_target_index;
  TraceDirection best_state;
};

void smith_waterman_scalar(const SmithWatermanScalarRequest& request,
                           SmithWatermanScalarOutput& output);

struct SoftSmithWatermanScalarRequest {
  const float* scores;
  std::size_t query_length;
  std::size_t target_length;
  float gap_open;
  float gap_extension;
  float temperature;
  float* match_workspace;
  float* insert_workspace;
  float* delete_workspace;
  std::size_t workspace_cells;
  float* match_grad_workspace;
  float* insert_grad_workspace;
  float* delete_grad_workspace;
};

struct SoftSmithWatermanScalarOutput {
  float log_partition;
  float* posteriors;
};

void soft_smith_waterman_scalar(const SoftSmithWatermanScalarRequest& request,
                                SoftSmithWatermanScalarOutput& output);

}  // namespace hikoboshi::primitives::alignment

#endif  // HIKOBOSHI_PRIMITIVES_ALIGNMENT_SMITH_WATERMAN_HPP

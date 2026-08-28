#ifndef HIKOBOSHI_UNIVERSAL_ALIGNMENT_PATH_HPP
#define HIKOBOSHI_UNIVERSAL_ALIGNMENT_PATH_HPP

/// @file
/// Public hard local affine Smith-Waterman path records.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hikoboshi::universal {

/// Sentinel used when one side of an alignment step is a gap.
inline constexpr std::int32_t kAlignmentGapSentinel = -1;

/// One ordered step in a public hard local affine Smith-Waterman path.
///
/// Indices are zero-based residue positions in the query and target inputs.
/// A value of `kAlignmentGapSentinel` marks the gap side of an insertion or
/// deletion. `residue_score` is the aligned residue-pair score for match
/// steps and is zero for gap steps; it is not a decomposition of the total SW
/// score.
struct AlignmentStep {
  std::int32_t query_index = kAlignmentGapSentinel;
  std::int32_t target_index = kAlignmentGapSentinel;
  float residue_score = 0.0F;
};

/// Local alignment path returned by pairwise and all-vs-all workflows.
///
/// `steps` are stored in alignment order. The start and end coordinates
/// describe the aligned local span, not the full input length. Empty or
/// unavailable paths keep their coordinate fields at `kAlignmentGapSentinel`.
struct AlignmentPath {
  std::vector<AlignmentStep> steps;
  std::size_t aligned_pairs = 0;
  std::int32_t query_start = kAlignmentGapSentinel;
  std::int32_t query_end = kAlignmentGapSentinel;
  std::int32_t target_start = kAlignmentGapSentinel;
  std::int32_t target_end = kAlignmentGapSentinel;
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_ALIGNMENT_PATH_HPP

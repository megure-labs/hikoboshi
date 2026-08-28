#ifndef HIKOBOSHI_API_SCORE_ALIGNMENT_HPP
#define HIKOBOSHI_API_SCORE_ALIGNMENT_HPP

/// @file
/// Public score-only entry point for externally-supplied alignments.
///
/// `hikoboshi::api::score_alignment` runs Hikoboshi's metric panel on a pair of
/// structures plus a caller-supplied `AlignmentPath`. It is the same set of
/// metrics the `pairwise` workflow returns, exposed without invoking Hikoboshi's
/// own alignment, so external alignment tools (TMalign, US-align, Foldseek,
/// DALI, MUSTANG, ...) can be scored through a single common metric pipeline
/// for cross-tool comparison.

#include <cstddef>

#include <hikoboshi/api/results.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::api {

/// Borrowed inputs for `score_alignment`.
///
/// `query_structure` and `target_structure` are normalized Hikoboshi structure
/// views. `correspondences` is an `AlignmentPath` whose steps reference the
/// caller's residue tables using zero-based indices; the caller owns all
/// referenced storage. `kAlignmentGapSentinel` may appear on either side of a
/// step to mark gaps. Match steps must satisfy
/// `0 <= query_index < query_structure.residue_count` and the analogous bound
/// for `target_index`; out-of-range match indices return
/// `InvalidArgument`.
struct ScoreAlignmentRequest {
  universal::StructureView query_structure{};
  universal::StructureView target_structure{};
  universal::AlignmentPath correspondences{};
};

/// Metrics returned by `score_alignment`.
///
/// Mirrors the metric fields of `PairwiseMetrics` (`raw_sw_score` excluded;
/// no Smith-Waterman recurrence is run). Bit-equality with `PairwiseMetrics`
/// holds when `correspondences` is the path the `pairwise` workflow
/// produced for the same `(query_structure, target_structure)` inputs:
///
///     pairwise(Q, T)            -> PairwiseResult{path, metrics}
///     score_alignment(Q, T, path) -> ScoreAlignmentResult{metrics, ...}
///
/// every `MetricValue` field equals its `PairwiseMetrics` counterpart.
/// `ecs` is reserved/unimplemented and returned as
/// `MetricInvalidReason::Unimplemented`.
struct ScoreAlignmentResult {
  universal::MetricValue rmsd{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue tm_score_query{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue tm_score_target{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt_byA{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt_byB{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt_aln{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue identity{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_query{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_target{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_mean{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_byA{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_byB{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue ecs{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  std::size_t aligned_pairs = 0;
};

/// Score an externally-supplied alignment.
///
/// Validates the structures and correspondences, then runs the same metric
/// pipeline the `pairwise` workflow uses for result assembly. Returns
/// `InvalidArgument` for empty structures, invalid coordinate spans, or
/// out-of-range correspondence indices on match steps. Empty correspondences
/// are accepted; metrics that require aligned pairs return
/// `MetricInvalidReason::InsufficientAlignedPairs`.
[[nodiscard]] universal::Status score_alignment(
    const ScoreAlignmentRequest& request,
    ScoreAlignmentResult& result);

}  // namespace hikoboshi::api

#endif  // HIKOBOSHI_API_SCORE_ALIGNMENT_HPP

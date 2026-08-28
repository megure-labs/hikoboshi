#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_RESOLVED_ALIGNMENT_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_RESOLVED_ALIGNMENT_HPP

#include <cmath>

#include <hikoboshi/algorithms/detail/pairwise_workspace.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::algorithms::detail {

struct ResolvedAlignmentProblem {
  hikoboshi::universal::ScoreMatrixView scores;
  hikoboshi::universal::ScoreSemantics semantics;
  hikoboshi::universal::AffineGapModel gaps;
  hikoboshi::universal::AlignmentAlgorithmId algorithm;
  hikoboshi::universal::TracebackPolicy traceback;
  hikoboshi::universal::StructureView query_structure;
  hikoboshi::universal::StructureView target_structure;
};

inline constexpr hikoboshi::universal::AlignmentAlgorithmId
    kHardLocalAffineSwV1Algorithm =
        hikoboshi::universal::AlignmentAlgorithmId::HardLocalAffineSwV1;

inline constexpr hikoboshi::universal::TracebackPolicy
    kPublicPairwiseTracebackPolicy =
        hikoboshi::universal::TracebackPolicy::RequiredForPublicPairwise;

inline constexpr hikoboshi::universal::TracebackPolicy
    kPublicAllVsAllTracebackPolicy =
        hikoboshi::universal::TracebackPolicy::RequiredForPublicAllVsAll;

inline constexpr bool is_raw_dot_v1_score_semantics(
    const hikoboshi::universal::ScoreSemantics semantics) noexcept {
  return semantics.dtype == hikoboshi::universal::DataType::Float32 &&
         semantics.layout ==
             hikoboshi::universal::ScoreMatrixLayout::RowMajorQueryByTarget &&
         semantics.higher_is_better && semantics.local_affine_additive &&
         semantics.normalization == hikoboshi::universal::ScoreNormalization::None &&
         semantics.scale_family == hikoboshi::universal::ScoreScaleFamily::RawDot &&
         semantics.method == hikoboshi::universal::ScoreMethod::RawDotV1;
}

inline constexpr bool is_contiguous_score_matrix(
    const hikoboshi::universal::ScoreMatrixView scores) noexcept {
  return scores.values != nullptr && scores.query_length != 0 &&
         scores.target_length != 0 && scores.row_stride == scores.target_length;
}

inline constexpr bool is_hard_sw_default_affine_gap_model(
    const hikoboshi::universal::AffineGapModel gaps) noexcept {
  return gaps.model == hikoboshi::universal::GapModel::Affine &&
         gaps.gap_open == -1.4F && gaps.gap_extension == -0.15F &&
         gaps.convention == hikoboshi::universal::GapConvention::
                                GapOpenPlusKMinusOneGapExtension;
}

inline bool is_supported_affine_gap_model(
    const hikoboshi::universal::AffineGapModel gaps) noexcept {
  return gaps.model == hikoboshi::universal::GapModel::Affine &&
         std::isfinite(gaps.gap_open) &&
         std::isfinite(gaps.gap_extension) &&
         gaps.convention == hikoboshi::universal::GapConvention::
                                GapOpenPlusKMinusOneGapExtension;
}

inline constexpr bool is_public_traceback_required(
    const hikoboshi::universal::TracebackPolicy traceback) noexcept {
  return traceback ==
             hikoboshi::universal::TracebackPolicy::RequiredForPublicPairwise ||
         traceback ==
             hikoboshi::universal::TracebackPolicy::RequiredForPublicAllVsAll;
}

inline bool is_supported_0_1_alignment_problem(
    const ResolvedAlignmentProblem& problem) noexcept {
  return is_contiguous_score_matrix(problem.scores) &&
         is_raw_dot_v1_score_semantics(problem.semantics) &&
         is_supported_affine_gap_model(problem.gaps) &&
         problem.algorithm == kHardLocalAffineSwV1Algorithm &&
         is_public_traceback_required(problem.traceback);
}

[[nodiscard]] hikoboshi::universal::Status run_resolved_alignment_problem(
    const ResolvedAlignmentProblem& problem,
    PairwiseWorkspace& workspace,
    hikoboshi::universal::AlignmentPath& path,
    double& raw_sw_score);

}  // namespace hikoboshi::algorithms::detail

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_RESOLVED_ALIGNMENT_HPP

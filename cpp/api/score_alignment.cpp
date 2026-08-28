#include <hikoboshi/api/score_alignment.hpp>

#include <hikoboshi/algorithms/metrics.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/structure.hpp>

#include <cstddef>

namespace hikoboshi::api {
namespace {

constexpr std::size_t structure_coordinate_count(std::size_t residue_count) noexcept {
  return residue_count * universal::kCanonicalAtomCount *
         universal::kCoordinateAxisCount;
}

constexpr std::size_t structure_atom_source_count(std::size_t residue_count) noexcept {
  return residue_count * universal::kCanonicalAtomCount;
}

universal::Status validate_structure_view(
    const universal::StructureView& structure) noexcept {
  if (structure.residue_count == 0) {
    return universal::invalid_argument_status(
        "score_alignment structure must contain at least one residue");
  }
  if (structure.coordinates.data == nullptr ||
      structure.coordinates.size <
          structure_coordinate_count(structure.residue_count)) {
    return universal::invalid_argument_status(
        "score_alignment structure coordinates are invalid");
  }
  if (structure.atom_sources.data == nullptr ||
      structure.atom_sources.size <
          structure_atom_source_count(structure.residue_count)) {
    return universal::invalid_argument_status(
        "score_alignment structure atom sources are invalid");
  }
  return universal::ok_status();
}

// Reject correspondences that reference a residue index outside the supplied
// structure. Gap-side sentinels are skipped. The aligned_pairs field is also
// recomputed from the steps so callers cannot smuggle a mismatched count
// into the result panel.
universal::Status validate_correspondences(
    const universal::AlignmentPath& path,
    const universal::StructureView& query,
    const universal::StructureView& target,
    std::size_t& aligned_pair_count) noexcept {
  aligned_pair_count = 0;
  for (const universal::AlignmentStep& step : path.steps) {
    const bool query_gap = step.query_index == universal::kAlignmentGapSentinel;
    const bool target_gap = step.target_index == universal::kAlignmentGapSentinel;
    if (query_gap && target_gap) {
      return universal::invalid_argument_status(
          "score_alignment correspondence step has gap on both sides");
    }
    if (!query_gap) {
      if (step.query_index < 0) {
        return universal::invalid_argument_status(
            "score_alignment query index must be non-negative or the gap sentinel");
      }
      if (static_cast<std::size_t>(step.query_index) >= query.residue_count) {
        return universal::invalid_argument_status(
            "score_alignment query index out of range for query structure");
      }
    }
    if (!target_gap) {
      if (step.target_index < 0) {
        return universal::invalid_argument_status(
            "score_alignment target index must be non-negative or the gap sentinel");
      }
      if (static_cast<std::size_t>(step.target_index) >= target.residue_count) {
        return universal::invalid_argument_status(
            "score_alignment target index out of range for target structure");
      }
    }
    if (!query_gap && !target_gap) {
      ++aligned_pair_count;
    }
  }
  return universal::ok_status();
}

ScoreAlignmentResult to_score_result(const algorithms::MetricBlock& block,
                                     std::size_t aligned_pair_count) noexcept {
  ScoreAlignmentResult out{};
  out.rmsd = block.rmsd;
  out.tm_score_query = block.tm_score_query;
  out.tm_score_target = block.tm_score_target;
  out.lddt = block.lddt;
  out.lddt_byA = block.lddt_byA;
  out.lddt_byB = block.lddt_byB;
  out.lddt_aln = block.lddt_aln;
  out.identity = block.identity;
  out.coverage_query = block.coverage_query;
  out.coverage_target = block.coverage_target;
  out.coverage_mean = block.coverage_mean;
  out.coverage_byA = block.coverage_byA;
  out.coverage_byB = block.coverage_byB;
  out.ecs = block.ecs;
  out.aligned_pairs = aligned_pair_count;
  return out;
}

}  // namespace

universal::Status score_alignment(const ScoreAlignmentRequest& request,
                                  ScoreAlignmentResult& result) {
  result = ScoreAlignmentResult{};

  universal::Status status = validate_structure_view(request.query_structure);
  if (!universal::is_ok(status)) {
    return status;
  }
  status = validate_structure_view(request.target_structure);
  if (!universal::is_ok(status)) {
    return status;
  }

  std::size_t aligned_pair_count = 0;
  status = validate_correspondences(request.correspondences,
                                    request.query_structure,
                                    request.target_structure,
                                    aligned_pair_count);
  if (!universal::is_ok(status)) {
    return status;
  }

  // Mirror the path here so the metric block is computed from a path whose
  // `aligned_pairs` field reflects the steps actually walked, not whatever
  // number the caller stored in the request. Empty embedding views fall back
  // to structure-side metadata inside `compute_metric_block` (lengths come
  // from the structure, identity uses structure residue codes).
  universal::AlignmentPath path = request.correspondences;
  path.aligned_pairs = aligned_pair_count;

  const universal::EmbeddingView empty_embedding{};
  const algorithms::MetricBlock block = algorithms::compute_metric_block(
      path,
      0.0,
      empty_embedding,
      empty_embedding,
      request.query_structure,
      request.target_structure);
  result = to_score_result(block, aligned_pair_count);
  return universal::ok_status();
}

}  // namespace hikoboshi::api

#include <hikoboshi/algorithms/pairwise.hpp>

#include <cstddef>
#include <cstdlib>

namespace hikoboshi::algorithms {
namespace {

hikoboshi::universal::Span<const char> best_codes(
    const hikoboshi::universal::EmbeddingView& embedding,
    const hikoboshi::universal::StructureView& structure) noexcept {
  if (embedding.residue_codes.data != nullptr &&
      embedding.residue_codes.size >= embedding.residue_count) {
    return embedding.residue_codes;
  }
  if (structure.residue_codes.data != nullptr &&
      structure.residue_codes.size >= structure.residue_count) {
    return structure.residue_codes;
  }
  return {nullptr, 0};
}

std::size_t query_length(const hikoboshi::universal::EmbeddingView& embedding,
                         const hikoboshi::universal::StructureView& structure) noexcept {
  return embedding.residue_count != 0 ? embedding.residue_count
                                      : structure.residue_count;
}

bool geometry_disabled() noexcept {
  static const bool disabled = [] {
    const char* v = std::getenv("HIKOBOSHI_SKIP_GEOMETRY");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return disabled;
}

}  // namespace

MetricBlock compute_metric_block(
    const hikoboshi::universal::AlignmentPath& path,
    double raw_sw_score,
    const hikoboshi::universal::EmbeddingView& query_embedding,
    const hikoboshi::universal::EmbeddingView& target_embedding,
    const hikoboshi::universal::StructureView& query_structure,
    const hikoboshi::universal::StructureView& target_structure) noexcept {
  MetricBlock block{};
  block.raw_sw_score = raw_sw_score;
  block.soft_sw_score =
      invalid_metric(hikoboshi::universal::MetricInvalidReason::Unavailable);
  block.aligned_pairs = path.aligned_pairs;

  const std::size_t q_len = query_length(query_embedding, query_structure);
  const std::size_t t_len = query_length(target_embedding, target_structure);

  const CoverageMetrics coverage = compute_coverage(path, q_len, t_len);
  block.coverage_query = coverage.query;
  block.coverage_target = coverage.target;
  block.coverage_mean = coverage.mean;

  block.identity =
      compute_identity(path, best_codes(query_embedding, query_structure),
                       best_codes(target_embedding, target_structure));
  // Coordinate-derived metrics. Skipped under HIKOBOSHI_SKIP_GEOMETRY so a
  // benchmark wall measures alignment only -- parity with aligners that emit a
  // correspondence and nothing else. lDDT is the expensive one: O(L^2) per
  // pair, twice.
  if (geometry_disabled()) {
    const auto unavailable =
        invalid_metric(hikoboshi::universal::MetricInvalidReason::Unavailable);
    block.rmsd = unavailable;
    block.tm_score_query = unavailable;
    block.tm_score_target = unavailable;
    block.lddt = unavailable;
    block.lddt_byA = unavailable;
    block.lddt_byB = unavailable;
    block.lddt_aln = unavailable;
    block.coverage_byA = unavailable;
    block.coverage_byB = unavailable;
  } else {
    block.rmsd = compute_rmsd(path, query_structure, target_structure);

    const TmScoreMetrics tm =
        compute_tm_scores(path, query_structure, target_structure, q_len,
                          t_len);
    block.tm_score_query = tm.query_norm;
    block.tm_score_target = tm.target_norm;
    const LddtMetrics lddt =
        compute_lddt(path, query_structure, target_structure);
    block.lddt = lddt.lddt;
    block.lddt_byA = lddt.lddt_byA;
    block.lddt_byB = lddt.lddt_byB;
    block.lddt_aln = lddt.lddt_aln;
    block.coverage_byA = lddt.coverage_byA;
    block.coverage_byB = lddt.coverage_byB;
  }
  block.ecs =
      invalid_metric(hikoboshi::universal::MetricInvalidReason::Unimplemented);
  return block;
}

void assemble_pairwise_result(
    double raw_sw_score,
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::EmbeddingView& query_embedding,
    const hikoboshi::universal::EmbeddingView& target_embedding,
    const hikoboshi::universal::StructureView& query_structure,
    const hikoboshi::universal::StructureView& target_structure,
    PairwiseResult& result) noexcept {
  result.raw_sw_score = raw_sw_score;
  if (&result.path != &path) {
    result.path = path;
  }
  result.metrics =
      compute_metric_block(result.path, raw_sw_score, query_embedding,
                           target_embedding, query_structure, target_structure);
}

}  // namespace hikoboshi::algorithms

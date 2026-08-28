#ifndef HIKOBOSHI_ALGORITHMS_METRICS_HPP
#define HIKOBOSHI_ALGORITHMS_METRICS_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::algorithms {

struct CoverageMetrics {
  hikoboshi::universal::MetricValue query;
  hikoboshi::universal::MetricValue target;
  hikoboshi::universal::MetricValue mean;
};

struct TmScoreMetrics {
  hikoboshi::universal::MetricValue query_norm;
  hikoboshi::universal::MetricValue target_norm;
};

struct LddtMetrics {
  hikoboshi::universal::MetricValue lddt;
  hikoboshi::universal::MetricValue lddt_byA;
  hikoboshi::universal::MetricValue lddt_byB;
  hikoboshi::universal::MetricValue lddt_aln;
  hikoboshi::universal::MetricValue coverage_byA;
  hikoboshi::universal::MetricValue coverage_byB;
};

struct MetricBlock {
  double raw_sw_score = 0.0;
  hikoboshi::universal::MetricValue soft_sw_score{
      0.0, false, hikoboshi::universal::MetricInvalidReason::Unavailable};
  std::size_t aligned_pairs = 0;
  hikoboshi::universal::MetricValue coverage_query;
  hikoboshi::universal::MetricValue coverage_target;
  hikoboshi::universal::MetricValue coverage_mean;
  hikoboshi::universal::MetricValue identity;
  hikoboshi::universal::MetricValue rmsd;
  hikoboshi::universal::MetricValue tm_score_query;
  hikoboshi::universal::MetricValue tm_score_target;
  hikoboshi::universal::MetricValue lddt;
  hikoboshi::universal::MetricValue lddt_byA;
  hikoboshi::universal::MetricValue lddt_byB;
  hikoboshi::universal::MetricValue lddt_aln;
  hikoboshi::universal::MetricValue coverage_byA;
  hikoboshi::universal::MetricValue coverage_byB;
  hikoboshi::universal::MetricValue ecs;
};

struct Point3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct AlignedCaPair {
  Point3 query{};
  Point3 target{};
};

struct AlignedPairIndex {
  std::size_t query_index = 0;
  std::size_t target_index = 0;
};

struct AlignedPairExtractionRequest {
  const hikoboshi::universal::AlignmentPath* path = nullptr;
  hikoboshi::universal::StructureView query_structure{};
  hikoboshi::universal::StructureView target_structure{};
};

struct AlignedPairExtractionOutput {
  hikoboshi::universal::Span<AlignedPairIndex> pair_indices{};
  hikoboshi::universal::Span<std::uint8_t> observed_pair_mask{};
  std::size_t aligned_pair_count = 0;
  std::size_t observed_pair_count = 0;
  bool truncated = false;
};

struct KabschTransform {
  double rotation[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  Point3 translation{};
};

struct KabschResult {
  bool valid = false;
  hikoboshi::universal::MetricInvalidReason reason =
      hikoboshi::universal::MetricInvalidReason::Unavailable;
  KabschTransform transform{};
  double rmsd = 0.0;
  std::size_t pair_count = 0;
};

hikoboshi::universal::MetricValue valid_metric(double value) noexcept;
hikoboshi::universal::MetricValue invalid_metric(
    hikoboshi::universal::MetricInvalidReason reason) noexcept;

CoverageMetrics compute_coverage(const hikoboshi::universal::AlignmentPath& path,
                                 std::size_t query_length,
                                 std::size_t target_length) noexcept;

hikoboshi::universal::MetricValue compute_identity(
    const hikoboshi::universal::AlignmentPath& path,
    hikoboshi::universal::Span<const char> query_codes,
    hikoboshi::universal::Span<const char> target_codes) noexcept;

KabschResult kabsch_superpose(
    hikoboshi::universal::Span<const Point3> query_points,
    hikoboshi::universal::Span<const Point3> target_points) noexcept;

bool has_complete_structure_coordinates(
    const hikoboshi::universal::StructureView& structure) noexcept;

bool extract_aligned_pair_index(
    const hikoboshi::universal::AlignmentStep& step,
    AlignedPairIndex& pair) noexcept;

void extract_aligned_pairs(const AlignedPairExtractionRequest& request,
                           AlignedPairExtractionOutput& output) noexcept;

bool load_aligned_observed_ca_pair(
    AlignedPairIndex index,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target,
    AlignedCaPair& pair) noexcept;

bool load_aligned_observed_ca_pair(
    const hikoboshi::universal::AlignmentStep& step,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target,
    AlignedCaPair& pair) noexcept;

KabschResult kabsch_superpose_aligned_ca(
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target) noexcept;

Point3 apply_transform(const KabschTransform& transform, Point3 point) noexcept;

hikoboshi::universal::MetricValue compute_rmsd(
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target) noexcept;

TmScoreMetrics compute_tm_scores(
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target,
    std::size_t query_length,
    std::size_t target_length) noexcept;

LddtMetrics compute_lddt(
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target,
    double r0 = 15.0) noexcept;

MetricBlock compute_metric_block(
    const hikoboshi::universal::AlignmentPath& path,
    double raw_sw_score,
    const hikoboshi::universal::EmbeddingView& query_embedding,
    const hikoboshi::universal::EmbeddingView& target_embedding,
    const hikoboshi::universal::StructureView& query_structure,
    const hikoboshi::universal::StructureView& target_structure) noexcept;

}  // namespace hikoboshi::algorithms

#endif  // HIKOBOSHI_ALGORITHMS_METRICS_HPP

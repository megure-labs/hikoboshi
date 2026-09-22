#include <hikoboshi/algorithms/metrics.hpp>

#include <cmath>
#include <cstddef>

namespace hikoboshi::algorithms {
namespace {

using hikoboshi::universal::MetricInvalidReason;
using hikoboshi::universal::StructureView;

double distance(Point3 a, Point3 b) noexcept {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double d0(std::size_t length) noexcept {
  const double adjusted = static_cast<double>(length > 15 ? length - 15 : 1);
  const double value = 1.24 * std::cbrt(adjusted) - 1.8;
  return value > 0.5 ? value : 0.5;
}

TmScoreMetrics tm_from_superposition(
    const hikoboshi::universal::AlignmentPath& path,
    const StructureView& query,
    const StructureView& target,
    const KabschResult& kabsch,
    std::size_t query_length,
    std::size_t target_length) noexcept {
  if (!kabsch.valid) {
    const auto invalid = invalid_metric(kabsch.reason);
    return {invalid, invalid};
  }
  const double query_scale = d0(query_length);
  const double target_scale = d0(target_length);
  double query_sum = 0.0;
  double target_sum = 0.0;
  AlignedCaPair pair{};
  for (const auto& step : path.steps) {
    if (!load_aligned_observed_ca_pair(step, query, target, pair)) {
      continue;
    }
    const Point3 transformed = apply_transform(kabsch.transform, pair.target);
    const double d = distance(pair.query, transformed);
    if (query_length != 0) {
      const double ratio = d / query_scale;
      query_sum += 1.0 / (1.0 + ratio * ratio);
    }
    if (target_length != 0) {
      const double ratio = d / target_scale;
      target_sum += 1.0 / (1.0 + ratio * ratio);
    }
  }
  // Each sum retains its original per-step arithmetic and accumulation order.
  return {
      query_length == 0 ? invalid_metric(MetricInvalidReason::ZeroDenominator)
                        : valid_metric(query_sum / static_cast<double>(query_length)),
      target_length == 0 ? invalid_metric(MetricInvalidReason::ZeroDenominator)
                         : valid_metric(target_sum / static_cast<double>(target_length))};
}

}  // namespace

TmScoreMetrics compute_tm_scores(
    const hikoboshi::universal::AlignmentPath& path,
    const StructureView& query,
    const StructureView& target,
    std::size_t query_length,
    std::size_t target_length) noexcept {
  const KabschResult kabsch = kabsch_superpose_aligned_ca(path, query, target);
  return tm_from_superposition(path, query, target, kabsch,
                               query_length, target_length);
}

SuperpositionMetrics compute_superposition_metrics(
    const hikoboshi::universal::AlignmentPath& path,
    const StructureView& query,
    const StructureView& target,
    std::size_t query_length,
    std::size_t target_length) noexcept {
  const KabschResult kabsch = kabsch_superpose_aligned_ca(path, query, target);
  return {kabsch.valid ? valid_metric(kabsch.rmsd) : invalid_metric(kabsch.reason),
          tm_from_superposition(path, query, target, kabsch,
                               query_length, target_length)};
}

}  // namespace hikoboshi::algorithms

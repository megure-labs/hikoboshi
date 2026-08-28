#include <hikoboshi/algorithms/metrics.hpp>

#include <cmath>
#include <cstddef>

namespace hikoboshi::algorithms {
namespace {

using hikoboshi::universal::MetricInvalidReason;
using hikoboshi::universal::MetricValue;
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

MetricValue tm_for_length(const hikoboshi::universal::AlignmentPath& path,
                          const StructureView& query,
                          const StructureView& target,
                          const KabschTransform& transform,
                          std::size_t norm_length) noexcept {
  if (norm_length == 0) {
    return invalid_metric(MetricInvalidReason::ZeroDenominator);
  }
  const double scale = d0(norm_length);
  double sum = 0.0;
  AlignedCaPair pair{};
  for (const auto& step : path.steps) {
    if (!load_aligned_observed_ca_pair(step, query, target, pair)) {
      continue;
    }
    const Point3 transformed = apply_transform(transform, pair.target);
    const double d = distance(pair.query, transformed);
    const double ratio = d / scale;
    sum += 1.0 / (1.0 + ratio * ratio);
  }
  return valid_metric(sum / static_cast<double>(norm_length));
}

}  // namespace

TmScoreMetrics compute_tm_scores(
    const hikoboshi::universal::AlignmentPath& path,
    const StructureView& query,
    const StructureView& target,
    std::size_t query_length,
    std::size_t target_length) noexcept {
  TmScoreMetrics result{
      invalid_metric(MetricInvalidReason::MissingStructureMetadata),
      invalid_metric(MetricInvalidReason::MissingStructureMetadata),
  };
  if (!has_complete_structure_coordinates(query) ||
      !has_complete_structure_coordinates(target)) {
    return result;
  }

  const KabschResult kabsch = kabsch_superpose_aligned_ca(path, query, target);
  if (!kabsch.valid) {
    result.query_norm = invalid_metric(kabsch.reason);
    result.target_norm = invalid_metric(kabsch.reason);
    return result;
  }

  result.query_norm =
      tm_for_length(path, query, target, kabsch.transform, query_length);
  result.target_norm =
      tm_for_length(path, query, target, kabsch.transform, target_length);
  return result;
}

}  // namespace hikoboshi::algorithms

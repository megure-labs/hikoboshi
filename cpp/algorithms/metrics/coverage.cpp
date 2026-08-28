#include <hikoboshi/algorithms/metrics.hpp>

#include <cstddef>

namespace hikoboshi::algorithms {

hikoboshi::universal::MetricValue valid_metric(double value) noexcept {
  return {value, true, hikoboshi::universal::MetricInvalidReason::None};
}

hikoboshi::universal::MetricValue invalid_metric(
    hikoboshi::universal::MetricInvalidReason reason) noexcept {
  return {0.0, false, reason};
}

CoverageMetrics compute_coverage(const hikoboshi::universal::AlignmentPath& path,
                                 std::size_t query_length,
                                 std::size_t target_length) noexcept {
  CoverageMetrics metrics{
      invalid_metric(hikoboshi::universal::MetricInvalidReason::ZeroDenominator),
      invalid_metric(hikoboshi::universal::MetricInvalidReason::ZeroDenominator),
      invalid_metric(hikoboshi::universal::MetricInvalidReason::ZeroDenominator),
  };

  const double aligned_pairs = static_cast<double>(path.aligned_pairs);
  if (query_length != 0) {
    metrics.query =
        valid_metric(aligned_pairs / static_cast<double>(query_length));
  }
  if (target_length != 0) {
    metrics.target =
        valid_metric(aligned_pairs / static_cast<double>(target_length));
  }
  if (query_length + target_length != 0) {
    metrics.mean =
        valid_metric((2.0 * aligned_pairs) /
                     static_cast<double>(query_length + target_length));
  }
  return metrics;
}

}  // namespace hikoboshi::algorithms

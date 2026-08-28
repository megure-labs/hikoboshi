#include <hikoboshi/errors/format.hpp>

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace hikoboshi::errors {

std::string format_metric_invalid_reason(
    universal::MetricInvalidReason reason) {
  switch (reason) {
    case universal::MetricInvalidReason::None:
      return "none";
    case universal::MetricInvalidReason::Unavailable:
      return "unavailable";
    case universal::MetricInvalidReason::MissingSequenceMetadata:
      return "missing_sequence_metadata";
    case universal::MetricInvalidReason::MissingStructureMetadata:
      return "missing_structure_metadata";
    case universal::MetricInvalidReason::InsufficientAlignedPairs:
      return "insufficient_aligned_pairs";
    case universal::MetricInvalidReason::ZeroDenominator:
      return "zero_denominator";
    case universal::MetricInvalidReason::Unimplemented:
      return "unimplemented";
  }
  return "unknown_metric_invalid_reason";
}

std::string format_metric(universal::MetricValue metric,
                          int significant_digits) {
  if (!metric.valid) {
    return kMetricNotAvailable;
  }

  if (significant_digits < 1) {
    significant_digits = 1;
  }
  if (significant_digits > std::numeric_limits<double>::max_digits10) {
    significant_digits = std::numeric_limits<double>::max_digits10;
  }

  std::ostringstream out;
  out << std::setprecision(significant_digits) << metric.value;
  return out.str();
}

}  // namespace hikoboshi::errors

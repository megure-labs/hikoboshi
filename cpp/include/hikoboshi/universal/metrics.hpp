#ifndef HIKOBOSHI_UNIVERSAL_METRICS_HPP
#define HIKOBOSHI_UNIVERSAL_METRICS_HPP

/// @file
/// Public metric validity and numeric value records.

#include <cstdint>

namespace hikoboshi::universal {

/// Machine-readable reason an optional metric is not available.
///
/// Invalid metrics are represented explicitly instead of using `0.0` as a
/// sentinel. Adapters render invalid values as `NA`, `None`, or another
/// language-appropriate null value.
enum class MetricInvalidReason : std::uint8_t {
  None = 0,
  Unavailable = 1,
  MissingSequenceMetadata = 2,
  MissingStructureMetadata = 3,
  InsufficientAlignedPairs = 4,
  ZeroDenominator = 5,
  Unimplemented = 6,
};

/// Numeric metric value plus validity metadata.
///
/// When `valid` is false, `value` has no semantic meaning and `reason`
/// explains why the metric could not be computed.
struct MetricValue {
  double value;
  bool valid;
  MetricInvalidReason reason;
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_METRICS_HPP

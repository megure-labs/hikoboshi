#ifndef HIKOBOSHI_ERRORS_FORMAT_HPP
#define HIKOBOSHI_ERRORS_FORMAT_HPP

#include <string>

#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::errors {

inline constexpr const char* kMetricNotAvailable = "NA";

std::string format_status_code(universal::StatusCode code);

std::string format_status(universal::Status status);

std::string format_metric_invalid_reason(universal::MetricInvalidReason reason);

std::string format_metric(universal::MetricValue metric,
                          int significant_digits = 6);

}  // namespace hikoboshi::errors

#endif  // HIKOBOSHI_ERRORS_FORMAT_HPP

#include <hikoboshi/errors/format.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/status.hpp>

#include <cstdlib>
#include <cstdio>
#include <string>

namespace pe = hikoboshi::errors;
namespace pu = hikoboshi::universal;

namespace {

void fail(const char* tag) {
  std::fprintf(stderr, "errors_formatter_tests: %s\n", tag);
  std::exit(1);
}

void expect_eq(const std::string& actual, const char* expected,
               const char* tag) {
  if (actual != expected) {
    std::fprintf(stderr,
                 "errors_formatter_tests: %s: expected '%s', got '%s'\n",
                 tag,
                 expected,
                 actual.c_str());
    std::exit(1);
  }
}

void test_status_code_formatting() {
  expect_eq(pe::format_status_code(pu::StatusCode::Ok), "ok", "ok code");
  expect_eq(pe::format_status_code(pu::StatusCode::InvalidArgument),
            "invalid_argument",
            "invalid argument code");
  expect_eq(pe::format_status_code(pu::StatusCode::FailedPrecondition),
            "failed_precondition",
            "failed precondition code");
  expect_eq(pe::format_status_code(pu::StatusCode::Unavailable),
            "unavailable",
            "unavailable code");
  expect_eq(pe::format_status_code(pu::StatusCode::Unimplemented),
            "unimplemented",
            "unimplemented code");
  expect_eq(pe::format_status_code(pu::StatusCode::InternalError),
            "internal_error",
            "internal error code");
  expect_eq(pe::format_status({pu::StatusCode::InvalidArgument,
                               "gap_open must be negative"}),
            "invalid_argument: gap_open must be negative",
            "status detail");
  expect_eq(pe::format_status({pu::StatusCode::Ok, nullptr}),
            "ok",
            "status without detail");
}

void test_invalid_metric_reasons() {
  expect_eq(pe::format_metric_invalid_reason(pu::MetricInvalidReason::None),
            "none",
            "reason none");
  expect_eq(pe::format_metric_invalid_reason(
                pu::MetricInvalidReason::Unavailable),
            "unavailable",
            "reason unavailable");
  expect_eq(pe::format_metric_invalid_reason(
                pu::MetricInvalidReason::MissingSequenceMetadata),
            "missing_sequence_metadata",
            "reason missing sequence");
  expect_eq(pe::format_metric_invalid_reason(
                pu::MetricInvalidReason::MissingStructureMetadata),
            "missing_structure_metadata",
            "reason missing structure");
  expect_eq(pe::format_metric_invalid_reason(
                pu::MetricInvalidReason::InsufficientAlignedPairs),
            "insufficient_aligned_pairs",
            "reason insufficient pairs");
  expect_eq(pe::format_metric_invalid_reason(
                pu::MetricInvalidReason::ZeroDenominator),
            "zero_denominator",
            "reason zero denominator");
  expect_eq(pe::format_metric_invalid_reason(
                pu::MetricInvalidReason::Unimplemented),
            "unimplemented",
            "reason unimplemented");
}

void test_metric_formatting() {
  const pu::MetricValue valid{0.123456789, true, pu::MetricInvalidReason::None};
  expect_eq(pe::format_metric(valid, 6), "0.123457", "valid metric precision");
  expect_eq(pe::format_metric({42.0, true, pu::MetricInvalidReason::None}, 0),
            "4e+01",
            "minimum precision clamp");

  const pu::MetricValue invalid{0.0,
                                false,
                                pu::MetricInvalidReason::InsufficientAlignedPairs};
  expect_eq(pe::format_metric(invalid), pe::kMetricNotAvailable, "invalid NA");

  const pu::MetricValue unavailable{
      19.0, false, pu::MetricInvalidReason::Unavailable};
  expect_eq(pe::format_metric(unavailable), "NA", "unavailable NA");
}

}  // namespace

int main() {
  if (std::string{pe::kMetricNotAvailable} != "NA") {
    fail("chartered NA literal changed");
  }
  test_status_code_formatting();
  test_invalid_metric_reasons();
  test_metric_formatting();
  return 0;
}

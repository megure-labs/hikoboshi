// Row-wise log_softmax parity goldens.
//
// The stored simple-row values match PyTorch
// (`torch.nn.functional.log_softmax(x, dim=-1)` on `torch.float32`
// inputs). Broader rows are checked against an in-test shifted
// log-sum-exp reference with the same last-dimension semantics.

#include <hikoboshi/dispatch/dispatch_table.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>
#include <hikoboshi/primitives/compute/log_softmax.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hiko_d = hikoboshi::dispatch;
namespace hiko_dr = hikoboshi::dispatch::registry;
namespace hiko_p = hikoboshi::primitives::compute;

namespace {

// Mirrors bench/numerical_tolerance.json:log_softmax_abs.
constexpr float kLogSoftmaxTolerance = 1.0e-6F;

void fail(const char* tag) {
  std::fprintf(stderr, "log_softmax_parity_test: %s\n", tag);
  std::exit(1);
}

bool within(float actual, float reference) {
  return std::fabs(actual - reference) <= kLogSoftmaxTolerance;
}

void check_row_against_reference(const float* actual, const float* reference,
                                 std::size_t dim, const char* tag) {
  for (std::size_t d = 0; d < dim; ++d) {
    if (!within(actual[d], reference[d])) {
      fail(tag);
    }
  }
}

void reference_log_softmax(const float* input, std::size_t row_count,
                           std::size_t dim, float* reference) {
  for (std::size_t r = 0; r < row_count; ++r) {
    const float* row = input + r * dim;
    float* out = reference + r * dim;

    double row_max = -INFINITY;
    for (std::size_t d = 0; d < dim; ++d) {
      const double value = static_cast<double>(row[d]);
      if (value > row_max) {
        row_max = value;
      }
    }

    double row_sum = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
      row_sum += std::exp(static_cast<double>(row[d]) - row_max);
    }

    const double log_row_sum = std::log(row_sum);
    for (std::size_t d = 0; d < dim; ++d) {
      out[d] =
          static_cast<float>((static_cast<double>(row[d]) - row_max) -
                             log_row_sum);
    }
  }
}

void run_scalar(const hiko_p::LogSoftmaxScalarRequest& request, float* output) {
  hiko_p::LogSoftmaxScalarOutput out{};
  out.output = output;
  hiko_p::log_softmax_scalar(request, out);
}

void run_through_dispatch(const hiko_p::LogSoftmaxScalarRequest& request,
                          float* output) {
  const hiko_d::DispatchTable& table = hiko_d::scalar_dispatch_table();
  if (table.log_softmax == nullptr) {
    fail("dispatch table log_softmax slot must be populated");
  }

  hiko_p::LogSoftmaxScalarOutput out{};
  out.output = output;
  table.log_softmax(request, out);
}

void test_simple_three_element_row() {
  // PyTorch: F.log_softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
  // -> [-2.4076059, -1.4076059, -0.40760595].
  const float input[3] = {1.0F, 2.0F, 3.0F};
  const float reference[3] = {-2.4076059F, -1.4076059F, -0.40760595F};
  float output[3] = {};

  hiko_p::LogSoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 3,
                              "simple PyTorch golden mismatch");
}

void test_multi_row_reference() {
  const float input[4 * 5] = {
      0.1F,   -0.2F,  0.3F,  -0.4F,  0.5F,
      1.0F,   2.0F,   3.0F,  4.0F,   5.0F,
      -5.0F,  -4.0F,  -3.0F, -2.0F,  -1.0F,
      0.0F,   0.0F,   0.0F,  0.0F,   0.0F,
  };
  float output[4 * 5] = {};
  float reference[4 * 5] = {};
  reference_log_softmax(input, 4, 5, reference);

  hiko_p::LogSoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 4;
  request.row_dimension = 5;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 4 * 5,
                              "multi-row log_softmax mismatch");
}

void test_large_magnitude_rows_are_stable() {
  const float input[2 * 3] = {
      10000.0F, 10001.0F, 9999.0F,
      -10000.0F, -9999.0F, -10001.0F,
  };
  const float shifted_reference[3] = {
      -1.4076059F,
      -0.40760595F,
      -2.4076059F,
  };
  float output[2 * 3] = {};

  hiko_p::LogSoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 2;
  request.row_dimension = 3;
  run_scalar(request, output);

  check_row_against_reference(output, shifted_reference, 3,
                              "large positive row must be shift-invariant");
  check_row_against_reference(output + 3, shifted_reference, 3,
                              "large negative row must be shift-invariant");

  for (float value : output) {
    if (!std::isfinite(value)) {
      fail("large-magnitude row must produce finite log probabilities");
    }
  }
}

void test_empty_dimensions_are_no_op() {
  const float input_storage = 7.0F;
  float output_storage = 99.0F;

  hiko_p::LogSoftmaxScalarRequest request{};
  request.input = &input_storage;
  request.row_count = 0;
  request.row_dimension = 3;
  run_scalar(request, &output_storage);
  if (output_storage != 99.0F) {
    fail("row_count == 0 must not write to output");
  }

  request.row_count = 1;
  request.row_dimension = 0;
  run_scalar(request, &output_storage);
  if (output_storage != 99.0F) {
    fail("row_dimension == 0 must not write to output");
  }
}

void test_dispatch_table_routes_to_scalar() {
  const float input[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 1.0F};
  float scalar_output[6] = {};
  float dispatch_output[6] = {};

  hiko_p::LogSoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 2;
  request.row_dimension = 3;
  run_scalar(request, scalar_output);
  run_through_dispatch(request, dispatch_output);
  check_row_against_reference(dispatch_output, scalar_output, 6,
                              "dispatch-table log_softmax slot mismatch");
}

void test_registry_records_log_softmax_capabilities() {
  const hiko_dr::RegisteredPrimitiveOpRecord* record =
      hiko_dr::find_primitive_op("hikoboshi.log_softmax.row_wise.v1");
  if (record == nullptr) {
    fail(
        "primitive_op_registry must include hikoboshi.log_softmax.row_wise.v1");
  }
  if (record->family != hiko_dr::PrimitiveOpFamily::Compute) {
    fail("log_softmax record must belong to the compute family");
  }
  if (record->capabilities.default_parity_mode != hiko_dr::ParityMode::Strict) {
    fail("log_softmax default parity mode must be strict");
  }
  if (record->capabilities.supported_parity_modes.size != 1 ||
      record->capabilities.supported_parity_modes.data[0] !=
          hiko_dr::ParityMode::Strict) {
    fail("log_softmax must support only strict parity mode");
  }
  if (record->dispatch_entry == nullptr) {
    fail("log_softmax record must register a non-null dispatch entry");
  }
}

}  // namespace

int main() {
  test_simple_three_element_row();
  test_multi_row_reference();
  test_large_magnitude_rows_are_stable();
  test_empty_dimensions_are_no_op();
  test_dispatch_table_routes_to_scalar();
  test_registry_records_log_softmax_capabilities();
  return 0;
}

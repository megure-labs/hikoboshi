// Row-wise softmax parity goldens.
//
// Reference values were computed in PyTorch
// (`torch.nn.functional.softmax(x, dim=-1)` on `torch.float32` inputs).
// Strict mode requires sub-1e-5 max-abs vs the PyTorch reference; fast
// mode requires within 1e-4 max-abs vs the same reference. Both modes
// share the current scalar kernel, so each row of goldens is checked
// against both tolerances.

#include <hikoboshi/dispatch/dispatch_table.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>
#include <hikoboshi/primitives/compute/softmax.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace hiko_p = hikoboshi::primitives::compute;
namespace hiko_d = hikoboshi::dispatch;
namespace hiko_dr = hikoboshi::dispatch::registry;

namespace {

constexpr float kStrictTolerance = 1e-5F;
constexpr float kFastTolerance = 1e-4F;

void fail(const char* tag) {
  std::fprintf(stderr, "softmax_parity_tests: %s\n", tag);
  std::exit(1);
}

bool within(float actual, float reference, float tolerance) {
  return std::fabs(actual - reference) <= tolerance;
}

void check_row_against_reference(const float* actual, const float* reference,
                                 std::size_t dim, const char* tag) {
  for (std::size_t d = 0; d < dim; ++d) {
    if (!within(actual[d], reference[d], kStrictTolerance)) {
      fail(tag);
    }
    if (!within(actual[d], reference[d], kFastTolerance)) {
      fail(tag);
    }
  }
}

void run_scalar(const hiko_p::SoftmaxScalarRequest& request, float* output) {
  hiko_p::SoftmaxScalarOutput out{};
  out.output = output;
  hiko_p::softmax_scalar(request, out);
}

void run_through_dispatch(const hiko_p::SoftmaxScalarRequest& request,
                          float* output) {
  const hiko_d::DispatchTable& table = hiko_d::scalar_dispatch_table();
  hiko_p::SoftmaxScalarOutput out{};
  out.output = output;
  if (table.softmax == nullptr) {
    fail("dispatch table softmax slot must be populated");
  }
  table.softmax(request, out);
}

void test_simple_three_element_row() {
  // PyTorch: F.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
  // -> [0.09003057, 0.24472847, 0.66524096]
  const float input[3] = {1.0F, 2.0F, 3.0F};
  const float reference[3] = {0.09003057F, 0.24472847F, 0.66524096F};
  float output[3] = {0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = nullptr;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 3,
                              "simple three-element row mismatch");
}

void test_all_zero_input_is_uniform() {
  const float input[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  const float quarter = 0.25F;
  const float reference[4] = {quarter, quarter, quarter, quarter};
  float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 4;
  request.temperature = 1.0F;
  request.mask = nullptr;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 4,
                              "all-zero row must produce a uniform 1/dim row");
}

void test_large_positive_input_shift_invariance() {
  // Shift-invariance: softmax([1,2,3]) == softmax([101,102,103]).
  const float input[3] = {101.0F, 102.0F, 103.0F};
  const float reference[3] = {0.09003057F, 0.24472847F, 0.66524096F};
  float output[3] = {0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = nullptr;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 3,
                              "large positive input must not overflow");

  for (std::size_t d = 0; d < 3; ++d) {
    if (!std::isfinite(output[d])) {
      fail("large positive input must not produce non-finite output");
    }
  }
}

void test_large_negative_input_no_underflow() {
  const float input[3] = {-101.0F, -100.0F, -99.0F};
  const float reference[3] = {0.09003057F, 0.24472847F, 0.66524096F};
  float output[3] = {0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = nullptr;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 3,
                              "large negative input must not underflow");

  float row_sum = 0.0F;
  for (std::size_t d = 0; d < 3; ++d) {
    row_sum += output[d];
  }
  if (!within(row_sum, 1.0F, kFastTolerance)) {
    fail("large negative input row must still sum to 1");
  }
}

void test_mask_excludes_position() {
  // PyTorch: F.softmax(torch.tensor([1.0, 2.0, 3.0]) + mask, dim=0)
  // with mask = [0, -inf, 0] -> [0.11920291, 0.0, 0.88079709].
  const float input[3] = {1.0F, 2.0F, 3.0F};
  const float mask[3] = {0.0F, hiko_p::kSoftmaxNegInf, 0.0F};
  const float reference[3] = {0.11920291F, 0.0F, 0.88079709F};
  float output[3] = {0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = mask;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 3,
                              "masked-position softmax mismatch");

  if (output[1] != 0.0F) {
    fail("a fully masked position must produce exactly zero");
  }
}

void test_fully_masked_row_emits_zeros() {
  // Contract: rows whose entries are all at or below kSoftmaxNegInf
  // emit all zeros rather than NaN.
  const float input[3] = {1.0F, 2.0F, 3.0F};
  const float mask[3] = {hiko_p::kSoftmaxNegInf, hiko_p::kSoftmaxNegInf,
                         hiko_p::kSoftmaxNegInf};
  float output[3] = {99.0F, 99.0F, 99.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = mask;
  run_scalar(request, output);
  for (std::size_t d = 0; d < 3; ++d) {
    if (output[d] != 0.0F) {
      fail("a fully masked row must emit all-zero output");
    }
  }
}

void test_temperature_scales_logits() {
  // PyTorch: F.softmax(torch.tensor([1.0, 2.0, 3.0]) / 2.0, dim=0)
  // -> [0.18632373, 0.30719589, 0.50648039]
  const float input[3] = {1.0F, 2.0F, 3.0F};
  const float reference[3] = {0.18632373F, 0.30719589F, 0.50648039F};
  float output[3] = {0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  request.temperature = 2.0F;
  request.mask = nullptr;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 3,
                              "temperature=2 softmax mismatch");
}

void test_multi_row_independence() {
  // Two independent rows, each [1, 2, 3]; rows must not influence each
  // other (the second row in the buffer is a different input row).
  const float input[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 1.0F};
  // Row 0: softmax([1, 2, 3])
  // Row 1: softmax([-1, 0, 1]) == softmax([1, 2, 3]) by shift invariance.
  const float reference[6] = {
      0.09003057F, 0.24472847F, 0.66524096F,
      0.09003057F, 0.24472847F, 0.66524096F,
  };
  float output[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 2;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = nullptr;
  run_scalar(request, output);
  check_row_against_reference(output, reference, 6,
                              "multi-row softmax mismatch");
}

void test_rows_sum_to_one() {
  // 4 rows x 5 columns of mixed-sign inputs; each row should sum to 1.
  const float input[4 * 5] = {
      0.1F, -0.2F, 0.3F, -0.4F, 0.5F,
      1.0F, 2.0F,  3.0F, 4.0F,  5.0F,
      -5.0F, -4.0F, -3.0F, -2.0F, -1.0F,
      0.0F, 0.0F,  0.0F, 0.0F,  0.0F,
  };
  float output[4 * 5] = {};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 4;
  request.row_dimension = 5;
  request.temperature = 1.0F;
  request.mask = nullptr;
  run_scalar(request, output);

  for (std::size_t r = 0; r < 4; ++r) {
    float row_sum = 0.0F;
    for (std::size_t d = 0; d < 5; ++d) {
      row_sum += output[r * 5 + d];
      if (!(output[r * 5 + d] >= 0.0F) || !(output[r * 5 + d] <= 1.0F)) {
        fail("softmax entries must lie in [0, 1]");
      }
    }
    if (!within(row_sum, 1.0F, kFastTolerance)) {
      fail("each softmax row must sum to 1 within fast tolerance");
    }
  }
}

void test_empty_dimensions_are_no_op() {
  // Calls with row_count == 0 or row_dimension == 0 must not touch the
  // output buffer.
  const float input_storage = 7.0F;
  float output_storage = 99.0F;
  hiko_p::SoftmaxScalarRequest request{};
  request.input = &input_storage;
  request.row_count = 0;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = nullptr;
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
  // The dispatch-table softmax slot must produce the same result as the
  // direct scalar call.
  const float input[3] = {1.0F, 2.0F, 3.0F};
  const float reference[3] = {0.09003057F, 0.24472847F, 0.66524096F};
  float output[3] = {0.0F, 0.0F, 0.0F};

  hiko_p::SoftmaxScalarRequest request{};
  request.input = input;
  request.row_count = 1;
  request.row_dimension = 3;
  request.temperature = 1.0F;
  request.mask = nullptr;
  run_through_dispatch(request, output);
  check_row_against_reference(output, reference, 3,
                              "dispatch-table softmax slot mismatch");
}

void test_registry_records_softmax_capabilities() {
  // The packet's capability declaration: cpu.scalar, supports Strict
  // and Fast, defaults to Fast.
  const hiko_dr::RegisteredPrimitiveOpRecord* record =
      hiko_dr::find_primitive_op("hikoboshi.softmax.row_wise.v1");
  if (record == nullptr) {
    fail("primitive_op_registry must include hikoboshi.softmax.row_wise.v1");
  }
  if (record->family != hiko_dr::PrimitiveOpFamily::Compute) {
    fail("softmax record must belong to the compute family");
  }
  if (record->capabilities.default_parity_mode != hiko_dr::ParityMode::Fast) {
    fail("softmax default parity mode must be fast");
  }
  bool has_strict = false;
  bool has_fast = false;
  for (std::size_t i = 0; i < record->capabilities.supported_parity_modes.size;
       ++i) {
    const hiko_dr::ParityMode mode =
        record->capabilities.supported_parity_modes.data[i];
    if (mode == hiko_dr::ParityMode::Strict) {
      has_strict = true;
    }
    if (mode == hiko_dr::ParityMode::Fast) {
      has_fast = true;
    }
  }
  if (!has_strict || !has_fast) {
    fail("softmax must support both strict and fast parity modes");
  }
  if (record->dispatch_entry == nullptr) {
    fail("softmax record must register a non-null dispatch entry");
  }
}

}  // namespace

int main() {
  test_simple_three_element_row();
  test_all_zero_input_is_uniform();
  test_large_positive_input_shift_invariance();
  test_large_negative_input_no_underflow();
  test_mask_excludes_position();
  test_fully_masked_row_emits_zeros();
  test_temperature_scales_logits();
  test_multi_row_independence();
  test_rows_sum_to_one();
  test_empty_dimensions_are_no_op();
  test_dispatch_table_routes_to_scalar();
  test_registry_records_softmax_capabilities();
  return 0;
}

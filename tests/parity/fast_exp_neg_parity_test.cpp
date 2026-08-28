#include <hikoboshi/primitives/detail/fast_exp_neg.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace hiko_pd = hikoboshi::primitives::detail;

namespace {

constexpr int kSampleCount = 10000;
constexpr float kMinInput = -20.0F;
constexpr float kMaxInput = 1.0F;
constexpr float kTolerance = 5.0e-5F;

void fail(const char* detail) {
  std::fprintf(stderr, "fast_exp_neg_parity_test: %s\n", detail);
  std::exit(1);
}

void test_fast_exp_neg() {
  float max_contract_diff = 0.0F;
  float max_nonpositive_exp_diff = 0.0F;
  float max_positive_clamp_diff = 0.0F;

  for (int index = 0; index < kSampleCount; ++index) {
    const float t =
        static_cast<float>(index) / static_cast<float>(kSampleCount - 1);
    const float x = kMinInput + (kMaxInput - kMinInput) * t;
    const float actual = hiko_pd::fast_exp_neg(x);
    const float expected_contract = x > 0.0F ? 1.0F : std::exp(x);
    const float contract_diff = std::fabs(actual - expected_contract);
    if (contract_diff > max_contract_diff) {
      max_contract_diff = contract_diff;
    }
    if (x <= 0.0F) {
      const float exp_diff = std::fabs(actual - std::exp(x));
      if (exp_diff > max_nonpositive_exp_diff) {
        max_nonpositive_exp_diff = exp_diff;
      }
    } else {
      const float clamp_diff = std::fabs(actual - 1.0F);
      if (clamp_diff > max_positive_clamp_diff) {
        max_positive_clamp_diff = clamp_diff;
      }
    }
  }

  std::printf("fast_exp_neg_parity_test: samples=%d range=[%.1f,%.1f] "
              "max_contract_diff=%.9g max_nonpositive_exp_diff=%.9g "
              "max_positive_clamp_diff=%.9g\n",
              kSampleCount, kMinInput, kMaxInput, max_contract_diff,
              max_nonpositive_exp_diff, max_positive_clamp_diff);

  if (max_contract_diff > kTolerance) {
    fail("fast_exp_neg contract drift exceeded tolerance");
  }
  if (max_nonpositive_exp_diff > kTolerance) {
    fail("fast_exp_neg non-positive exp drift exceeded tolerance");
  }
  if (max_positive_clamp_diff != 0.0F) {
    fail("fast_exp_neg positive clamp changed");
  }
}

void test_fast_exp_neg_sq() {
  float max_abs_diff = 0.0F;
  for (int index = 0; index < kSampleCount; ++index) {
    const float t =
        static_cast<float>(index) / static_cast<float>(kSampleCount - 1);
    const float x = -5.0F + 10.0F * t;
    const float actual = hiko_pd::fast_exp_neg_sq(x);
    const float expected = std::exp(-(x * x));
    const float diff = std::fabs(actual - expected);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
    }
  }
  std::printf("fast_exp_neg_parity_test: sq_samples=%d max_sq_abs_diff=%.9g\n",
              kSampleCount, max_abs_diff);
  if (max_abs_diff > kTolerance) {
    fail("fast_exp_neg_sq drift exceeded tolerance");
  }
}

void test_fast_exp_neg_nonfinite() {
  const float nan_value = std::numeric_limits<float>::quiet_NaN();
  if (!std::isnan(hiko_pd::fast_exp_neg(nan_value))) {
    fail("fast_exp_neg did not preserve NaN");
  }
  if (hiko_pd::fast_exp_neg(-std::numeric_limits<float>::infinity()) != 0.0F) {
    fail("fast_exp_neg negative infinity clamp changed");
  }
  if (hiko_pd::fast_exp_neg(std::numeric_limits<float>::infinity()) != 1.0F) {
    fail("fast_exp_neg positive infinity clamp changed");
  }
}

}  // namespace

int main() {
  test_fast_exp_neg();
  test_fast_exp_neg_sq();
  test_fast_exp_neg_nonfinite();
  return 0;
}

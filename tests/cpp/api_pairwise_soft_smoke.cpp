#include <hikoboshi/api/engine.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;

namespace {

constexpr float kMpnn64SoftGapOpen = -3.21337F;
constexpr float kMpnn64SoftGapExtension = -0.111704F;

void fail(const char* message) {
  std::fprintf(stderr, "api_pairwise_soft_smoke: %s\n", message);
  std::exit(1);
}

bool nearly_equal(float lhs, float rhs) {
  return std::fabs(lhs - rhs) <= 1.0e-6F;
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values,
                                  const std::vector<char>& codes,
                                  std::size_t residue_count,
                                  std::size_t dimension) {
  return {residue_count,
          dimension,
          {values.data(), values.size()},
          {codes.data(), codes.size()},
          {nullptr, 0}};
}

void require_soft_metrics_match(const hiko::PairwiseResult& actual,
                                const hiko::PairwiseResult& expected,
                                const char* context) {
  if (!actual.metrics.soft_sw_score.valid ||
      !expected.metrics.soft_sw_score.valid ||
      !nearly_equal(actual.metrics.soft_sw_score.value,
                    expected.metrics.soft_sw_score.value)) {
    fail(context);
  }
}

void test_explicit_soft_mode_returns_finite_metrics() {
  // An explicitly-soft PairwiseEmbeddingRequest must take the soft branch
  // and produce a populated path with finite metrics on a small
  // diagonal-dominant fixture. Hard is the Hikoboshi 0.1.0 default, so this
  // test pins soft explicitly rather than relying on the default mode.
  const std::vector<float> values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C', 'D'};
  hiko::PairwiseEmbeddingRequest request{
      embedding_view(values, codes, 3, 3),
      embedding_view(values, codes, 3, 3)};
  request.mode = hiko::AlignmentMode::Soft;
  if (request.mode != hiko::AlignmentMode::Soft) {
    fail("explicitly-soft PairwiseEmbeddingRequest mode must be Soft");
  }

  const hiko::Engine engine;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("explicit soft-mode pairwise must return Ok");
  }
  if (result.value.path.aligned_pairs == 0) {
    fail("explicit soft-mode pairwise must populate the path");
  }
  if (!std::isfinite(result.value.metrics.raw_sw_score)) {
    fail("explicit soft-mode raw_sw_score must be finite");
  }
  if (!result.value.metrics.soft_sw_score.valid) {
    fail("explicit soft-mode soft_sw_score must be valid");
  }
  if (result.value.metrics.soft_sw_score.value !=
      result.value.metrics.raw_sw_score) {
    fail("soft-only mode must mirror soft_sw_score into raw_sw_score");
  }
}

void test_default_soft_mode_uses_mpnn_soft_gap_family() {
  const std::vector<float> query_values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<float> target_values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<char> query_codes = {'A', 'C', 'D'};
  const std::vector<char> target_codes = {'A', 'D'};
  hiko::PairwiseEmbeddingRequest default_request{
      embedding_view(query_values, query_codes, 3, 3),
      embedding_view(target_values, target_codes, 2, 3)};
  default_request.mode = hiko::AlignmentMode::Soft;

  hiko::PairwiseEmbeddingRequest explicit_request = default_request;
  explicit_request.alignment.gap_open = kMpnn64SoftGapOpen;
  explicit_request.alignment.gap_extension = kMpnn64SoftGapExtension;

  const hiko::Engine engine;
  const auto default_result = engine.pairwise(default_request);
  const auto explicit_result = engine.pairwise(explicit_request);
  if (default_result.status.code != hiko_u::StatusCode::Ok ||
      explicit_result.status.code != hiko_u::StatusCode::Ok) {
    fail("default and explicit MPNN soft pairwise requests must return Ok");
  }
  if (!nearly_equal(default_result.value.metrics.raw_sw_score,
                    explicit_result.value.metrics.raw_sw_score)) {
    fail("default MPNN soft raw score must use the calibrated soft gaps");
  }
  require_soft_metrics_match(default_result.value, explicit_result.value,
                             "default MPNN soft metrics must use the calibrated soft gaps");
}

void test_soft_mode_with_explicit_temperature() {
  const std::vector<float> values = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C'};
  hiko::PairwiseEmbeddingRequest request{embedding_view(values, codes, 2, 2),
                                        embedding_view(values, codes, 2, 2)};
  request.mode = hiko::AlignmentMode::Soft;
  request.temperature = 1.0F;

  const hiko::Engine engine;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("explicit soft-mode pairwise must return Ok");
  }
  if (!std::isfinite(result.value.metrics.raw_sw_score)) {
    fail("explicit soft-mode raw_sw_score must be finite");
  }
  if (result.value.path.aligned_pairs == 0) {
    fail("explicit soft-mode pairwise must populate the path on diagonal input");
  }
}

void test_soft_mode_rejects_non_positive_temperature() {
  const std::vector<float> values = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C'};
  hiko::PairwiseEmbeddingRequest request{embedding_view(values, codes, 2, 2),
                                        embedding_view(values, codes, 2, 2)};
  request.mode = hiko::AlignmentMode::Soft;
  request.temperature = 0.0F;

  const hiko::Engine engine;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("soft mode with temperature=0 must return InvalidArgument");
  }
}

void test_soft_mode_rejects_nan_temperature() {
  const std::vector<float> values = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C'};
  hiko::PairwiseEmbeddingRequest request{embedding_view(values, codes, 2, 2),
                                        embedding_view(values, codes, 2, 2)};
  request.mode = hiko::AlignmentMode::Soft;
  request.temperature = std::nanf("");

  const hiko::Engine engine;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("soft mode with NaN temperature must return InvalidArgument");
  }
}

void test_hard_mode_ignores_invalid_temperature() {
  const std::vector<float> values = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C'};
  hiko::PairwiseEmbeddingRequest request{embedding_view(values, codes, 2, 2),
                                        embedding_view(values, codes, 2, 2)};
  request.mode = hiko::AlignmentMode::Hard;
  request.temperature = 0.0F;  // ignored when mode == Hard

  const hiko::Engine engine;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("hard mode with temperature=0 must still return Ok (temperature ignored)");
  }
  if (result.value.path.aligned_pairs != 2) {
    fail("hard-mode aligned_pairs mismatch on diagonal input");
  }
  if (result.value.metrics.soft_sw_score.valid) {
    fail("hard mode must not populate soft_sw_score");
  }
}

void test_both_mode_returns_hard_primary_and_soft_score() {
  const std::vector<float> values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C', 'D'};
  hiko::PairwiseEmbeddingRequest hard_request{
      embedding_view(values, codes, 3, 3),
      embedding_view(values, codes, 3, 3)};
  hard_request.mode = hiko::AlignmentMode::Hard;

  hiko::PairwiseEmbeddingRequest soft_request = hard_request;
  soft_request.mode = hiko::AlignmentMode::Soft;
  soft_request.alignment.gap_open = kMpnn64SoftGapOpen;
  soft_request.alignment.gap_extension = kMpnn64SoftGapExtension;

  hiko::PairwiseEmbeddingRequest both_request = hard_request;
  both_request.mode = hiko::AlignmentMode::Both;
  both_request.temperature = 1.0F;

  const hiko::Engine engine;
  const auto hard_result = engine.pairwise(hard_request);
  const auto soft_result = engine.pairwise(soft_request);
  const auto both_result = engine.pairwise(both_request);
  if (hard_result.status.code != hiko_u::StatusCode::Ok ||
      soft_result.status.code != hiko_u::StatusCode::Ok ||
      both_result.status.code != hiko_u::StatusCode::Ok) {
    fail("hard, soft, and both pairwise requests must return Ok");
  }
  if (both_result.value.metrics.raw_sw_score !=
      hard_result.value.metrics.raw_sw_score) {
    fail("both mode raw_sw_score must keep the hard score");
  }
  if (both_result.value.path.aligned_pairs !=
      hard_result.value.path.aligned_pairs) {
    fail("both mode path must keep the hard path");
  }
  if (!both_result.value.metrics.soft_sw_score.valid ||
      !std::isfinite(both_result.value.metrics.soft_sw_score.value)) {
    fail("both mode must populate finite soft_sw_score");
  }
  if (std::fabs(both_result.value.metrics.soft_sw_score.value -
                soft_result.value.metrics.raw_sw_score) > 1e-6) {
    fail("both mode soft_sw_score must match explicit soft raw_sw_score");
  }
}

void test_both_mode_rejects_non_positive_temperature() {
  const std::vector<float> values = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C'};
  hiko::PairwiseEmbeddingRequest request{embedding_view(values, codes, 2, 2),
                                        embedding_view(values, codes, 2, 2)};
  request.mode = hiko::AlignmentMode::Both;
  request.temperature = 0.0F;

  const hiko::Engine engine;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("both mode with temperature=0 must return InvalidArgument");
  }
}

}  // namespace

int main() {
  test_explicit_soft_mode_returns_finite_metrics();
  test_default_soft_mode_uses_mpnn_soft_gap_family();
  test_soft_mode_with_explicit_temperature();
  test_soft_mode_rejects_non_positive_temperature();
  test_soft_mode_rejects_nan_temperature();
  test_hard_mode_ignores_invalid_temperature();
  test_both_mode_returns_hard_primary_and_soft_score();
  test_both_mode_rejects_non_positive_temperature();
  return 0;
}

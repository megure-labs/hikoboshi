#include <hikoboshi/modules/soft_smith_waterman.hpp>
#include <hikoboshi/universal/status.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko_m = hikoboshi::modules;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* tag) {
  std::fprintf(stderr, "soft_smith_waterman_module_goldens: %s\n", tag);
  std::exit(1);
}

#include "support/soft_sw_orihime_reference.inc"

struct ModuleSoftSwBuffers {
  std::vector<float> match_alpha;
  std::vector<float> insert_alpha;
  std::vector<float> delete_alpha;
  std::vector<float> match_beta;
  std::vector<float> insert_beta;
  std::vector<float> delete_beta;
  std::vector<float> posteriors;
  float log_partition = 0.0F;

  ModuleSoftSwBuffers(std::size_t lq, std::size_t lt) {
    const std::size_t cells = (lq + 1U) * (lt + 1U);
    match_alpha.assign(cells, 0.0F);
    insert_alpha.assign(cells, 0.0F);
    delete_alpha.assign(cells, 0.0F);
    match_beta.assign(cells, 0.0F);
    insert_beta.assign(cells, 0.0F);
    delete_beta.assign(cells, 0.0F);
    posteriors.assign(lq * lt, 0.0F);
  }

  hiko_m::SoftSmithWatermanRequest make_request(const float* scores,
                                             std::size_t lq,
                                             std::size_t lt,
                                             float gap_open,
                                             float gap_ext,
                                             float temperature) {
    hiko_m::SoftSmithWatermanRequest request{};
    request.scores = scores;
    request.query_length = lq;
    request.target_length = lt;
    request.gap_open = gap_open;
    request.gap_extension = gap_ext;
    request.temperature = temperature;
    request.match_workspace = match_alpha.data();
    request.insert_workspace = insert_alpha.data();
    request.delete_workspace = delete_alpha.data();
    request.workspace_cells = match_alpha.size();
    request.match_grad_workspace = match_beta.data();
    request.insert_grad_workspace = insert_beta.data();
    request.delete_grad_workspace = delete_beta.data();
    return request;
  }

  hiko_m::SoftSmithWatermanOutput make_output() noexcept {
    hiko_m::SoftSmithWatermanOutput output{};
    output.log_partition = &log_partition;
    output.posteriors = posteriors.data();
    return output;
  }
};

float max_abs_error(const float* a, const float* b, std::size_t count) {
  float worst = 0.0F;
  for (std::size_t idx = 0; idx < count; ++idx) {
    const float diff = std::fabs(a[idx] - b[idx]);
    if (diff > worst) {
      worst = diff;
    }
  }
  return worst;
}

void require_close(const char* tag,
                   double actual,
                   double expected,
                   double tolerance) {
  if (std::fabs(actual - expected) > tolerance) {
    std::fprintf(stderr,
                 "soft_smith_waterman_module_goldens: %s mismatch "
                 "actual=%.12g expected=%.12g tolerance=%.3g\n",
                 tag,
                 actual,
                 expected,
                 tolerance);
    fail(tag);
  }
}

void verify_orihime_case(std::size_t case_index,
                     std::size_t lq,
                     std::size_t lt,
                     float gap_open,
                     float gap_ext,
                     float temperature,
                     float reference_log_partition,
                     const float* reference_scores,
                     const float* reference_posteriors) {
  ModuleSoftSwBuffers buffers(lq, lt);
  const auto request = buffers.make_request(reference_scores, lq, lt, gap_open,
                                            gap_ext, temperature);
  const auto output = buffers.make_output();
  const auto status = hiko_m::soft_smith_waterman(request, output);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("soft_smith_waterman returned non-ok for valid request");
  }

  const float partition_tolerance =
      temperature >= 1e-3F
          ? 1e-3F * std::max(1.0F, std::fabs(reference_log_partition))
          : 5e-3F * std::max(1.0F, std::fabs(reference_log_partition));
  if (std::fabs(buffers.log_partition - reference_log_partition) >
      partition_tolerance) {
    std::fprintf(stderr,
                 "soft_smith_waterman_module_goldens: case %zu log_partition "
                 "mismatch: ours=%.6f orihime=%.6f tol=%.6f\n",
                 case_index, static_cast<double>(buffers.log_partition),
                 static_cast<double>(reference_log_partition),
                 static_cast<double>(partition_tolerance));
    fail("orihime log_partition parity violated through module");
  }

  const float posterior_tolerance = temperature >= 1e-3F ? 1e-4F : 1e-3F;
  const float worst =
      max_abs_error(buffers.posteriors.data(), reference_posteriors, lq * lt);
  if (worst > posterior_tolerance) {
    std::fprintf(stderr,
                 "soft_smith_waterman_module_goldens: case %zu posterior "
                 "max-abs error %.6f exceeds tolerance %.6f (T=%.6g)\n",
                 case_index, static_cast<double>(worst),
                 static_cast<double>(posterior_tolerance),
                 static_cast<double>(temperature));
    fail("orihime posterior parity violated through module");
  }
}

void test_orihime_reference_parity_through_module() {
  verify_orihime_case(1, kCase1Lq, kCase1Lt, kCase1GapOpen, kCase1GapExt,
                  kCase1Temperature, kCase1LogPartition, kCase1Scores,
                  kCase1Posteriors);
  verify_orihime_case(2, kCase2Lq, kCase2Lt, kCase2GapOpen, kCase2GapExt,
                  kCase2Temperature, kCase2LogPartition, kCase2Scores,
                  kCase2Posteriors);
  verify_orihime_case(3, kCase3Lq, kCase3Lt, kCase3GapOpen, kCase3GapExt,
                  kCase3Temperature, kCase3LogPartition, kCase3Scores,
                  kCase3Posteriors);
  verify_orihime_case(4, kCase4Lq, kCase4Lt, kCase4GapOpen, kCase4GapExt,
                  kCase4Temperature, kCase4LogPartition, kCase4Scores,
                  kCase4Posteriors);
  verify_orihime_case(5, kCase5Lq, kCase5Lt, kCase5GapOpen, kCase5GapExt,
                  kCase5Temperature, kCase5LogPartition, kCase5Scores,
                  kCase5Posteriors);
  verify_orihime_case(6, kCase6Lq, kCase6Lt, kCase6GapOpen, kCase6GapExt,
                  kCase6Temperature, kCase6LogPartition, kCase6Scores,
                  kCase6Posteriors);
  verify_orihime_case(7, kCase7Lq, kCase7Lt, kCase7GapOpen, kCase7GapExt,
                  kCase7Temperature, kCase7LogPartition, kCase7Scores,
                  kCase7Posteriors);
  verify_orihime_case(8, kCase8Lq, kCase8Lt, kCase8GapOpen, kCase8GapExt,
                  kCase8Temperature, kCase8LogPartition, kCase8Scores,
                  kCase8Posteriors);
  verify_orihime_case(9, kCase9Lq, kCase9Lt, kCase9GapOpen, kCase9GapExt,
                  kCase9Temperature, kCase9LogPartition, kCase9Scores,
                  kCase9Posteriors);
  verify_orihime_case(10, kCase10Lq, kCase10Lt, kCase10GapOpen, kCase10GapExt,
                  kCase10Temperature, kCase10LogPartition, kCase10Scores,
                  kCase10Posteriors);
  verify_orihime_case(11, kCase11Lq, kCase11Lt, kCase11GapOpen, kCase11GapExt,
                  kCase11Temperature, kCase11LogPartition, kCase11Scores,
                  kCase11Posteriors);
}

void test_module_reports_posteriors_for_tiny_case() {
  const float score1 = 0.0F;
  const float score2 = static_cast<float>(std::log(2.0));
  const float scores[] = {score1, score2};
  ModuleSoftSwBuffers buffers(1U, 2U);
  const auto request =
      buffers.make_request(scores, 1U, 2U, 0.0F, 0.0F, 1.0F);
  const auto output = buffers.make_output();
  const auto status = hiko_m::soft_smith_waterman(request, output);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("soft_smith_waterman returned non-ok for tiny posterior case");
  }

  const double exp_score2 = std::exp(static_cast<double>(score2));
  const double z = 3.0 + exp_score2;
  require_close("tiny posterior[0]", buffers.posteriors[0], 2.0 / z, 1.0e-5);
  require_close("tiny posterior[1]", buffers.posteriors[1],
                exp_score2 / z, 1.0e-5);
}

void test_module_rejects_null_scores() {
  ModuleSoftSwBuffers buffers(4U, 4U);
  hiko_m::SoftSmithWatermanRequest request =
      buffers.make_request(nullptr, 4U, 4U, -1.4F, -0.15F, 1.0F);
  const auto output = buffers.make_output();
  const auto status = hiko_m::soft_smith_waterman(request, output);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("null scores must return InvalidArgument");
  }
}

void test_module_rejects_zero_dimensions() {
  ModuleSoftSwBuffers buffers(4U, 4U);
  std::vector<float> scores(16U, 0.5F);
  hiko_m::SoftSmithWatermanRequest request =
      buffers.make_request(scores.data(), 0U, 4U, -1.4F, -0.15F, 1.0F);
  hiko_m::SoftSmithWatermanOutput output = buffers.make_output();
  if (hiko_m::soft_smith_waterman(request, output).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("Lq=0 must return InvalidArgument");
  }
  request.query_length = 4U;
  request.target_length = 0U;
  if (hiko_m::soft_smith_waterman(request, output).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("Lt=0 must return InvalidArgument");
  }
}

void test_module_rejects_non_positive_temperature() {
  ModuleSoftSwBuffers buffers(4U, 4U);
  std::vector<float> scores(16U, 0.5F);
  hiko_m::SoftSmithWatermanRequest request =
      buffers.make_request(scores.data(), 4U, 4U, -1.4F, -0.15F, 0.0F);
  const auto output = buffers.make_output();
  if (hiko_m::soft_smith_waterman(request, output).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("temperature=0 must return InvalidArgument");
  }
  request.temperature = -1.0F;
  if (hiko_m::soft_smith_waterman(request, output).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("negative temperature must return InvalidArgument");
  }
}

void test_module_rejects_undersized_workspace() {
  ModuleSoftSwBuffers buffers(4U, 4U);
  std::vector<float> scores(16U, 0.5F);
  hiko_m::SoftSmithWatermanRequest request =
      buffers.make_request(scores.data(), 4U, 4U, -1.4F, -0.15F, 1.0F);
  request.workspace_cells = 4U;
  const auto output = buffers.make_output();
  if (hiko_m::soft_smith_waterman(request, output).code !=
      hiko_u::StatusCode::FailedPrecondition) {
    fail("undersized workspace must return FailedPrecondition");
  }
}

void test_module_rejects_null_log_partition() {
  ModuleSoftSwBuffers buffers(4U, 4U);
  std::vector<float> scores(16U, 0.5F);
  hiko_m::SoftSmithWatermanRequest request =
      buffers.make_request(scores.data(), 4U, 4U, -1.4F, -0.15F, 1.0F);
  hiko_m::SoftSmithWatermanOutput output = buffers.make_output();
  output.log_partition = nullptr;
  if (hiko_m::soft_smith_waterman(request, output).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("null log_partition must return InvalidArgument");
  }
}

void test_module_rejects_null_posteriors() {
  ModuleSoftSwBuffers buffers(4U, 4U);
  std::vector<float> scores(16U, 0.5F);
  hiko_m::SoftSmithWatermanRequest request =
      buffers.make_request(scores.data(), 4U, 4U, -1.4F, -0.15F, 1.0F);
  hiko_m::SoftSmithWatermanOutput output = buffers.make_output();
  output.posteriors = nullptr;
  if (hiko_m::soft_smith_waterman(request, output).code !=
      hiko_u::StatusCode::InvalidArgument) {
    fail("null posteriors must return InvalidArgument");
  }
}

}  // namespace

int main() {
  test_orihime_reference_parity_through_module();
  test_module_reports_posteriors_for_tiny_case();
  test_module_rejects_null_scores();
  test_module_rejects_zero_dimensions();
  test_module_rejects_non_positive_temperature();
  test_module_rejects_undersized_workspace();
  test_module_rejects_null_log_partition();
  test_module_rejects_null_posteriors();
  return 0;
}

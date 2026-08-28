#include <hikoboshi/dispatch/scalar_forward.hpp>

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace hikoboshi::dispatch {
namespace {

#ifndef HIKOBOSHI_GEMM_PARITY_MODE_DEFAULT_FAST
#define HIKOBOSHI_GEMM_PARITY_MODE_DEFAULT_FAST 0
#endif

constexpr GemmParityMode kBuildDefaultParityMode =
#if HIKOBOSHI_GEMM_PARITY_MODE_DEFAULT_FAST
    GemmParityMode::Fast;
#else
    GemmParityMode::Strict;
#endif

GemmParityMode compute_active_gemm_parity_mode() noexcept {
  const char* const raw = std::getenv("HIKOBOSHI_GEMM_PARITY_MODE");
  if (raw == nullptr || raw[0] == '\0') {
    return kBuildDefaultParityMode;
  }
  if (std::strcmp(raw, "strict") == 0) {
    return GemmParityMode::Strict;
  }
  if (std::strcmp(raw, "fast") == 0) {
    return GemmParityMode::Fast;
  }
  return kBuildDefaultParityMode;
}

void reset_path(hikoboshi::universal::AlignmentPath& path) noexcept {
  path.steps.clear();
  path.aligned_pairs = 0;
  path.query_start = hikoboshi::universal::kAlignmentGapSentinel;
  path.query_end = hikoboshi::universal::kAlignmentGapSentinel;
  path.target_start = hikoboshi::universal::kAlignmentGapSentinel;
  path.target_end = hikoboshi::universal::kAlignmentGapSentinel;
}

void copy_traceback_output(
    const hikoboshi::primitives::alignment::TracebackScalarOutput& output,
    hikoboshi::universal::AlignmentPath& path) {
  path.steps.erase(path.steps.begin() +
                       static_cast<std::ptrdiff_t>(output.step_count),
                   path.steps.end());
  path.aligned_pairs = output.aligned_pairs;
  path.query_start = output.query_start;
  path.query_end = output.query_end;
  path.target_start = output.target_start;
  path.target_end = output.target_end;
}

}  // namespace

void knn_forward(ScalarTag,
                 const hikoboshi::primitives::compute::KnnScalarRequest& request,
                 const hikoboshi::primitives::compute::KnnScalarOutput& output) {
  hikoboshi::primitives::compute::knn_scalar(request, output);
}

void rbf_forward(ScalarTag,
                 const hikoboshi::primitives::compute::RbfScalarRequest& request,
                 float* output) {
  hikoboshi::primitives::compute::rbf_scalar(request, output);
}

void atom_pair_distance_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::AtomPairDistanceScalarRequest& request,
    const hikoboshi::primitives::compute::AtomPairDistanceScalarOutput& output) {
  hikoboshi::primitives::compute::atom_pair_distance_scalar(request, output);
}

void gather_forward(ScalarTag,
                    const hikoboshi::primitives::compute::GatherScalarRequest& request,
                    float* output) {
  hikoboshi::primitives::compute::gather_scalar(request, output);
}

void layer_norm_forward(ScalarTag,
                        const hikoboshi::primitives::compute::LayerNormScalarRequest& request,
                        float* output) {
  hikoboshi::primitives::compute::layer_norm_scalar(request, output);
}

void reduce_sum_rows_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::ReduceRowsScalarRequest& request,
    float* output) {
  hikoboshi::primitives::compute::reduce_sum_rows_scalar(request, output);
}

void reduce_mean_rows_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::ReduceRowsScalarRequest& request,
    float* output) {
  hikoboshi::primitives::compute::reduce_mean_rows_scalar(request, output);
}

void softmax_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::SoftmaxScalarRequest& request,
    const hikoboshi::primitives::compute::SoftmaxScalarOutput& output) {
  hikoboshi::primitives::compute::softmax_scalar(request, output);
}

void log_softmax_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::LogSoftmaxScalarRequest& request,
    const hikoboshi::primitives::compute::LogSoftmaxScalarOutput& output) {
  hikoboshi::primitives::compute::log_softmax_scalar(request, output);
}

void gemm_nn_forward(ScalarTag,
                     StrictParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output) {
  hikoboshi::primitives::linalg::gemm_nn_scalar(request, output);
}

void gemm_nn_forward(ScalarTag,
                     FastParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output) {
  hikoboshi::primitives::linalg::gemm_nn_scalar_fast(request, output);
}

void gemm_nt_forward(ScalarTag,
                     StrictParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output) {
  hikoboshi::primitives::linalg::gemm_nt_scalar(request, output);
}

void gemm_nt_forward(ScalarTag,
                     FastParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output) {
  hikoboshi::primitives::linalg::gemm_nt_scalar_fast(request, output);
}

GemmParityMode active_gemm_parity_mode() noexcept {
  static const GemmParityMode kActive = compute_active_gemm_parity_mode();
  return kActive;
}

void gemm_nn_forward(ScalarTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output) {
  if (active_gemm_parity_mode() == GemmParityMode::Fast) {
    hikoboshi::primitives::linalg::gemm_nn_scalar_fast(request, output);
  } else {
    hikoboshi::primitives::linalg::gemm_nn_scalar(request, output);
  }
}

void gemm_nt_forward(ScalarTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output) {
  if (active_gemm_parity_mode() == GemmParityMode::Fast) {
    hikoboshi::primitives::linalg::gemm_nt_scalar_fast(request, output);
  } else {
    hikoboshi::primitives::linalg::gemm_nt_scalar(request, output);
  }
}

void smith_waterman_forward(
    ScalarTag,
    const hikoboshi::primitives::alignment::SmithWatermanScalarRequest& request,
    hikoboshi::primitives::alignment::SmithWatermanScalarOutput& output) {
  hikoboshi::primitives::alignment::smith_waterman_scalar(request, output);
}

void soft_smith_waterman_forward(
    ScalarTag,
    const hikoboshi::primitives::alignment::SoftSmithWatermanScalarRequest& request,
    hikoboshi::primitives::alignment::SoftSmithWatermanScalarOutput& output) {
  hikoboshi::primitives::alignment::soft_smith_waterman_scalar(request, output);
}

void traceback_forward(
    ScalarTag,
    const hikoboshi::primitives::alignment::TracebackScalarRequest& request,
    hikoboshi::universal::AlignmentPath& path) {
  reset_path(path);
  if (request.best_query_index < 0 || request.best_target_index < 0) {
    return;
  }

  const std::size_t max_steps = request.query_length + request.target_length;
  if (path.steps.capacity() < max_steps) {
    return;
  }

  path.steps.assign(max_steps, hikoboshi::universal::AlignmentStep{});
  hikoboshi::primitives::alignment::TracebackScalarOutput output{};
  output.steps = {path.steps.data(), path.steps.size()};
  hikoboshi::primitives::alignment::traceback_scalar(request, output);
  copy_traceback_output(output, path);
}

}  // namespace hikoboshi::dispatch

#ifndef HIKOBOSHI_DISPATCH_SCALAR_FORWARD_HPP
#define HIKOBOSHI_DISPATCH_SCALAR_FORWARD_HPP

#include <hikoboshi/dispatch/backend_tag.hpp>
#include <hikoboshi/primitives/alignment/smith_waterman.hpp>
#include <hikoboshi/primitives/alignment/traceback.hpp>
#include <hikoboshi/primitives/compute/atom_pair_distance.hpp>
#include <hikoboshi/primitives/compute/axpy.hpp>
#include <hikoboshi/primitives/compute/bias_add.hpp>
#include <hikoboshi/primitives/compute/gather.hpp>
#include <hikoboshi/primitives/compute/gelu.hpp>
#include <hikoboshi/primitives/compute/knn.hpp>
#include <hikoboshi/primitives/compute/layer_norm.hpp>
#include <hikoboshi/primitives/compute/log_softmax.hpp>
#include <hikoboshi/primitives/compute/rbf.hpp>
#include <hikoboshi/primitives/compute/reduce.hpp>
#include <hikoboshi/primitives/compute/softmax.hpp>
#include <hikoboshi/primitives/linalg/gemm.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/backend.hpp>

namespace hikoboshi::dispatch {

void knn_forward(ScalarTag,
                 const hikoboshi::primitives::compute::KnnScalarRequest& request,
                 const hikoboshi::primitives::compute::KnnScalarOutput& output);

void rbf_forward(ScalarTag,
                 const hikoboshi::primitives::compute::RbfScalarRequest& request,
                 float* output);

void atom_pair_distance_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::AtomPairDistanceScalarRequest& request,
    const hikoboshi::primitives::compute::AtomPairDistanceScalarOutput& output);

void gather_forward(ScalarTag,
                    const hikoboshi::primitives::compute::GatherScalarRequest& request,
                    float* output);

inline void bias_add_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::BiasAddScalarRequest& request,
    float* output) noexcept {
  hikoboshi::primitives::compute::bias_add_scalar(request, output);
}

inline void gelu_inplace_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::GeluInplaceScalarRequest& request,
    float* output) noexcept {
  hikoboshi::primitives::compute::gelu_inplace_scalar(request, output);
}

inline void axpy_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::AxpyScalarRequest& request,
    float* output) noexcept {
  hikoboshi::primitives::compute::axpy_scalar(request, output);
}

void layer_norm_forward(ScalarTag,
                        const hikoboshi::primitives::compute::LayerNormScalarRequest& request,
                        float* output);

void reduce_sum_rows_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::ReduceRowsScalarRequest& request,
    float* output);

void reduce_mean_rows_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::ReduceRowsScalarRequest& request,
    float* output);

void softmax_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::SoftmaxScalarRequest& request,
    const hikoboshi::primitives::compute::SoftmaxScalarOutput& output);

void log_softmax_forward(
    ScalarTag,
    const hikoboshi::primitives::compute::LogSoftmaxScalarRequest& request,
    const hikoboshi::primitives::compute::LogSoftmaxScalarOutput& output);

// Default GEMM dispatch routes to the active parity mode selected by
// `HIKOBOSHI_GEMM_PARITY_MODE` (env var override) or the build-time
// `HIKOBOSHI_GEMM_PARITY_MODE_DEFAULT_FAST` macro (set from the
// `hikoboshi_gemm_parity_mode` Meson option). Explicit-parity callers
// should pass `StrictParityTag{}` or `FastParityTag{}` instead.
void gemm_nn_forward(ScalarTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output);

void gemm_nt_forward(ScalarTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output);

void gemm_nn_forward(ScalarTag,
                     StrictParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output);

void gemm_nn_forward(ScalarTag,
                     FastParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output);

void gemm_nt_forward(ScalarTag,
                     StrictParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output);

void gemm_nt_forward(ScalarTag,
                     FastParityTag,
                     const hikoboshi::primitives::linalg::GemmScalarRequest& request,
                     float* output);

// Active parity mode for the unqualified `gemm_*_forward` overloads.
//
// Resolves the `HIKOBOSHI_GEMM_PARITY_MODE` env var if set ("strict" or
// "fast") and otherwise returns the build-time default. The result is
// cached after the first call to keep the env lookup off the GEMM hot
// path.
enum class GemmParityMode : unsigned char {
  Strict = 0,
  Fast = 1,
};

GemmParityMode active_gemm_parity_mode() noexcept;

void smith_waterman_forward(
    ScalarTag,
    const hikoboshi::primitives::alignment::SmithWatermanScalarRequest& request,
    hikoboshi::primitives::alignment::SmithWatermanScalarOutput& output);

void soft_smith_waterman_forward(
    ScalarTag,
    const hikoboshi::primitives::alignment::SoftSmithWatermanScalarRequest& request,
    hikoboshi::primitives::alignment::SoftSmithWatermanScalarOutput& output);

void traceback_forward(
    ScalarTag,
    const hikoboshi::primitives::alignment::TracebackScalarRequest& request,
    hikoboshi::universal::AlignmentPath& path);

}  // namespace hikoboshi::dispatch

#endif  // HIKOBOSHI_DISPATCH_SCALAR_FORWARD_HPP

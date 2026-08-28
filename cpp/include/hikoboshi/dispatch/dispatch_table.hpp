#ifndef HIKOBOSHI_DISPATCH_DISPATCH_TABLE_HPP
#define HIKOBOSHI_DISPATCH_DISPATCH_TABLE_HPP

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/universal/backend.hpp>

namespace hikoboshi::dispatch {

struct DispatchTable {
  using KnnForward =
      void (*)(const hikoboshi::primitives::compute::KnnScalarRequest&,
               const hikoboshi::primitives::compute::KnnScalarOutput&);
  using RbfForward =
      void (*)(const hikoboshi::primitives::compute::RbfScalarRequest&, float*);
  using GatherForward =
      void (*)(const hikoboshi::primitives::compute::GatherScalarRequest&, float*);
  using LayerNormForward =
      void (*)(const hikoboshi::primitives::compute::LayerNormScalarRequest&,
               float*);
  using ReduceRowsForward =
      void (*)(const hikoboshi::primitives::compute::ReduceRowsScalarRequest&,
               float*);
  using SoftmaxForward =
      void (*)(const hikoboshi::primitives::compute::SoftmaxScalarRequest&,
               const hikoboshi::primitives::compute::SoftmaxScalarOutput&);
  using LogSoftmaxForward =
      void (*)(const hikoboshi::primitives::compute::LogSoftmaxScalarRequest&,
               const hikoboshi::primitives::compute::LogSoftmaxScalarOutput&);
  using GemmForward =
      void (*)(const hikoboshi::primitives::linalg::GemmScalarRequest&, float*);
  using SmithWatermanForward =
      void (*)(const hikoboshi::primitives::alignment::SmithWatermanScalarRequest&,
               hikoboshi::primitives::alignment::SmithWatermanScalarOutput&);
  using TracebackForward =
      void (*)(const hikoboshi::primitives::alignment::TracebackScalarRequest&,
               hikoboshi::universal::AlignmentPath&);

  hikoboshi::universal::Backend backend;
  KnnForward knn;
  RbfForward rbf;
  GatherForward gather;
  LayerNormForward layer_norm;
  ReduceRowsForward reduce_sum_rows;
  ReduceRowsForward reduce_mean_rows;
  SoftmaxForward softmax;
  LogSoftmaxForward log_softmax;
  GemmForward gemm_nn;
  GemmForward gemm_nt;
  // Parity-mode-specific GEMM entries. Strict matches the existing
  // primitive parity goldens bit-for-bit; fast is the BLIS-style
  // streaming-outer-product variant ported from the archive. The
  // default `gemm_nn` and `gemm_nt` slots above resolve through the
  // active parity mode (Meson option + env override).
  GemmForward gemm_nn_strict;
  GemmForward gemm_nn_fast;
  GemmForward gemm_nt_strict;
  GemmForward gemm_nt_fast;
  SmithWatermanForward smith_waterman;
  TracebackForward traceback;
};

const DispatchTable& scalar_dispatch_table() noexcept;
const DispatchTable& selected_dispatch_table() noexcept;

}  // namespace hikoboshi::dispatch

#endif  // HIKOBOSHI_DISPATCH_DISPATCH_TABLE_HPP

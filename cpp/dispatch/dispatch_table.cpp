#include <hikoboshi/dispatch/dispatch_table.hpp>

#include <cstddef>
#include <string_view>

#include <hikoboshi/dispatch/backend_tag.hpp>
#include <hikoboshi/dispatch/cpu_features.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>
#include <hikoboshi/dispatch/scalar_forward.hpp>

#include "simd/dispatch_select.hpp"

namespace hikoboshi::dispatch {
namespace {

template <typename FnPtr>
FnPtr lookup_entry(const std::string_view op_id) noexcept {
  const registry::RegisteredPrimitiveOpRecord* record =
      registry::find_primitive_op(op_id);
  if (record == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<FnPtr>(const_cast<void*>(record->dispatch_entry));
}

void gemm_nn_scalar_strict_dispatch(
    const hikoboshi::primitives::linalg::GemmScalarRequest& request,
    float* output) {
  gemm_nn_forward(ScalarTag{}, StrictParityTag{}, request, output);
}

void gemm_nn_scalar_fast_dispatch(
    const hikoboshi::primitives::linalg::GemmScalarRequest& request,
    float* output) {
  gemm_nn_forward(ScalarTag{}, FastParityTag{}, request, output);
}

void gemm_nt_scalar_strict_dispatch(
    const hikoboshi::primitives::linalg::GemmScalarRequest& request,
    float* output) {
  gemm_nt_forward(ScalarTag{}, StrictParityTag{}, request, output);
}

void gemm_nt_scalar_fast_dispatch(
    const hikoboshi::primitives::linalg::GemmScalarRequest& request,
    float* output) {
  gemm_nt_forward(ScalarTag{}, FastParityTag{}, request, output);
}

DispatchTable build_scalar_dispatch_table() noexcept {
  DispatchTable table{};
  table.backend = hikoboshi::universal::Backend::Scalar;
  table.knn = lookup_entry<DispatchTable::KnnForward>("hikoboshi.knn.v1");
  table.rbf = lookup_entry<DispatchTable::RbfForward>("hikoboshi.rbf.v1");
  table.gather =
      lookup_entry<DispatchTable::GatherForward>("hikoboshi.gather.v1");
  table.layer_norm =
      lookup_entry<DispatchTable::LayerNormForward>("hikoboshi.layer_norm.v1");
  table.reduce_sum_rows = lookup_entry<DispatchTable::ReduceRowsForward>(
      "hikoboshi.reduce_sum_rows.v1");
  table.reduce_mean_rows = lookup_entry<DispatchTable::ReduceRowsForward>(
      "hikoboshi.reduce_mean_rows.v1");
  table.softmax = lookup_entry<DispatchTable::SoftmaxForward>(
      "hikoboshi.softmax.row_wise.v1");
  table.log_softmax = lookup_entry<DispatchTable::LogSoftmaxForward>(
      "hikoboshi.log_softmax.row_wise.v1");
  table.gemm_nn =
      lookup_entry<DispatchTable::GemmForward>("hikoboshi.gemm.nn.v1");
  table.gemm_nt =
      lookup_entry<DispatchTable::GemmForward>("hikoboshi.gemm.nt.v1");
  table.gemm_nn_strict = &gemm_nn_scalar_strict_dispatch;
  table.gemm_nn_fast = &gemm_nn_scalar_fast_dispatch;
  table.gemm_nt_strict = &gemm_nt_scalar_strict_dispatch;
  table.gemm_nt_fast = &gemm_nt_scalar_fast_dispatch;
  table.smith_waterman = lookup_entry<DispatchTable::SmithWatermanForward>(
      "hikoboshi.smith_waterman.v1");
  table.traceback =
      lookup_entry<DispatchTable::TracebackForward>("hikoboshi.traceback.v1");
  return table;
}

}  // namespace

const DispatchTable& scalar_dispatch_table() noexcept {
  static const DispatchTable kScalarDispatchTable = build_scalar_dispatch_table();
  return kScalarDispatchTable;
}

const DispatchTable& selected_dispatch_table() noexcept {
  const CpuFeatures& features = detected_cpu_features();
  if (avx512_dispatch_available(features)) {
    return avx512_dispatch_table();
  }
  if (avx2_dispatch_available(features)) {
    return avx2_dispatch_table();
  }
  return scalar_dispatch_table();
}

}  // namespace hikoboshi::dispatch

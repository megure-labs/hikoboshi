#include <hikoboshi/dispatch/registry/primitive_op.hpp>

#include <cstddef>
#include <string_view>

#include <hikoboshi/dispatch/backend_tag.hpp>
#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor_role.hpp>

namespace hikoboshi::dispatch::registry {
namespace {

namespace hiko_u = hikoboshi::universal;
namespace hiko_p = hikoboshi::primitives;

// Entry-wrapper functions. Each wrapper has a stable function-pointer
// signature so the dispatch table can cast `dispatch_entry` back to the
// matching `DispatchTable` typedef.

void knn_scalar_entry(const hiko_p::compute::KnnScalarRequest& request,
                      const hiko_p::compute::KnnScalarOutput& output) {
  knn_forward(ScalarTag{}, request, output);
}

void rbf_scalar_entry(const hiko_p::compute::RbfScalarRequest& request,
                      float* output) {
  rbf_forward(ScalarTag{}, request, output);
}

void gather_scalar_entry(const hiko_p::compute::GatherScalarRequest& request,
                         float* output) {
  gather_forward(ScalarTag{}, request, output);
}

void layer_norm_scalar_entry(
    const hiko_p::compute::LayerNormScalarRequest& request, float* output) {
  layer_norm_forward(ScalarTag{}, request, output);
}

void reduce_sum_rows_scalar_entry(
    const hiko_p::compute::ReduceRowsScalarRequest& request, float* output) {
  reduce_sum_rows_forward(ScalarTag{}, request, output);
}

void reduce_mean_rows_scalar_entry(
    const hiko_p::compute::ReduceRowsScalarRequest& request, float* output) {
  reduce_mean_rows_forward(ScalarTag{}, request, output);
}

void softmax_scalar_entry(const hiko_p::compute::SoftmaxScalarRequest& request,
                          const hiko_p::compute::SoftmaxScalarOutput& output) {
  softmax_forward(ScalarTag{}, request, output);
}

void log_softmax_scalar_entry(
    const hiko_p::compute::LogSoftmaxScalarRequest& request,
    const hiko_p::compute::LogSoftmaxScalarOutput& output) {
  log_softmax_forward(ScalarTag{}, request, output);
}

void bias_add_scalar_entry(const hiko_p::compute::BiasAddScalarRequest& request,
                           float* output) {
  bias_add_forward(ScalarTag{}, request, output);
}

void gelu_scalar_entry(const hiko_p::compute::GeluInplaceScalarRequest& request,
                       float* output) {
  gelu_inplace_forward(ScalarTag{}, request, output);
}

void axpy_scalar_entry(const hiko_p::compute::AxpyScalarRequest& request,
                       float* output) {
  axpy_forward(ScalarTag{}, request, output);
}

void atom_pair_distance_scalar_entry(
    const hiko_p::compute::AtomPairDistanceScalarRequest& request,
    const hiko_p::compute::AtomPairDistanceScalarOutput& output) {
  atom_pair_distance_forward(ScalarTag{}, request, output);
}

void gemm_nn_scalar_entry(const hiko_p::linalg::GemmScalarRequest& request,
                          float* output) {
  gemm_nn_forward(ScalarTag{}, request, output);
}

void gemm_nt_scalar_entry(const hiko_p::linalg::GemmScalarRequest& request,
                          float* output) {
  gemm_nt_forward(ScalarTag{}, request, output);
}

void smith_waterman_scalar_entry(
    const hiko_p::alignment::SmithWatermanScalarRequest& request,
    hiko_p::alignment::SmithWatermanScalarOutput& output) {
  smith_waterman_forward(ScalarTag{}, request, output);
}

void traceback_scalar_entry(
    const hiko_p::alignment::TracebackScalarRequest& request,
    hiko_u::AlignmentPath& path) {
  traceback_forward(ScalarTag{}, request, path);
}

// Family textual spellings.
constexpr std::string_view kFamilyCompute{"compute"};
constexpr std::string_view kFamilyLinalg{"linalg"};
constexpr std::string_view kFamilyAlignment{"alignment"};

constexpr std::string_view kVersionV1{"v1"};

// Op identity strings. The dotted form is `hikoboshi.<op_name>.v1`; GEMM
// adds the matrix-orientation segment as a sub-name.
constexpr std::string_view kOpIdKnn{"hikoboshi.knn.v1"};
constexpr std::string_view kOpIdRbf{"hikoboshi.rbf.v1"};
constexpr std::string_view kOpIdGather{"hikoboshi.gather.v1"};
constexpr std::string_view kOpIdLayerNorm{"hikoboshi.layer_norm.v1"};
constexpr std::string_view kOpIdReduceSumRows{"hikoboshi.reduce_sum_rows.v1"};
constexpr std::string_view kOpIdReduceMeanRows{"hikoboshi.reduce_mean_rows.v1"};
constexpr std::string_view kOpIdSoftmaxRowWise{"hikoboshi.softmax.row_wise.v1"};
constexpr std::string_view kOpIdLogSoftmaxRowWise{
    "hikoboshi.log_softmax.row_wise.v1"};
constexpr std::string_view kOpIdBiasAdd{"hikoboshi.bias_add.v1"};
constexpr std::string_view kOpIdGelu{"hikoboshi.gelu.v1"};
constexpr std::string_view kOpIdAxpy{"hikoboshi.axpy.v1"};
constexpr std::string_view kOpIdAtomPairDistance{
    "hikoboshi.atom_pair_distance.v1"};
constexpr std::string_view kOpIdGemmNn{"hikoboshi.gemm.nn.v1"};
constexpr std::string_view kOpIdGemmNt{"hikoboshi.gemm.nt.v1"};
constexpr std::string_view kOpIdSmithWaterman{"hikoboshi.smith_waterman.v1"};
constexpr std::string_view kOpIdTraceback{"hikoboshi.traceback.v1"};

// Capability arrays. All ops in the closed 0.1.0 set are wired only on
// the scalar CPU backend. The `gemm-dual-mode-strict-fast` packet will
// extend the GEMM records' supported_parity_modes to include `Fast` once
// the BLIS-style microkernel lands.
constexpr hiko_u::PackageBackendRequirement kScalarBackends[] = {
    hiko_u::PackageBackendRequirement::CpuScalar,
};

constexpr ParityMode kStrictOnly[] = {ParityMode::Strict};
constexpr ParityMode kStrictOrFast[] = {ParityMode::Strict, ParityMode::Fast};

// Per-op signature role arrays. Spans reference these constexpr arrays.

constexpr hiko_u::TensorRole kKnnInputs[] = {
    hiko_u::TensorRole::Coordinates3d,
    hiko_u::TensorRole::Coordinates3d,
    hiko_u::TensorRole::ValidityMask,
    hiko_u::TensorRole::ValidityMask,
};
constexpr hiko_u::TensorRole kKnnOutputs[] = {
    hiko_u::TensorRole::NeighborIndices,
    hiko_u::TensorRole::SquaredDistances,
};
constexpr hiko_u::ParameterRole kKnnParameters[] = {
    hiko_u::ParameterRole::NeighborCount,
    hiko_u::ParameterRole::IncludeSelfFlag,
    hiko_u::ParameterRole::TreatZeroAsInvalidFlag,
};

constexpr hiko_u::TensorRole kRbfInputs[] = {
    hiko_u::TensorRole::SquaredDistances,
};
constexpr hiko_u::TensorRole kRbfOutputs[] = {
    hiko_u::TensorRole::RbfFeatures,
};
constexpr hiko_u::ParameterRole kRbfParameters[] = {
    hiko_u::ParameterRole::FeatureCount,
    hiko_u::ParameterRole::CenterMin,
    hiko_u::ParameterRole::CenterMax,
    hiko_u::ParameterRole::Sigma,
};

constexpr hiko_u::TensorRole kGatherInputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
    hiko_u::TensorRole::GatherIndices,
};
constexpr hiko_u::TensorRole kGatherOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};

constexpr hiko_u::TensorRole kLayerNormInputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
    hiko_u::TensorRole::EncoderNormGamma,
    hiko_u::TensorRole::EncoderNormBeta,
};
constexpr hiko_u::TensorRole kLayerNormOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};
constexpr hiko_u::ParameterRole kLayerNormParameters[] = {
    hiko_u::ParameterRole::Epsilon,
};

constexpr hiko_u::TensorRole kReduceRowsInputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};
constexpr hiko_u::TensorRole kReduceRowsOutputs[] = {
    hiko_u::TensorRole::ReducedVector,
};

// Softmax consumes a row-major activation matrix and produces an
// activation matrix of identical shape. The temperature scalar and the
// optional additive mask have no exact match in the closed
// `ParameterRole` / `TensorRole` vocabularies yet; the registered
// signature therefore lists only the activation tensors. Future
// vocabulary additions can broaden the signature without changing the
// op identity.
constexpr hiko_u::TensorRole kSoftmaxInputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};
constexpr hiko_u::TensorRole kSoftmaxOutputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};

constexpr hiko_u::TensorRole kLogSoftmaxInputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};
constexpr hiko_u::TensorRole kLogSoftmaxOutputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};

constexpr hiko_u::TensorRole kBiasAddInputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
    hiko_u::TensorRole::EncoderBias,
};
constexpr hiko_u::TensorRole kBiasAddOutputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};

constexpr hiko_u::TensorRole kGeluInputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};
constexpr hiko_u::TensorRole kGeluOutputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};

constexpr hiko_u::TensorRole kAxpyInputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
    hiko_u::TensorRole::ActivationMatrix,
};
constexpr hiko_u::TensorRole kAxpyOutputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};
constexpr hiko_u::ParameterRole kAxpyParameters[] = {
    hiko_u::ParameterRole::Alpha,
};

constexpr hiko_u::TensorRole kAtomPairDistanceInputs[] = {
    hiko_u::TensorRole::Coordinates3d,
    hiko_u::TensorRole::AtomSource,
    hiko_u::TensorRole::NeighborIndices,
};
constexpr hiko_u::TensorRole kAtomPairDistanceOutputs[] = {
    hiko_u::TensorRole::SquaredDistances,
};

constexpr hiko_u::TensorRole kGemmInputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
    hiko_u::TensorRole::EncoderWeight,
};
constexpr hiko_u::TensorRole kGemmOutputs[] = {
    hiko_u::TensorRole::ActivationMatrix,
};
constexpr hiko_u::ParameterRole kGemmParameters[] = {
    hiko_u::ParameterRole::RowCount,
    hiko_u::ParameterRole::InnerDim,
    hiko_u::ParameterRole::ColumnCount,
};

constexpr hiko_u::TensorRole kSmithWatermanInputs[] = {
    hiko_u::TensorRole::ScoreMatrix,
    hiko_u::TensorRole::GapParameterRecord,
};
constexpr hiko_u::TensorRole kSmithWatermanOutputs[] = {
    hiko_u::TensorRole::AlignmentPath,
};
constexpr hiko_u::ParameterRole kSmithWatermanParameters[] = {
    hiko_u::ParameterRole::GapOpen,
    hiko_u::ParameterRole::GapExtension,
};

constexpr hiko_u::TensorRole kTracebackInputs[] = {
    hiko_u::TensorRole::ScoreMatrix,
};
constexpr hiko_u::TensorRole kTracebackOutputs[] = {
    hiko_u::TensorRole::AlignmentPath,
};

template <typename T, std::size_t N>
constexpr hiko_u::Span<const T> as_span(const T (&array)[N]) noexcept {
  return {array, N};
}

}  // namespace

universal::Span<const RegisteredPrimitiveOpRecord>
primitive_op_registry() noexcept {
  static const RegisteredPrimitiveOpRecord kRecords[] = {
      {
          {kOpIdKnn, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kKnnInputs), as_span(kKnnOutputs), as_span(kKnnParameters)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&knn_scalar_entry),
      },
      {
          {kOpIdRbf, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kRbfInputs), as_span(kRbfOutputs), as_span(kRbfParameters)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&rbf_scalar_entry),
      },
      {
          {kOpIdGather, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kGatherInputs), as_span(kGatherOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&gather_scalar_entry),
      },
      {
          {kOpIdLayerNorm, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kLayerNormInputs), as_span(kLayerNormOutputs),
           as_span(kLayerNormParameters)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&layer_norm_scalar_entry),
      },
      {
          {kOpIdReduceSumRows, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kReduceRowsInputs), as_span(kReduceRowsOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&reduce_sum_rows_scalar_entry),
      },
      {
          {kOpIdReduceMeanRows, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kReduceRowsInputs), as_span(kReduceRowsOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&reduce_mean_rows_scalar_entry),
      },
      {
          {kOpIdSoftmaxRowWise, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kSoftmaxInputs), as_span(kSoftmaxOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOrFast), ParityMode::Fast},
          reinterpret_cast<const void*>(&softmax_scalar_entry),
      },
      {
          {kOpIdLogSoftmaxRowWise, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kLogSoftmaxInputs), as_span(kLogSoftmaxOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&log_softmax_scalar_entry),
      },
      {
          {kOpIdBiasAdd, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kBiasAddInputs), as_span(kBiasAddOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&bias_add_scalar_entry),
      },
      {
          {kOpIdGelu, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kGeluInputs), as_span(kGeluOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&gelu_scalar_entry),
      },
      {
          {kOpIdAxpy, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kAxpyInputs), as_span(kAxpyOutputs),
           as_span(kAxpyParameters)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&axpy_scalar_entry),
      },
      {
          {kOpIdAtomPairDistance, kFamilyCompute, kVersionV1},
          PrimitiveOpFamily::Compute,
          {as_span(kAtomPairDistanceInputs), as_span(kAtomPairDistanceOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&atom_pair_distance_scalar_entry),
      },
      {
          {kOpIdGemmNn, kFamilyLinalg, kVersionV1},
          PrimitiveOpFamily::Linalg,
          {as_span(kGemmInputs), as_span(kGemmOutputs),
           as_span(kGemmParameters)},
          {as_span(kScalarBackends), as_span(kStrictOrFast),
           ParityMode::Strict},
          reinterpret_cast<const void*>(&gemm_nn_scalar_entry),
      },
      {
          {kOpIdGemmNt, kFamilyLinalg, kVersionV1},
          PrimitiveOpFamily::Linalg,
          {as_span(kGemmInputs), as_span(kGemmOutputs),
           as_span(kGemmParameters)},
          {as_span(kScalarBackends), as_span(kStrictOrFast),
           ParityMode::Strict},
          reinterpret_cast<const void*>(&gemm_nt_scalar_entry),
      },
      {
          {kOpIdSmithWaterman, kFamilyAlignment, kVersionV1},
          PrimitiveOpFamily::Alignment,
          {as_span(kSmithWatermanInputs), as_span(kSmithWatermanOutputs),
           as_span(kSmithWatermanParameters)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&smith_waterman_scalar_entry),
      },
      {
          {kOpIdTraceback, kFamilyAlignment, kVersionV1},
          PrimitiveOpFamily::Alignment,
          {as_span(kTracebackInputs), as_span(kTracebackOutputs),
           hiko_u::Span<const hiko_u::ParameterRole>{nullptr, 0}},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&traceback_scalar_entry),
      },
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

const RegisteredPrimitiveOpRecord* find_primitive_op(
    const std::string_view op_id) noexcept {
  const universal::Span<const RegisteredPrimitiveOpRecord> records =
      primitive_op_registry();
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].identity.op_id == op_id) {
      return &records.data[index];
    }
  }
  return nullptr;
}

}  // namespace hikoboshi::dispatch::registry

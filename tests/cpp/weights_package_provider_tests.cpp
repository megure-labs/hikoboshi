#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include "../../cpp/weights/embedded_mpnn_d64.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace hiko_w = hikoboshi::weights;
namespace hiko_g = hikoboshi::weights::generated::mpnn_d64;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;

static_assert(std::is_same<hiko_d::Mpnn64Weights,
                           hiko_u::detail::Mpnn64Weights>::value,
              "default_mpnn64 opaque state must use the exact module view type");

std::uint64_t stage_flag(const hiko_u::PackageValidationStage stage) {
  return 1ull << static_cast<unsigned>(stage);
}

bool has_error(const hiko_u::PackageValidationReport& report,
               const hiko_u::PackageValidationStage stage,
               const std::string_view code) {
  for (std::size_t index = 0; index < report.diagnostics.size; ++index) {
    const hiko_u::PackageValidationDiagnostic& diagnostic =
        report.diagnostics.data[index];
    if (diagnostic.severity == hiko_u::PackageDiagnosticSeverity::Error &&
        diagnostic.stage == stage && diagnostic.code == code &&
        !diagnostic.message.empty()) {
      return true;
    }
  }
  return false;
}

bool has_buffer_error(const hiko_w::PackageValidationBuffer& buffer,
                      std::size_t diagnostic_count,
                      const hiko_u::PackageValidationStage stage,
                      const std::string_view code) {
  for (std::size_t index = 0; index < diagnostic_count; ++index) {
    const hiko_u::PackageValidationDiagnostic& diagnostic =
        buffer.diagnostics[index];
    if (diagnostic.severity == hiko_u::PackageDiagnosticSeverity::Error &&
        diagnostic.stage == stage && diagnostic.code == code &&
        !diagnostic.message.empty()) {
      return true;
    }
  }
  return false;
}

bool report_rejects(const hiko_u::PackageValidationReport& report,
                    const hiko_u::PackageValidationStage stage,
                    const std::string_view code) {
  return !report.ok && report.accepted_handle.descriptor == nullptr &&
         (report.failed_stage_flags & stage_flag(stage)) != 0 &&
         has_error(report, stage, code);
}

hiko_u::PackageValidationReport validate_descriptor(
    hiko_u::PackageDescriptor& descriptor,
    hiko_w::PackageValidationBuffer& buffer) {
  const hiko_u::PackageHandle package{nullptr, &descriptor};
  return hiko_w::validate_mpnn64_package(package, buffer);
}

std::array<hiko_g::TensorBlobInfo, hiko_g::kRuntimeTensorCount> tensor_fixture() {
  std::array<hiko_g::TensorBlobInfo, hiko_g::kRuntimeTensorCount> tensors{};
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    tensors[index] = hiko_g::kRuntimeTensors[index];
  }
  return tensors;
}

bool fixture_rejects(
    const std::array<hiko_g::TensorBlobInfo, hiko_g::kRuntimeTensorCount>& tensors,
    const std::string_view code) {
  hiko_w::PackageValidationBuffer buffer{};
  std::size_t diagnostic_count = 0;
  const bool ok = hiko_w::detail::validate_mpnn_d64_generated_tensors(
      {tensors.data(), tensors.size()}, buffer, diagnostic_count);
  return !ok &&
         has_buffer_error(
             buffer, diagnostic_count,
             hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes, code);
}

bool tensor_name_present(const hiko_u::WeightsView& view, std::string_view name) {
  for (std::size_t index = 0; index < view.tensors.size; ++index) {
    if (view.tensors.data[index].name == name) {
      return true;
    }
  }
  return false;
}

hiko_u::PackageDescriptor substitution_matrix_descriptor_fixture() {
  static constexpr hiko_u::PackageInputKind kSequenceTokenInputs[1] = {
      hiko_u::PackageInputKind::SequenceTokens,
  };
  static constexpr hiko_u::PackagePreprocessingCapability kTokenization[1] = {
      hiko_u::PackagePreprocessingCapability::Tokenization,
  };
  static constexpr hiko_u::PackageOutputKind kSubstitutionOutputs[1] = {
      hiko_u::PackageOutputKind::SubstitutionScores,
  };
  static constexpr hiko_u::DataType kDtypes[1] = {
      hiko_u::DataType::Float32,
  };
  static constexpr hiko_u::PackageTensorLayout kLayouts[1] = {
      hiko_u::PackageTensorLayout::RowMajor,
  };
  static constexpr hiko_u::PackageBackendRequirement kBackends[1] = {
      hiko_u::PackageBackendRequirement::CpuScalar,
  };
  static constexpr hiko_u::ScoreInputKind kScoreInputs[1] = {
      hiko_u::ScoreInputKind::SequenceTokens,
  };
  static constexpr hiko_u::TracebackPolicy kTracebacks[2] = {
      hiko_u::TracebackPolicy::RequiredForPublicPairwise,
      hiko_u::TracebackPolicy::RequiredForPublicAllVsAll,
  };

  hiko_u::PackageDescriptor descriptor{};
  descriptor.identity = {"0.1.0",
                         "Hikoboshi-BLOSUM62-future-fixture",
                         "Hikoboshi-substitution-matrix",
                         "future-fixture-v1",
                         hiko_u::PackageKind::SubstitutionMatrix};
  descriptor.execution = {hiko_u::PackageExecutionMode::RegisteredArchitecture,
                          "hikoboshi_substitution_matrix_v1",
                          {kBackends, 1}};
  descriptor.capabilities = {0,
                             {kSequenceTokenInputs, 1},
                             {kTokenization, 1},
                             {kSubstitutionOutputs, 1},
                             {kDtypes, 1},
                             {kLayouts, 1},
                             {kBackends, 1}};
  descriptor.inputs = {{kSequenceTokenInputs, 1}};
  descriptor.outputs = {{kSubstitutionOutputs, 1}};
  descriptor.scoring = {
      hiko_u::ScoreMethod::SubstitutionLookupV1,
      {kScoreInputs, 1},
      hiko_u::ScoreOutputKind::ScoreMatrix,
      {hiko_u::DataType::Float32,
       hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget,
       true,
       true,
       hiko_u::ScoreNormalization::CalibratedLogOdds,
       hiko_u::ScoreScaleFamily::LogOdds,
       hiko_u::ScoreMethod::SubstitutionLookupV1}};
    descriptor.gaps = {"Hikoboshi 0.1.0 hard-SW",
                       hiko_u::GapModel::Affine,
                       hiko_w::kHardSwDefaultGapOpen,
                       hiko_w::kHardSwDefaultGapExtension,
                       hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
                       hiko_u::ScoreMethod::SubstitutionLookupV1};
    descriptor.soft_gaps = {"Hikoboshi 0.1.0 soft-SW",
                            hiko_u::GapModel::Affine,
                            hiko_w::kSoftSwMpnn64GapOpen,
                            hiko_w::kSoftSwMpnn64GapExtension,
                            hiko_u::GapConvention::
                                GapOpenPlusKMinusOneGapExtension,
                            hiko_u::ScoreMethod::SubstitutionLookupV1};
  descriptor.alignment = {hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1,
                          {kTracebacks, 2}};
  descriptor.compatibility_views = {{nullptr, nullptr}};
  return descriptor;
}

int main() {
  const hiko_u::Result<hiko_w::PackageHandle> package_result =
      hiko_w::default_mpnn64_package();
  if (package_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.descriptor == nullptr) {
    return 1;
  }

  const hiko_u::PackageDescriptor& descriptor =
      *package_result.value.descriptor;
  const hiko_u::Result<hiko_w::PackageHandle> registry_package =
      hiko_w::default_package("hikoboshi-mpnn-d64");
  const hiko_u::Result<hiko_w::PackageHandle> registry_alias =
      hiko_w::default_package("mpnn64");
  const hiko_u::Result<hiko_w::PackageHandle> registry_old_name =
      hiko_w::default_package("Hikoboshi-MPNN-64");
  if (registry_package.status.code != hiko_u::StatusCode::Ok ||
      registry_alias.status.code != hiko_u::StatusCode::Ok ||
      registry_old_name.status.code != hiko_u::StatusCode::Ok ||
      registry_package.value.descriptor != package_result.value.descriptor ||
      registry_alias.value.descriptor != package_result.value.descriptor ||
      registry_old_name.value.descriptor != package_result.value.descriptor ||
      registry_package.value.opaque != package_result.value.opaque ||
      registry_alias.value.opaque != package_result.value.opaque ||
      registry_old_name.value.opaque != package_result.value.opaque) {
    return 24;
  }
  hiko_w::PackageValidationBuffer buffer{};
  const hiko_u::PackageValidationReport valid_report =
      hiko_w::validate_mpnn64_package(package_result.value, buffer);
  if (!valid_report.ok || valid_report.diagnostics.size != 0 ||
      valid_report.warnings.size != 0 ||
      (valid_report.passed_stage_flags &
       stage_flag(hiko_u::PackageValidationStage::PreparedStateBuild)) == 0) {
    return 2;
  }
  if (valid_report.accepted_handle.opaque != package_result.value.opaque ||
      valid_report.accepted_handle.descriptor != package_result.value.descriptor) {
    return 19;
  }

  const hiko_u::Result<hiko_u::WeightsHandle> weights_result =
      hiko_w::default_mpnn64();
  // Regression for the API/CLI/Python default-weight routes: they consume the
  // compatibility WeightsHandle from default_mpnn64(), which must remain the
  // exact view accepted from the compiled hikoboshi-mpnn-d64 package validation.
  if (weights_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.opaque != descriptor.compatibility_views.weights.opaque ||
      weights_result.value.opaque != package_result.value.opaque ||
      weights_result.value.view == nullptr ||
      weights_result.value.view !=
          descriptor.compatibility_views.weights.view ||
      weights_result.value.view->metadata.model_name !=
          hiko_w::kDefaultMpnnD64ModelName ||
      weights_result.value.view->metadata.checksum !=
          hiko_w::default_mpnn_d64_manifest().checksum) {
    return 3;
  }
  if (weights_result.value.opaque == hiko_g::kSafetensorsBlob ||
      tensor_name_present(*weights_result.value.view, "gap") ||
      tensor_name_present(*weights_result.value.view, "gap_open")) {
    return 11;
  }
  const auto* prepared =
      static_cast<const hiko_d::Mpnn64Weights*>(weights_result.value.opaque);
  if (prepared == nullptr || prepared->layer_count != hiko_w::kDefaultMpnn64LayerCount ||
      prepared->layers == nullptr ||
      prepared->W_e.weight.size !=
          hiko_d::kMpnn64HiddenDimension * hiko_d::kMpnn64HiddenDimension ||
      prepared->W_e.bias.size != hiko_d::kMpnn64HiddenDimension ||
      prepared->edge_embedding.linear.weight.size !=
          hiko_d::kMpnn64HiddenDimension * hiko_d::kMpnn64EdgeFeatureCount ||
      prepared->edge_embedding.norm.weight.size != hiko_d::kMpnn64HiddenDimension ||
      prepared->positional_encoding.weight.size !=
          hiko_d::kMpnn64PositionalEncodingCount *
              hiko_d::kMpnn64PositionalEncodingInputDimension ||
      prepared->layers[0].W1.weight.size !=
          hiko_d::kMpnn64HiddenDimension * hiko_d::kMpnn64MessageInputDimension ||
      prepared->layers[0].ffn.W_in.weight.size !=
          hiko_d::kMpnn64FfnHiddenDimension * hiko_d::kMpnn64HiddenDimension ||
      prepared->layers[0].norm1.weight.size != hiko_d::kMpnn64HiddenDimension ||
      prepared->input.projection_weight != prepared->W_e.weight.data ||
      prepared->layers[0].edge_projection_weight !=
          prepared->edge_embedding.linear.weight.data ||
      prepared->layers[0].message_weight != prepared->layers[0].W2.weight.data ||
      prepared->layers[0].ffn1_weight !=
          prepared->layers[0].ffn.W_in.weight.data ||
      prepared->layers[0].ffn2_weight !=
          prepared->layers[0].ffn.W_out.weight.data) {
    return 18;
  }

  if (descriptor.identity.package_id != hiko_w::kDefaultMpnnD64ModelName ||
      descriptor.identity.package_family != hiko_w::kDefaultMpnn64ModelFamily ||
      descriptor.identity.aliases.size != 3 ||
      descriptor.identity.aliases.data[0] != "mpnn64" ||
      descriptor.identity.aliases.data[1] != "mpnn-64" ||
      descriptor.identity.aliases.data[2] != "Hikoboshi-MPNN-64" ||
      descriptor.execution.mode !=
          hiko_u::PackageExecutionMode::RegisteredArchitecture ||
      descriptor.execution.architecture_id !=
          hiko_w::kDefaultMpnn64ArchitectureId ||
      descriptor.scoring.method != hiko_u::ScoreMethod::RawDotV1 ||
      descriptor.scoring.semantics.dtype != hiko_u::DataType::Float32 ||
      descriptor.scoring.semantics.layout !=
          hiko_u::ScoreMatrixLayout::RowMajorQueryByTarget ||
      descriptor.scoring.semantics.normalization !=
          hiko_u::ScoreNormalization::None ||
      descriptor.scoring.semantics.scale_family !=
          hiko_u::ScoreScaleFamily::RawDot ||
        descriptor.gaps.gap_open != hiko_w::kHardSwDefaultGapOpen ||
        descriptor.gaps.gap_extension != hiko_w::kHardSwDefaultGapExtension ||
        descriptor.soft_gaps.gap_open != hiko_w::kSoftSwMpnn64GapOpen ||
        descriptor.soft_gaps.gap_extension !=
            hiko_w::kSoftSwMpnn64GapExtension ||
        descriptor.alignment.algorithm !=
          hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1 ||
      descriptor.capabilities.dtypes.data[0] != hiko_u::DataType::Float32 ||
      descriptor.capabilities.layouts.data[0] !=
          hiko_u::PackageTensorLayout::RowMajor ||
      descriptor.execution.backend_requirements.data[0] !=
          hiko_u::PackageBackendRequirement::CpuScalar) {
    return 4;
  }

  {
    hiko_u::PackageDescriptor rejected = descriptor;
    rejected.execution.mode = hiko_u::PackageExecutionMode::GraphIr;
    hiko_w::PackageValidationBuffer reject_buffer{};
    const hiko_u::PackageValidationReport report =
        validate_descriptor(rejected, reject_buffer);
    if (!report_rejects(
            report, hiko_u::PackageValidationStage::ArchitectureRegistration,
            "unsupported_execution_mode")) {
      return 5;
    }
  }

  {
    hiko_u::PackageDescriptor rejected = descriptor;
    const hiko_u::PackageInputKind inputs[1] = {
        hiko_u::PackageInputKind::StructureAllAtom};
    rejected.inputs.routes = {inputs, 1};
    rejected.capabilities.input_routes = {inputs, 1};
    hiko_w::PackageValidationBuffer reject_buffer{};
    const hiko_u::PackageValidationReport report =
        validate_descriptor(rejected, reject_buffer);
    if (!report_rejects(report, hiko_u::PackageValidationStage::InputRoute,
                        "unsupported_input_modality")) {
      return 6;
    }
  }

  {
    hiko_u::PackageDescriptor rejected =
        substitution_matrix_descriptor_fixture();
    if (rejected.identity.package_kind !=
            hiko_u::PackageKind::SubstitutionMatrix ||
        rejected.inputs.routes.data[0] !=
            hiko_u::PackageInputKind::SequenceTokens ||
        rejected.outputs.kinds.data[0] !=
            hiko_u::PackageOutputKind::SubstitutionScores ||
        rejected.scoring.method !=
            hiko_u::ScoreMethod::SubstitutionLookupV1 ||
        rejected.scoring.semantics.scale_family !=
            hiko_u::ScoreScaleFamily::LogOdds ||
        rejected.gaps.model != hiko_u::GapModel::Affine ||
        rejected.alignment.algorithm !=
            hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1) {
      return 20;
    }
    hiko_w::PackageValidationBuffer reject_buffer{};
    const hiko_u::PackageValidationReport report =
        validate_descriptor(rejected, reject_buffer);
    if (!report_rejects(report, hiko_u::PackageValidationStage::InputRoute,
                        "unsupported_input_modality")) {
      return 21;
    }
    if (!report_rejects(report, hiko_u::PackageValidationStage::ScoringMethod,
                        "unsupported_score_method")) {
      return 22;
    }
    if (!report_rejects(report,
                        hiko_u::PackageValidationStage::WorkflowCompatibility,
                        "unsupported_output_kind")) {
      return 23;
    }
  }

  {
    hiko_u::PackageDescriptor rejected = descriptor;
    rejected.scoring.method = hiko_u::ScoreMethod::CosineV1;
    hiko_w::PackageValidationBuffer reject_buffer{};
    const hiko_u::PackageValidationReport report =
        validate_descriptor(rejected, reject_buffer);
    if (!report_rejects(report, hiko_u::PackageValidationStage::ScoringMethod,
                        "unsupported_score_method")) {
      return 7;
    }
  }

  {
    hiko_u::PackageDescriptor rejected = descriptor;
    rejected.alignment.algorithm =
        hiko_u::AlignmentAlgorithmId::GlobalAffineSwV1;
    hiko_w::PackageValidationBuffer reject_buffer{};
    const hiko_u::PackageValidationReport report =
        validate_descriptor(rejected, reject_buffer);
    if (!report_rejects(
            report, hiko_u::PackageValidationStage::AlignmentAlgorithm,
            "unsupported_alignment_algorithm")) {
      return 8;
    }
  }

  {
    hiko_u::PackageDescriptor rejected = descriptor;
    const hiko_u::PackageBackendRequirement backends[1] = {
        hiko_u::PackageBackendRequirement::GpuCuda};
    rejected.execution.backend_requirements = {backends, 1};
    rejected.capabilities.backends = {backends, 1};
    hiko_w::PackageValidationBuffer reject_buffer{};
    const hiko_u::PackageValidationReport report =
        validate_descriptor(rejected, reject_buffer);
    if (!report_rejects(
            report, hiko_u::PackageValidationStage::WorkflowCompatibility,
            "unsupported_backend_requirement")) {
      return 9;
    }
  }

  {
    hiko_u::PackageDescriptor rejected = descriptor;
    const hiko_u::DataType dtypes[1] = {hiko_u::DataType::Float64};
    rejected.capabilities.dtypes = {dtypes, 1};
    hiko_w::PackageValidationBuffer reject_buffer{};
    const hiko_u::PackageValidationReport report =
        validate_descriptor(rejected, reject_buffer);
    if (!report_rejects(
            report,
            hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
            "unsupported_dtype")) {
      return 10;
    }
  }

  {
    auto tensors = tensor_fixture();
    tensors[0].name = "removed.W_e.bias";
    if (!fixture_rejects(tensors, "missing_required_tensor")) {
      return 12;
    }
  }

  {
    auto tensors = tensor_fixture();
    tensors[0].dtype = "float64";
    if (!fixture_rejects(tensors, "tensor_dtype_mismatch")) {
      return 13;
    }
  }

  {
    auto tensors = tensor_fixture();
    const std::size_t bad_shape[1] = {63};
    tensors[0].shape = bad_shape;
    if (!fixture_rejects(tensors, "tensor_shape_mismatch")) {
      return 14;
    }
  }

  {
    auto tensors = tensor_fixture();
    tensors[0].byte_length += sizeof(float);
    if (!fixture_rejects(tensors, "tensor_byte_length_mismatch")) {
      return 15;
    }
  }

  {
    auto tensors = tensor_fixture();
    tensors[0].data_offset += 1;
    if (!fixture_rejects(tensors, "tensor_alignment_mismatch")) {
      return 16;
    }
  }

  {
    auto tensors = tensor_fixture();
    tensors[0].name = "gap";
    if (!fixture_rejects(tensors, "historical_tensor_in_runtime_weights")) {
      return 17;
    }
  }

  return 0;
}

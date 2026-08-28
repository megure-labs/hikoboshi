#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace hiko_w = hikoboshi::weights;
namespace hiko_u = hikoboshi::universal;

constexpr bool gap_pairs_differ(float lhs_open,
                                float lhs_extension,
                                float rhs_open,
                                float rhs_extension) {
  return lhs_open != rhs_open || lhs_extension != rhs_extension;
}

static_assert(std::is_standard_layout<hiko_w::TensorManifestView>::value,
              "TensorManifestView must stay a plain view");
static_assert(std::is_standard_layout<hiko_w::WeightManifestView>::value,
              "WeightManifestView must stay a plain view");
static_assert(hiko_w::kDefaultMpnn64HiddenDim == 64,
              "Hikoboshi-MPNN-64 hidden dimension is part of the weights contract");
static_assert(hiko_w::kDefaultMpnn64NeighborCount == 64,
              "Hikoboshi-MPNN-64 KNN count is part of the weights contract");
static_assert(hiko_w::kDefaultMpnn64RbfCount == 16,
              "Hikoboshi-MPNN-64 RBF order is part of the weights contract");
static_assert(hiko_w::kDefaultMpnn64LayerCount == 3,
              "Hikoboshi-MPNN-64 layer count is part of the weights contract");
static_assert(hiko_w::kHardSwDefaultGapOpen == -1.40000F,
              "default weights must preserve hard-SW gap_open");
static_assert(hiko_w::kHardSwDefaultGapExtension == -0.150000F,
              "default weights must preserve hard-SW gap_ext");
static_assert(hiko_w::kSoftSwMpnn64GapOpen == -3.21337F,
              "default weights must expose MPNN soft-SW gap_open");
static_assert(hiko_w::kSoftSwMpnn64GapExtension == -0.111704F,
              "default weights must expose MPNN soft-SW gap_ext");
static_assert(hiko_w::kHardSwEsm2_8mGapOpen == -1.01982F,
              "default weights must expose ESM2 hard-SW gap_open");
static_assert(hiko_w::kHardSwEsm2_8mGapExtension == +0.225736F,
              "default weights must expose ESM2 hard-SW gap_ext");
static_assert(hiko_w::kSoftSwEsm2_8mGapOpen == -6.72805F,
              "default weights must expose ESM2 soft-SW gap_open");
static_assert(hiko_w::kSoftSwEsm2_8mGapExtension == -0.0159468F,
              "default weights must expose ESM2 soft-SW gap_ext");
static_assert(
    gap_pairs_differ(hiko_w::kHardSwDefaultGapOpen,
                     hiko_w::kHardSwDefaultGapExtension,
                     hiko_w::kSoftSwMpnn64GapOpen,
                     hiko_w::kSoftSwMpnn64GapExtension) &&
        gap_pairs_differ(hiko_w::kHardSwDefaultGapOpen,
                         hiko_w::kHardSwDefaultGapExtension,
                         hiko_w::kHardSwEsm2_8mGapOpen,
                         hiko_w::kHardSwEsm2_8mGapExtension) &&
        gap_pairs_differ(hiko_w::kHardSwDefaultGapOpen,
                         hiko_w::kHardSwDefaultGapExtension,
                         hiko_w::kSoftSwEsm2_8mGapOpen,
                         hiko_w::kSoftSwEsm2_8mGapExtension) &&
        gap_pairs_differ(hiko_w::kSoftSwMpnn64GapOpen,
                         hiko_w::kSoftSwMpnn64GapExtension,
                         hiko_w::kHardSwEsm2_8mGapOpen,
                         hiko_w::kHardSwEsm2_8mGapExtension) &&
        gap_pairs_differ(hiko_w::kSoftSwMpnn64GapOpen,
                         hiko_w::kSoftSwMpnn64GapExtension,
                         hiko_w::kSoftSwEsm2_8mGapOpen,
                         hiko_w::kSoftSwEsm2_8mGapExtension) &&
        gap_pairs_differ(hiko_w::kHardSwEsm2_8mGapOpen,
                         hiko_w::kHardSwEsm2_8mGapExtension,
                         hiko_w::kSoftSwEsm2_8mGapOpen,
                         hiko_w::kSoftSwEsm2_8mGapExtension),
    "the MPNN/ESM2 hard/T=1 gap pairs must remain distinct");
static_assert(std::is_same<hiko_w::PackageHandle, hiko_u::PackageHandle>::value,
              "weights provider headers may mention universal package handles");
static_assert(std::is_standard_layout<hiko_w::PackageValidationBuffer>::value,
              "weights package validation buffer must stay a plain record");

int main() {
  const hiko_w::WeightManifestView& manifest = hiko_w::default_mpnn_d64_manifest();
  hiko_w::PackageValidationBuffer validation_buffer{};
  const hiko_u::Result<hiko_w::PackageHandle> package_result =
      hiko_w::default_mpnn_d64_package();
  const hiko_u::PackageValidationReport validation_report =
      hiko_w::validate_mpnn_d64_package(package_result.value, validation_buffer);
  const hiko_u::Result<hiko_u::WeightsHandle> result = hiko_w::default_mpnn_d64();
  constexpr std::string_view kTensorSchema{
      "safetensors:hikoboshi-mpnn-d64-v1;runtime_tensors=73;excluded=gap,gap_open"};

  return manifest.model_name == hiko_w::kDefaultMpnnD64ModelName &&
                 manifest.model_family == hiko_w::kDefaultMpnn64ModelFamily &&
                 manifest.hidden_dimension == hiko_w::kDefaultMpnn64HiddenDim &&
                 manifest.neighbor_count == hiko_w::kDefaultMpnn64NeighborCount &&
                 manifest.rbf_count == hiko_w::kDefaultMpnn64RbfCount &&
                 manifest.rbf_feature_order == hiko_w::kDefaultMpnn64RbfFeatureOrder &&
                 manifest.layer_count == hiko_w::kDefaultMpnn64LayerCount &&
                 manifest.tensor_schema == kTensorSchema &&
                 manifest.message_scale == hiko_w::kDefaultMpnn64MessageScale &&
                 manifest.dtype == std::string_view{"float32"} &&
                 manifest.tensors.size == 73 &&
                 manifest.tensors.data[0].name == std::string_view{"W_e.bias"} &&
                 manifest.tensors.data[0].shape.size == 1 &&
                 manifest.tensors.data[0].shape.data[0] == 64 &&
                 manifest.tensors.data[0].checksum.size() == 64 &&
                   manifest.gap_open == hiko_w::kHardSwDefaultGapOpen &&
                   manifest.gap_extension == hiko_w::kHardSwDefaultGapExtension &&
                   manifest.soft_gap_open == hiko_w::kSoftSwMpnn64GapOpen &&
                   manifest.soft_gap_extension ==
                       hiko_w::kSoftSwMpnn64GapExtension &&
                   manifest.similarity == std::string_view{"raw_dot_product"} &&
                 manifest.provenance_status != hiko_w::kPendingProvenanceSentinel &&
                 package_result.status.code == hiko_u::StatusCode::Ok &&
                 package_result.value.descriptor != nullptr &&
                 package_result.value.descriptor->identity.package_id ==
                     hiko_w::kDefaultMpnnD64ModelName &&
                 package_result.value.descriptor->execution.architecture_id ==
                     hiko_w::kDefaultMpnn64ArchitectureId &&
                 package_result.value.descriptor->scoring.method ==
                     hiko_u::ScoreMethod::RawDotV1 &&
                 package_result.value.descriptor->alignment.algorithm ==
                     hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1 &&
                 validation_report.ok &&
                 validation_report.diagnostics.size == 0 &&
                 result.status.code == hiko_u::StatusCode::Ok &&
                 result.value.opaque != nullptr && result.value.view != nullptr &&
                 result.value.view->metadata.model_name ==
                     hiko_w::kDefaultMpnnD64ModelName &&
                 result.value.view->metadata.checksum == manifest.checksum &&
                 result.value.view->tensors.size == manifest.tensors.size &&
                 result.value.view->tensors.data[0].name ==
                     manifest.tensors.data[0].name &&
                 result.value.view ==
                     package_result.value.descriptor->compatibility_views.weights.view
             ? 0
             : 1;
}

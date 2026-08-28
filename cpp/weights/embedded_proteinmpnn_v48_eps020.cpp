#include "embedded_proteinmpnn_v48_eps020.hpp"

#include "generated/proteinmpnn_v48_eps020_blob.hpp"

#include <hikoboshi/universal/detail/proteinmpnn_v48_020_schema.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>
#include <hikoboshi/universal/tensor.hpp>
#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <array>
#include <cstddef>
#include <string_view>

namespace hikoboshi::weights::detail {
namespace {

namespace generated = ::hikoboshi::weights::generated::proteinmpnn_v48_eps020;
namespace hiko_u = ::hikoboshi::universal;
namespace hiko_d = ::hikoboshi::universal::detail;

using FloatSpan = hiko_u::Span<const float>;
using ExpectedTensor = hiko_d::ProteinMpnnV48020TensorSchemaEntry;

static_assert(hiko_d::kProteinMpnnV48020SchemaTensorCount ==
                  generated::kRuntimeTensorCount,
              "ProteinMPNN v_48_020 schema count must match generated blob");
static_assert(hiko_d::kProteinMpnnV48020WeightViewTensorCount == 118,
              "ProteinMPNN v_48_020 typed weight view must contain 118 tensors");

hiko_u::TensorView tensor_view_for(
    const generated::TensorBlobInfo& info) noexcept {
  return {generated::kSafetensorsBlob + generated::kSafetensorsDataOffset +
              info.data_offset,
          {info.shape, info.rank},
          {info.strides, info.rank},
          hiko_u::DataType::Float32,
          info.name};
}

const std::array<hiko_u::TensorView, generated::kRuntimeTensorCount>&
tensor_views() noexcept {
  static const auto kViews = []() {
    std::array<hiko_u::TensorView, generated::kRuntimeTensorCount> views{};
    for (std::size_t index = 0; index < generated::kRuntimeTensorCount;
         ++index) {
      views[index] = tensor_view_for(generated::kRuntimeTensors[index]);
    }
    return views;
  }();
  return kViews;
}

void add_error(PackageValidationBuffer& buffer,
               std::size_t& diagnostic_count,
               const hiko_u::PackageValidationStage stage,
               const std::string_view code,
               const std::string_view message) noexcept {
  if (diagnostic_count < kPackageValidationDiagnosticCapacity) {
    buffer.diagnostics[diagnostic_count++] = {
        hiko_u::PackageDiagnosticSeverity::Error,
        stage,
        code,
        message,
    };
  }
}

bool expected_tensor_name(std::string_view name) noexcept {
  for (const ExpectedTensor& expected : hiko_d::kProteinMpnnV48020TensorSchema) {
    if (name == expected.name) {
      return true;
    }
  }
  return false;
}

std::size_t expected_byte_length(const ExpectedTensor& expected) noexcept {
  return expected.element_count * sizeof(float);
}

bool same_expected_shape(const ExpectedTensor& expected,
                         const generated::TensorBlobInfo& info) noexcept {
  if (info.rank != expected.rank || info.shape == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < expected.rank; ++index) {
    if (info.shape[index] != expected.shape[index]) {
      return false;
    }
  }
  return true;
}

bool row_major_strides(const generated::TensorBlobInfo& info) noexcept {
  if (info.strides == nullptr || info.shape == nullptr || info.rank == 0) {
    return false;
  }
  std::size_t expected_stride = 1;
  for (std::size_t reverse = 0; reverse < info.rank; ++reverse) {
    const std::size_t index = info.rank - reverse - 1;
    if (info.strides[index] != expected_stride) {
      return false;
    }
    expected_stride *= info.shape[index];
  }
  return true;
}

bool aligned_float_range(const generated::TensorBlobInfo& info) noexcept {
  if (info.data_offset + info.byte_length > generated::kSafetensorsDataLength ||
      info.byte_length % sizeof(float) != 0) {
    return false;
  }
  return (generated::kSafetensorsDataOffset + info.data_offset) %
             alignof(float) ==
         0;
}

const generated::TensorBlobInfo* find_tensor(
    hiko_u::Span<const generated::TensorBlobInfo> tensors,
    std::string_view name,
    std::size_t& match_count) noexcept {
  const generated::TensorBlobInfo* found = nullptr;
  match_count = 0;
  for (std::size_t index = 0; index < tensors.size; ++index) {
    if (tensors.data[index].name == name) {
      found = &tensors.data[index];
      ++match_count;
    }
  }
  return found;
}

FloatSpan float_span_for(const generated::TensorBlobInfo* info) noexcept {
  if (info == nullptr) {
    return {nullptr, 0};
  }
  const void* raw = generated::kSafetensorsBlob +
                    generated::kSafetensorsDataOffset + info->data_offset;
  return {static_cast<const float*>(raw), info->byte_length / sizeof(float)};
}

hiko_d::ProteinMpnnV48020LinearWeights linear(std::string_view prefix) noexcept {
  const generated::TensorBlobInfo* weight = nullptr;
  const generated::TensorBlobInfo* bias = nullptr;
  for (const generated::TensorBlobInfo& info : generated::kRuntimeTensors) {
    if (info.name.size() == prefix.size() + std::string_view{".weight"}.size() &&
        info.name.substr(0, prefix.size()) == prefix &&
        info.name.substr(prefix.size()) == std::string_view{".weight"}) {
      weight = &info;
    } else if (info.name.size() ==
                   prefix.size() + std::string_view{".bias"}.size() &&
               info.name.substr(0, prefix.size()) == prefix &&
               info.name.substr(prefix.size()) == std::string_view{".bias"}) {
      bias = &info;
    }
  }
  return {float_span_for(weight), float_span_for(bias)};
}

hiko_d::ProteinMpnnV48020NormWeights norm(std::string_view prefix) noexcept {
  const hiko_d::ProteinMpnnV48020LinearWeights as_linear = linear(prefix);
  return {as_linear.weight, as_linear.bias};
}

hiko_d::ProteinMpnnV48020EmbeddingWeights embedding(
    std::string_view prefix) noexcept {
  const std::string_view suffix{".weight"};
  for (const generated::TensorBlobInfo& info : generated::kRuntimeTensors) {
    if (info.name.size() == prefix.size() + suffix.size() &&
        info.name.substr(0, prefix.size()) == prefix &&
        info.name.substr(prefix.size()) == suffix) {
      return {float_span_for(&info)};
    }
  }
  return {{nullptr, 0}};
}

hiko_d::ProteinMpnnV48020EdgeEmbeddingWeights edge_embedding(
    std::string_view prefix) noexcept {
  return {embedding(prefix).weight};
}

hiko_d::ProteinMpnnV48020EncoderLayerWeights encoder_layer(
    std::size_t index) noexcept {
  switch (index) {
    case 0:
      return {linear("encoder_layers.0.W1"),
              linear("encoder_layers.0.W2"),
              linear("encoder_layers.0.W3"),
              linear("encoder_layers.0.W11"),
              linear("encoder_layers.0.W12"),
              linear("encoder_layers.0.W13"),
              norm("encoder_layers.0.norm1"),
              norm("encoder_layers.0.norm2"),
              norm("encoder_layers.0.norm3"),
              {linear("encoder_layers.0.dense.W_in"),
               linear("encoder_layers.0.dense.W_out")}};
    case 1:
      return {linear("encoder_layers.1.W1"),
              linear("encoder_layers.1.W2"),
              linear("encoder_layers.1.W3"),
              linear("encoder_layers.1.W11"),
              linear("encoder_layers.1.W12"),
              linear("encoder_layers.1.W13"),
              norm("encoder_layers.1.norm1"),
              norm("encoder_layers.1.norm2"),
              norm("encoder_layers.1.norm3"),
              {linear("encoder_layers.1.dense.W_in"),
               linear("encoder_layers.1.dense.W_out")}};
    default:
      return {linear("encoder_layers.2.W1"),
              linear("encoder_layers.2.W2"),
              linear("encoder_layers.2.W3"),
              linear("encoder_layers.2.W11"),
              linear("encoder_layers.2.W12"),
              linear("encoder_layers.2.W13"),
              norm("encoder_layers.2.norm1"),
              norm("encoder_layers.2.norm2"),
              norm("encoder_layers.2.norm3"),
              {linear("encoder_layers.2.dense.W_in"),
               linear("encoder_layers.2.dense.W_out")}};
  }
}

hiko_d::ProteinMpnnV48020DecoderLayerWeights decoder_layer(
    std::size_t index) noexcept {
  switch (index) {
    case 0:
      return {linear("decoder_layers.0.W1"),
              linear("decoder_layers.0.W2"),
              linear("decoder_layers.0.W3"),
              norm("decoder_layers.0.norm1"),
              norm("decoder_layers.0.norm2"),
              {linear("decoder_layers.0.dense.W_in"),
               linear("decoder_layers.0.dense.W_out")}};
    case 1:
      return {linear("decoder_layers.1.W1"),
              linear("decoder_layers.1.W2"),
              linear("decoder_layers.1.W3"),
              norm("decoder_layers.1.norm1"),
              norm("decoder_layers.1.norm2"),
              {linear("decoder_layers.1.dense.W_in"),
               linear("decoder_layers.1.dense.W_out")}};
    default:
      return {linear("decoder_layers.2.W1"),
              linear("decoder_layers.2.W2"),
              linear("decoder_layers.2.W3"),
              norm("decoder_layers.2.norm1"),
              norm("decoder_layers.2.norm2"),
              {linear("decoder_layers.2.dense.W_in"),
               linear("decoder_layers.2.dense.W_out")}};
  }
}

std::array<hiko_d::ProteinMpnnV48020EncoderLayerWeights,
           hiko_d::kProteinMpnnV48020NumEncoderLayers>
make_encoder_layers() noexcept {
  return {encoder_layer(0), encoder_layer(1), encoder_layer(2)};
}

std::array<hiko_d::ProteinMpnnV48020DecoderLayerWeights,
           hiko_d::kProteinMpnnV48020NumDecoderLayers>
make_decoder_layers() noexcept {
  return {decoder_layer(0), decoder_layer(1), decoder_layer(2)};
}

hiko_d::ProteinMpnnV48020FeatureWeights make_features() noexcept {
  return {{linear("features.embeddings.linear")},
          edge_embedding("features.edge_embedding"),
          norm("features.norm_edges")};
}

hiko_d::ProteinMpnnV48020Weights make_prepared_weights() noexcept {
  return {make_features(),
          linear("W_e"),
          embedding("W_s"),
          make_encoder_layers(),
          make_decoder_layers(),
          linear("W_out")};
}

const hiko_d::ProteinMpnnV48020Weights kPreparedWeights =
    make_prepared_weights();

const hiko_u::ModelMetadataView& metadata() noexcept {
  static const hiko_u::ModelMetadataView kMetadata{
      kDefaultProteinMpnnV48Eps020ModelName,
      kDefaultProteinMpnnV48020ModelFamily,
      "v_48_020",
      kDefaultProteinMpnnV48020HiddenDim,
      generated::kSourceArtifactSha256,
      generated::kSafetensorsBlobSha256,
  };
  return kMetadata;
}

const hiko_u::WeightsView& weights_view() noexcept {
  static const hiko_u::WeightsView kView{
      metadata(),
      {tensor_views().data(), tensor_views().size()},
  };
  return kView;
}

const hiko_u::WeightsHandle& weights_handle() noexcept {
  static const hiko_u::WeightsHandle kHandle{
      &kPreparedWeights,
      &weights_view(),
  };
  return kHandle;
}

bool same_shape(const hiko_u::Span<const std::size_t>& manifest_shape,
                const generated::TensorBlobInfo& info) noexcept {
  if (manifest_shape.size != info.rank) {
    return false;
  }
  for (std::size_t index = 0; index < info.rank; ++index) {
    if (manifest_shape.data[index] != info.shape[index]) {
      return false;
    }
  }
  return true;
}

}  // namespace

const hiko_u::WeightsHandle& embedded_proteinmpnn_v48_eps020_handle() noexcept {
  return weights_handle();
}

bool validate_proteinmpnn_v48_eps020_generated_tensors(
    hiko_u::Span<const generated::TensorBlobInfo> tensors,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = true;
  if (tensors.size != generated::kRuntimeTensorCount) {
    ok = false;
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
              "tensor_count_mismatch",
              "ProteinMPNN v_48_020 runtime tensor table must contain exactly 118 tensors");
  }

  for (std::size_t index = 0; index < tensors.size; ++index) {
    const generated::TensorBlobInfo& info = tensors.data[index];
    if (!expected_tensor_name(info.name)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "unexpected_runtime_tensor",
                "ProteinMPNN v_48_020 runtime tensor table contains an unexpected tensor name");
    }
  }

  for (const ExpectedTensor& expected : hiko_d::kProteinMpnnV48020TensorSchema) {
    std::size_t match_count = 0;
    const generated::TensorBlobInfo* info =
        find_tensor(tensors, expected.name, match_count);
    if (match_count == 0 || info == nullptr) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "missing_required_tensor",
                "ProteinMPNN v_48_020 runtime tensor table is missing a required tensor");
      continue;
    }
    if (match_count > 1) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "duplicate_required_tensor",
                "ProteinMPNN v_48_020 runtime tensor table contains a duplicate tensor");
    }
    if (info->dtype != std::string_view{"float32"}) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_dtype_mismatch",
                "ProteinMPNN v_48_020 runtime tensors must be float32");
    }
    if (info->rank != expected.rank) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_rank_mismatch",
                "ProteinMPNN v_48_020 runtime tensor rank does not match the schema");
    }
    if (!same_expected_shape(expected, *info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_shape_mismatch",
                "ProteinMPNN v_48_020 runtime tensor shape does not match the schema");
    }
    if (info->byte_length != expected_byte_length(expected)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_byte_length_mismatch",
                "ProteinMPNN v_48_020 runtime tensor byte length does not match its float32 shape");
    }
    if (!row_major_strides(*info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_stride_mismatch",
                "ProteinMPNN v_48_020 runtime tensors must use contiguous row-major strides");
    }
    if (!aligned_float_range(*info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_alignment_mismatch",
                "ProteinMPNN v_48_020 runtime tensor byte range must be in-bounds and float-aligned");
    }
  }
  return ok;
}

bool embedded_proteinmpnn_v48_eps020_manifest_matches() noexcept {
  const WeightManifestView& manifest = default_proteinmpnn_v48_eps020_manifest();
  if (manifest.model_name != kDefaultProteinMpnnV48Eps020ModelName ||
      manifest.model_family != kDefaultProteinMpnnV48020ModelFamily ||
      manifest.hidden_dimension != kDefaultProteinMpnnV48020HiddenDim ||
      manifest.neighbor_count != kDefaultProteinMpnnV48020NeighborCount ||
      manifest.rbf_count != kDefaultProteinMpnnV48020RbfCount ||
      manifest.layer_count != kDefaultProteinMpnnV48020LayerCount ||
      manifest.tensor_schema != generated::kTensorSchema ||
      manifest.message_scale != kDefaultProteinMpnnV48020MessageScale ||
      manifest.dtype != std::string_view{"float32"} ||
      manifest.checksum != generated::kSafetensorsBlobSha256 ||
      manifest.checksum_algorithm != std::string_view{"sha256"} ||
      manifest.source_checkpoint_checksum != generated::kSourceArtifactSha256 ||
      manifest.gap_parameter_family != kInverseFoldingGapFamily ||
      manifest.gap_open != 0.0F ||
      manifest.gap_extension != 0.0F ||
      manifest.soft_gap_parameter_family != kInverseFoldingGapFamily ||
      manifest.soft_gap_open != 0.0F ||
      manifest.soft_gap_extension != 0.0F ||
      manifest.similarity != kInverseFoldingSimilarity ||
      manifest.tensors.size != generated::kRuntimeTensorCount ||
      weights_view().tensors.size != generated::kRuntimeTensorCount) {
    return false;
  }

  for (std::size_t index = 0; index < generated::kRuntimeTensorCount; ++index) {
    const TensorManifestView& manifest_tensor = manifest.tensors.data[index];
    const generated::TensorBlobInfo& generated_tensor =
        generated::kRuntimeTensors[index];
    if (generated_tensor.data_offset + generated_tensor.byte_length >
            generated::kSafetensorsDataLength ||
        manifest_tensor.name != generated_tensor.name ||
        manifest_tensor.dtype != generated_tensor.dtype ||
        manifest_tensor.checksum != generated_tensor.checksum ||
        !same_shape(manifest_tensor.shape, generated_tensor)) {
      return false;
    }
  }
  return true;
}

}  // namespace hikoboshi::weights::detail

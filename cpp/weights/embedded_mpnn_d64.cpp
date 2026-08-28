#include "embedded_mpnn_d64.hpp"

#include "generated/mpnn_d64_blob.hpp"

#include <hikoboshi/universal/detail/mpnn_d64_weights.hpp>
#include <hikoboshi/universal/detail/mpnn_d64_slots.hpp>
#include <hikoboshi/weights/manifest.hpp>

#include <array>
#include <cstddef>
#include <string_view>

namespace hikoboshi::weights::detail {
namespace {

namespace generated = hikoboshi::weights::generated::mpnn_d64;
namespace hiko_u = hikoboshi::universal;

using FloatSpan = hiko_u::Span<const float>;
using RepresentativeWEWeightSlot =
    hiko_u::detail::Mpnn64Slot<hiko_u::detail::Mpnn64SlotId::w_e_weight>;

static_assert(RepresentativeWEWeightSlot::data_offset ==
                  generated::kRuntimeTensors[1].data_offset,
              "hikoboshi-mpnn-d64 W_e.weight slot offset must match metadata");
static_assert(RepresentativeWEWeightSlot::byte_length ==
                  generated::kRuntimeTensors[1].byte_length,
              "hikoboshi-mpnn-d64 W_e.weight slot bytes must match metadata");
static_assert(RepresentativeWEWeightSlot::rank ==
                  generated::kRuntimeTensors[1].rank,
              "hikoboshi-mpnn-d64 W_e.weight slot rank must match metadata");

struct ExpectedTensor {
  std::string_view name;
  std::size_t rank;
  std::size_t shape[2];
};

#define HIKOBOSHI_MPNN64_LINEAR(prefix, out_dim, in_dim) \
  {prefix ".bias", 1, {out_dim, 0}}, {prefix ".weight", 2, {out_dim, in_dim}}

#define HIKOBOSHI_MPNN64_NORM(prefix) \
  {prefix ".bias", 1, {64, 0}}, {prefix ".weight", 1, {64, 0}}

#define HIKOBOSHI_MPNN64_LAYER(index)                                  \
  HIKOBOSHI_MPNN64_LINEAR("layers." #index ".W1", 64, 192),           \
      HIKOBOSHI_MPNN64_LINEAR("layers." #index ".W11", 64, 192),      \
      HIKOBOSHI_MPNN64_LINEAR("layers." #index ".W12", 64, 64),       \
      HIKOBOSHI_MPNN64_LINEAR("layers." #index ".W13", 64, 64),       \
      HIKOBOSHI_MPNN64_LINEAR("layers." #index ".W2", 64, 64),        \
      HIKOBOSHI_MPNN64_LINEAR("layers." #index ".W3", 64, 64),        \
      HIKOBOSHI_MPNN64_LINEAR("layers." #index ".ffn.W_in", 256, 64), \
      HIKOBOSHI_MPNN64_LINEAR("layers." #index ".ffn.W_out", 64, 256), \
      HIKOBOSHI_MPNN64_NORM("layers." #index ".norm1"),               \
      HIKOBOSHI_MPNN64_NORM("layers." #index ".norm2"),               \
      HIKOBOSHI_MPNN64_NORM("layers." #index ".norm3")

constexpr ExpectedTensor kExpectedTensors[] = {
    HIKOBOSHI_MPNN64_LINEAR("W_e", 64, 64),
    {"edge_embedding.norm.bias", 1, {64, 0}},
    {"edge_embedding.norm.weight", 1, {64, 0}},
    {"edge_embedding.weight", 2, {64, 416}},
    HIKOBOSHI_MPNN64_LAYER(0),
    HIKOBOSHI_MPNN64_LAYER(1),
    HIKOBOSHI_MPNN64_LAYER(2),
    HIKOBOSHI_MPNN64_LINEAR("positional_encoding", 16, 66),
};

#undef HIKOBOSHI_MPNN64_LAYER
#undef HIKOBOSHI_MPNN64_NORM
#undef HIKOBOSHI_MPNN64_LINEAR

static_assert(sizeof(kExpectedTensors) / sizeof(kExpectedTensors[0]) ==
                  generated::kRuntimeTensorCount,
              "hikoboshi-mpnn-d64 expected tensor table must match runtime count");

hikoboshi::universal::TensorView tensor_view_for(
    const generated::TensorBlobInfo& info) noexcept {
  return {generated::kSafetensorsBlob + generated::kSafetensorsDataOffset +
              info.data_offset,
          {info.shape, info.rank},
          {info.strides, info.rank},
          hikoboshi::universal::DataType::Float32,
          info.name};
}

std::array<hikoboshi::universal::TensorView, generated::kRuntimeTensorCount>
make_tensor_views() noexcept {
  std::array<hikoboshi::universal::TensorView,
             generated::kRuntimeTensorCount>
      views{};
  for (std::size_t index = 0; index < generated::kRuntimeTensorCount; ++index) {
    views[index] = tensor_view_for(generated::kRuntimeTensors[index]);
  }
  return views;
}

const auto kTensors = make_tensor_views();

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

bool ignored_historical_tensor_name(std::string_view name) noexcept {
  for (const std::string_view ignored : generated::kIgnoredHistoricalTensorNames) {
    if (name == ignored) {
      return true;
    }
  }
  return false;
}

bool expected_tensor_name(std::string_view name) noexcept {
  for (const ExpectedTensor& expected : kExpectedTensors) {
    if (name == expected.name) {
      return true;
    }
  }
  return false;
}

std::size_t expected_byte_length(const ExpectedTensor& expected) noexcept {
  std::size_t elements = 1;
  for (std::size_t index = 0; index < expected.rank; ++index) {
    elements *= expected.shape[index];
  }
  return elements * sizeof(float);
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

hiko_u::detail::Mpnn64LinearWeights linear(std::string_view prefix) noexcept {
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

template <hiko_u::detail::Mpnn64SlotId WeightSlotId>
hiko_u::detail::Mpnn64LinearWeights linear_with_representative_weight_slot(
    std::string_view prefix) noexcept {
  using WeightSlot = hiko_u::detail::Mpnn64Slot<WeightSlotId>;
  static_assert(WeightSlot::rank == 2,
                "representative linear weight slot must be rank-2");
  static_assert(WeightSlot::element_byte_length == sizeof(float),
                "representative linear weight slot must be float32");
  return linear(prefix);
}

hiko_u::detail::Mpnn64NormWeights norm(std::string_view prefix) noexcept {
  const hiko_u::detail::Mpnn64LinearWeights as_linear = linear(prefix);
  return {as_linear.weight, as_linear.bias};
}

hiko_u::detail::Mpnn64LayerWeights with_compat_aliases(
    hiko_u::detail::Mpnn64LayerWeights layer_weights) noexcept {
  const hiko_u::detail::Mpnn64LinearWeights edge_embedding =
      linear("edge_embedding");
  layer_weights.edge_projection_weight = edge_embedding.weight.data;
  layer_weights.message_norm_gamma = layer_weights.norm1.weight.data;
  layer_weights.message_norm_beta = layer_weights.norm1.bias.data;
  layer_weights.message_weight = layer_weights.W2.weight.data;
  layer_weights.message_bias = layer_weights.W2.bias.data;
  layer_weights.ffn1_weight = layer_weights.ffn.W_in.weight.data;
  layer_weights.ffn1_bias = layer_weights.ffn.W_in.bias.data;
  layer_weights.ffn_norm_gamma = layer_weights.norm3.weight.data;
  layer_weights.ffn_norm_beta = layer_weights.norm3.bias.data;
  layer_weights.ffn2_weight = layer_weights.ffn.W_out.weight.data;
  layer_weights.ffn2_bias = layer_weights.ffn.W_out.bias.data;
  return layer_weights;
}

hiko_u::detail::Mpnn64LayerWeights layer(std::size_t index) noexcept {
  switch (index) {
    case 0:
      return with_compat_aliases(
          {nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           linear("layers.0.W1"),
           linear("layers.0.W11"),
           linear("layers.0.W12"),
           linear("layers.0.W13"),
           linear("layers.0.W2"),
           linear("layers.0.W3"),
           {linear("layers.0.ffn.W_in"),
            linear("layers.0.ffn.W_out")},
           norm("layers.0.norm1"),
           norm("layers.0.norm2"),
           norm("layers.0.norm3")});
    case 1:
      return with_compat_aliases(
          {nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           linear("layers.1.W1"),
           linear("layers.1.W11"),
           linear("layers.1.W12"),
           linear("layers.1.W13"),
           linear("layers.1.W2"),
           linear("layers.1.W3"),
           {linear("layers.1.ffn.W_in"),
            linear("layers.1.ffn.W_out")},
           norm("layers.1.norm1"),
           norm("layers.1.norm2"),
           norm("layers.1.norm3")});
    default:
      return with_compat_aliases(
          {nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           nullptr,
           linear("layers.2.W1"),
           linear("layers.2.W11"),
           linear("layers.2.W12"),
           linear("layers.2.W13"),
           linear("layers.2.W2"),
           linear("layers.2.W3"),
           {linear("layers.2.ffn.W_in"),
            linear("layers.2.ffn.W_out")},
           norm("layers.2.norm1"),
           norm("layers.2.norm2"),
           norm("layers.2.norm3")});
  }
}

std::array<hiko_u::detail::Mpnn64LayerWeights, kDefaultMpnn64LayerCount>
make_prepared_layers() noexcept {
  return {layer(0), layer(1), layer(2)};
}

const auto kPreparedLayers = make_prepared_layers();

hiko_u::detail::Mpnn64Weights make_prepared_weights() noexcept {
  const hiko_u::detail::Mpnn64LinearWeights W_e =
      linear_with_representative_weight_slot<
          hiko_u::detail::Mpnn64SlotId::w_e_weight>("W_e");
  return {{W_e.weight.data, W_e.bias.data},
          kPreparedLayers.data(),
          nullptr,
          nullptr,
          W_e,
          {linear("edge_embedding"), norm("edge_embedding.norm")},
          linear("positional_encoding"),
          kPreparedLayers.size()};
}

const hiko_u::detail::Mpnn64Weights kPreparedWeights = make_prepared_weights();

const hikoboshi::universal::ModelMetadataView kMetadata{
    kDefaultMpnnD64ModelName,
    kDefaultMpnn64ModelFamily,
    "archive-embedded-header-0348104439a78dae",
    kDefaultMpnn64HiddenDim,
    "archive-embedded-header-0348104439a78dae",
    generated::kSafetensorsBlobSha256,
};

const hikoboshi::universal::WeightsView kWeightsView{
    kMetadata,
    {kTensors.data(), kTensors.size()},
};

const hikoboshi::universal::WeightsHandle kWeightsHandle{
    &kPreparedWeights,
    &kWeightsView,
};

bool same_shape(const hikoboshi::universal::Span<const std::size_t>& manifest_shape,
                const generated::TensorBlobInfo& generated_tensor) noexcept {
  if (manifest_shape.size != generated_tensor.rank) {
    return false;
  }
  for (std::size_t index = 0; index < generated_tensor.rank; ++index) {
    if (manifest_shape.data[index] != generated_tensor.shape[index]) {
      return false;
    }
  }
  return true;
}

}  // namespace

const hikoboshi::universal::WeightsHandle& embedded_mpnn_d64_handle() noexcept {
  return kWeightsHandle;
}

bool validate_mpnn_d64_generated_tensors(
    hiko_u::Span<const generated::TensorBlobInfo> tensors,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = true;
  if (tensors.size != generated::kRuntimeTensorCount) {
    ok = false;
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
              "tensor_count_mismatch",
              "hikoboshi-mpnn-d64 runtime tensor table must contain exactly the required architecture tensors");
  }

  for (std::size_t index = 0; index < tensors.size; ++index) {
    const generated::TensorBlobInfo& info = tensors.data[index];
    if (ignored_historical_tensor_name(info.name)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "historical_tensor_in_runtime_weights",
                "historical gap tensors must remain excluded from runtime MPNN weights");
    } else if (!expected_tensor_name(info.name)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "unexpected_runtime_tensor",
                "hikoboshi-mpnn-d64 runtime tensor table contains an unexpected tensor name");
    }
  }

  for (const ExpectedTensor& expected : kExpectedTensors) {
    std::size_t match_count = 0;
    const generated::TensorBlobInfo* info =
        find_tensor(tensors, expected.name, match_count);
    if (match_count == 0 || info == nullptr) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "missing_required_tensor",
                "hikoboshi-mpnn-d64 runtime tensor table is missing a required architecture tensor");
      continue;
    }
    if (match_count > 1) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "duplicate_required_tensor",
                "hikoboshi-mpnn-d64 runtime tensor table contains a duplicate architecture tensor");
    }
    if (info->dtype != std::string_view{"float32"}) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_dtype_mismatch",
                "hikoboshi-mpnn-d64 runtime tensors must be float32");
    }
    if (info->rank != expected.rank) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_rank_mismatch",
                "hikoboshi-mpnn-d64 runtime tensor rank does not match the architecture slot");
    }
    if (!same_expected_shape(expected, *info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_shape_mismatch",
                "hikoboshi-mpnn-d64 runtime tensor shape does not match the architecture slot");
    }
    if (info->byte_length != expected_byte_length(expected)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_byte_length_mismatch",
                "hikoboshi-mpnn-d64 runtime tensor byte length does not match its float32 shape");
    }
    if (!row_major_strides(*info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_stride_mismatch",
                "hikoboshi-mpnn-d64 runtime tensors must use contiguous row-major strides");
    }
    if (!aligned_float_range(*info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_alignment_mismatch",
                "hikoboshi-mpnn-d64 runtime tensor byte range must be in-bounds and float-aligned");
    }
  }

  return ok;
}

bool embedded_mpnn_d64_manifest_matches() noexcept {
  const WeightManifestView& manifest = default_mpnn_d64_manifest();
  if (manifest.model_name != kDefaultMpnnD64ModelName ||
      manifest.model_family != kDefaultMpnn64ModelFamily ||
      manifest.hidden_dimension != kDefaultMpnn64HiddenDim ||
      manifest.neighbor_count != kDefaultMpnn64NeighborCount ||
      manifest.rbf_count != kDefaultMpnn64RbfCount ||
      manifest.layer_count != kDefaultMpnn64LayerCount ||
      manifest.tensor_schema != generated::kTensorSchema ||
      manifest.dtype != std::string_view{"float32"} ||
      manifest.checksum != generated::kSafetensorsBlobSha256 ||
      manifest.checksum_algorithm != std::string_view{"sha256"} ||
      manifest.source_checkpoint_checksum != generated::kSourceArtifactSha256 ||
      manifest.gap_parameter_family != kHardSwGapFamily ||
      manifest.gap_open != kHardSwDefaultGapOpen ||
      manifest.gap_extension != kHardSwDefaultGapExtension ||
      manifest.soft_gap_parameter_family != kSoftSwGapFamily ||
      manifest.soft_gap_open != kSoftSwMpnn64GapOpen ||
      manifest.soft_gap_extension != kSoftSwMpnn64GapExtension ||
      manifest.similarity != std::string_view{"raw_dot_product"} ||
      manifest.tensors.size != generated::kRuntimeTensorCount ||
      kWeightsView.tensors.size != generated::kRuntimeTensorCount) {
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

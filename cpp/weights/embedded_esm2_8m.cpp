#include "embedded_esm2_8m.hpp"

#include "generated/esm2_8m_blob.hpp"

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>
#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <array>
#include <cstddef>
#include <string_view>

namespace hikoboshi::weights::detail {
namespace {

namespace generated = ::hikoboshi::weights::generated::esm2_8m;
namespace hiko_u = ::hikoboshi::universal;

// Casey's compacted 29-token alphabet (see
// hikoboshi_train/esm2_encoder.py:24-44). Slots 0-19 are canonical amino
// acids in training order, slots 20-24 are the non-standard residues,
// slots 25-28 are PAD, CLS/BOS, EOS, MASK. The four FAIR-only ESM2
// tokens (<unk>, ., -, <null_1>) are dropped — the embedding table is
// 29 rows, not 33. Each string view references a string literal with
// static lifetime.
constexpr std::string_view kEsm2TokenTable[kDefaultEsm2_8mVocabSize] = {
    "A",     "R",     "N",     "D",     "C",     "Q",     "E",
    "G",     "H",     "I",     "L",     "K",     "M",     "F",
    "P",     "S",     "T",     "W",     "Y",     "V",     "B",
    "U",     "Z",     "O",     "X",     "<pad>", "<cls>", "<eos>",
    "<mask>",
};
static_assert(sizeof(kEsm2TokenTable) / sizeof(kEsm2TokenTable[0]) ==
                  kDefaultEsm2_8mVocabSize,
              "hikoboshi-esm2-8m token table size must match vocab size");

// Expected runtime tensors, alphabetically sorted to match the
// safetensors on-disk layout and the embedded blob's kRuntimeTensors[]
// ordering.
struct ExpectedTensor {
  std::string_view name;
  std::size_t rank;
  std::size_t shape[2];
};

#define HIKOBOSHI_ESM2_LINEAR(prefix, out_dim, in_dim)              \
  ExpectedTensor{prefix ".bias", 1, {(out_dim), 0}}, ExpectedTensor { \
    prefix ".weight", 2, { (out_dim), (in_dim) }                  \
  }

#define HIKOBOSHI_ESM2_NORM(prefix)                                          \
  ExpectedTensor{prefix ".bias", 1, {kDefaultEsm2_8mHiddenDim, 0}},        \
      ExpectedTensor {                                                     \
    prefix ".weight", 1, { kDefaultEsm2_8mHiddenDim, 0 }                   \
  }

#define HIKOBOSHI_ESM2_LAYER(index)                                              \
  HIKOBOSHI_ESM2_LINEAR("layers." #index ".attn.k_proj",                         \
                      kDefaultEsm2_8mHiddenDim, kDefaultEsm2_8mHiddenDim),     \
      HIKOBOSHI_ESM2_LINEAR("layers." #index ".attn.out_proj",                   \
                          kDefaultEsm2_8mHiddenDim, kDefaultEsm2_8mHiddenDim), \
      HIKOBOSHI_ESM2_NORM("layers." #index ".attn.pre_norm"),                    \
      HIKOBOSHI_ESM2_LINEAR("layers." #index ".attn.q_proj",                     \
                          kDefaultEsm2_8mHiddenDim, kDefaultEsm2_8mHiddenDim), \
      HIKOBOSHI_ESM2_LINEAR("layers." #index ".attn.v_proj",                     \
                          kDefaultEsm2_8mHiddenDim, kDefaultEsm2_8mHiddenDim), \
      HIKOBOSHI_ESM2_LINEAR("layers." #index ".ffn.in",                          \
                          kDefaultEsm2_8mFfnHiddenDim,                         \
                          kDefaultEsm2_8mHiddenDim),                           \
      HIKOBOSHI_ESM2_LINEAR("layers." #index ".ffn.out",                         \
                          kDefaultEsm2_8mHiddenDim,                            \
                          kDefaultEsm2_8mFfnHiddenDim),                        \
      HIKOBOSHI_ESM2_NORM("layers." #index ".ffn.pre_norm")

constexpr ExpectedTensor kExpectedTensors[] = {
    ExpectedTensor{"embedding_table",
                   2,
                   {kDefaultEsm2_8mVocabSize, kDefaultEsm2_8mHiddenDim}},
    HIKOBOSHI_ESM2_NORM("final_norm"),
    HIKOBOSHI_ESM2_LAYER(0),
    HIKOBOSHI_ESM2_LAYER(1),
    HIKOBOSHI_ESM2_LAYER(2),
    HIKOBOSHI_ESM2_LAYER(3),
    HIKOBOSHI_ESM2_LAYER(4),
    HIKOBOSHI_ESM2_LAYER(5),
};

#undef HIKOBOSHI_ESM2_LAYER
#undef HIKOBOSHI_ESM2_NORM
#undef HIKOBOSHI_ESM2_LINEAR

static_assert(sizeof(kExpectedTensors) / sizeof(kExpectedTensors[0]) ==
                  generated::kRuntimeTensorCount,
              "hikoboshi-esm2-8m expected tensor table must match runtime count");

// Manifest tensor table (alphabetical, matches generated layout).
const std::array<TensorManifestView, generated::kRuntimeTensorCount>&
manifest_tensor_entries() noexcept {
  static const auto kEntries = []() {
    std::array<TensorManifestView, generated::kRuntimeTensorCount> entries{};
    for (std::size_t index = 0; index < generated::kRuntimeTensorCount;
         ++index) {
      const generated::TensorBlobInfo& info = generated::kRuntimeTensors[index];
      entries[index] = TensorManifestView{
          info.name,
          {info.shape, info.rank},
          info.dtype,
          info.checksum,
      };
    }
    return entries;
  }();
  return kEntries;
}

// Public WeightManifestView for the embedded hikoboshi-esm2-8m package.
//
// MPNN-specific fields (neighbor_count, rbf_count, rbf_feature_order,
// message_scale) are intentionally zero/empty for ESM2; the schema is
// generic enough that future packages with extra slot families fill
// the matching fields. Soft-SW T=1 defaults reflect the joint MLM + alignment
// checkpoint, while hard-SW defaults reflect the separately recorded
// near-zero-temperature gap anneal. Per-tensor SHA-256 records live in the
// `tensors` span.
constexpr std::string_view kEsm2DefaultModelVersion{
    "afa959cb694cfbe7"};
constexpr std::string_view kEsm2DefaultSourceCheckpoint{
    "megure-labs-internal://checkpoints/hikoboshi-esm2-8m/final_model.pt"};
constexpr std::string_view kEsm2DefaultGenerationTool{
    "convert_esm2_8m_to_safetensors.py (Megure Labs internal)"};
constexpr std::string_view kEsm2DefaultGenerationToolVersion{
    "esm2-8m-weights-package-20260514"};
constexpr std::string_view kEsm2DefaultGenerationDate{"2026-05-14"};
constexpr std::string_view kEsm2DefaultValidationStatus{
    "checksum-verified; per-tensor SHA-256 recorded; hard-SW defaults "
    "preserved; soft-SW descriptor metadata recovered"};
constexpr std::string_view kEsm2DefaultProvenanceStatus{
    "real safetensors blob embedded; historical gap tensors remain excluded "
    "from runtime weights"};

const WeightManifestView& esm2_manifest_storage() noexcept {
  static const WeightManifestView kManifest{
      std::string_view{"0.1.0"},
      kDefaultEsm2_8mModelName,
      kDefaultEsm2_8mModelFamily,
      kEsm2DefaultModelVersion,
      kDefaultEsm2_8mHiddenDim,
      0,
      0,
      std::string_view{},
      kDefaultEsm2_8mAttentionLayerCount,
      generated::kTensorSchema,
      0.0F,
      std::string_view{"float32"},
      {manifest_tensor_entries().data(), manifest_tensor_entries().size()},
      generated::kSafetensorsBlobSha256,
      std::string_view{"sha256"},
      kEsm2DefaultSourceCheckpoint,
      generated::kSourceArtifactSha256,
      kEsm2DefaultGenerationTool,
      kEsm2DefaultGenerationToolVersion,
      kEsm2DefaultGenerationDate,
      kHardSwGapFamily,
      kHardSwEsm2_8mGapOpen,
      kHardSwEsm2_8mGapExtension,
      kSoftSwGapFamily,
      kSoftSwEsm2_8mGapOpen,
      kSoftSwEsm2_8mGapExtension,
      std::string_view{"raw_dot_product"},
      kEsm2DefaultValidationStatus,
      kEsm2DefaultProvenanceStatus,
  };
  return kManifest;
}

// Typed weight bindings for the modules-layer ESM2 forward pass are
// constructed in the algorithms/engine layer, not here: the weights
// dependency-graph layer must not include `hikoboshi/modules/...`. The
// engine pulls the typed view together from the embedded safetensors
// blob and the descriptor at prepared-state-build time. This TU
// surfaces the raw payload pointer (as the `WeightsHandle.opaque`)
// plus the per-tensor `WeightsView` so consumers downstream can
// resolve individual tensor spans without re-parsing the safetensors
// header.

const std::array<hiko_u::TensorView, generated::kRuntimeTensorCount>&
tensor_views() noexcept {
  static const auto kViews = []() {
    std::array<hiko_u::TensorView, generated::kRuntimeTensorCount> views{};
    for (std::size_t index = 0; index < generated::kRuntimeTensorCount;
         ++index) {
      const generated::TensorBlobInfo& info = generated::kRuntimeTensors[index];
      views[index] = hiko_u::TensorView{
          generated::kSafetensorsBlob + generated::kSafetensorsDataOffset +
              info.data_offset,
          {info.shape, info.rank},
          {info.strides, info.rank},
          hiko_u::DataType::Float32,
          info.name,
      };
    }
    return views;
  }();
  return kViews;
}

const hiko_u::ModelMetadataView& esm2_metadata() noexcept {
  static const hiko_u::ModelMetadataView kMetadata{
      kDefaultEsm2_8mModelName,
      kDefaultEsm2_8mModelFamily,
      kEsm2DefaultModelVersion,
      kDefaultEsm2_8mHiddenDim,
      generated::kSourceArtifactSha256,
      generated::kSafetensorsBlobSha256,
  };
  return kMetadata;
}

const hiko_u::WeightsView& esm2_weights_view() noexcept {
  static const hiko_u::WeightsView kView{
      esm2_metadata(),
      {tensor_views().data(), tensor_views().size()},
  };
  return kView;
}

const hiko_u::WeightsHandle& esm2_weights_handle() noexcept {
  static const hiko_u::WeightsHandle kHandle{
      // The opaque is the raw safetensors payload. The algorithms-layer
      // ESM2 prepared-state builder (see `cpp/algorithms/` once the
      // sequence-input pairwise path lands) parses the embedded blob
      // into a typed `modules::detail::Esm2Weights` view; the weights
      // layer cannot construct that typed view itself because
      // `hikoboshi/modules/*` is below the include-graph edge that
      // `weights` may consume.
      static_cast<const void*>(generated::kSafetensorsBlob),
      &esm2_weights_view(),
  };
  return kHandle;
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

bool expected_tensor_name(const std::string_view name) noexcept {
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
    const hiko_u::Span<const generated::TensorBlobInfo> tensors,
    const std::string_view name, std::size_t& match_count) noexcept {
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

const hiko_u::WeightsHandle& embedded_esm2_8m_handle() noexcept {
  return esm2_weights_handle();
}

hiko_u::Span<const std::string_view> embedded_esm2_8m_tokenizer_table() noexcept {
  return {kEsm2TokenTable,
          sizeof(kEsm2TokenTable) / sizeof(kEsm2TokenTable[0])};
}

bool validate_esm2_8m_generated_tensors(
    hiko_u::Span<const generated::TensorBlobInfo> tensors,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept {
  bool ok = true;
  if (tensors.size != generated::kRuntimeTensorCount) {
    ok = false;
    add_error(buffer, diagnostic_count,
              hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
              "tensor_count_mismatch",
              "hikoboshi-esm2-8m runtime tensor table must contain exactly the "
              "required architecture tensors");
  }

  for (std::size_t index = 0; index < tensors.size; ++index) {
    const generated::TensorBlobInfo& info = tensors.data[index];
    if (!expected_tensor_name(info.name)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "unexpected_runtime_tensor",
                "hikoboshi-esm2-8m runtime tensor table contains an unexpected "
                "tensor name");
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
                "hikoboshi-esm2-8m runtime tensor table is missing a required "
                "architecture tensor");
      continue;
    }
    if (match_count > 1) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "duplicate_required_tensor",
                "hikoboshi-esm2-8m runtime tensor table contains a duplicate "
                "architecture tensor");
    }
    if (info->dtype != std::string_view{"float32"}) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_dtype_mismatch",
                "hikoboshi-esm2-8m runtime tensors must be float32");
    }
    if (info->rank != expected.rank) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_rank_mismatch",
                "hikoboshi-esm2-8m runtime tensor rank does not match the "
                "architecture slot");
    }
    if (!same_expected_shape(expected, *info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_shape_mismatch",
                "hikoboshi-esm2-8m runtime tensor shape does not match the "
                "architecture slot");
    }
    if (info->byte_length != expected_byte_length(expected)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_byte_length_mismatch",
                "hikoboshi-esm2-8m runtime tensor byte length does not match "
                "its float32 shape");
    }
    if (!row_major_strides(*info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_stride_mismatch",
                "hikoboshi-esm2-8m runtime tensors must use contiguous "
                "row-major strides");
    }
    if (!aligned_float_range(*info)) {
      ok = false;
      add_error(buffer, diagnostic_count,
                hiko_u::PackageValidationStage::TensorTableRolesShapesDtypes,
                "tensor_alignment_mismatch",
                "hikoboshi-esm2-8m runtime tensor byte range must be in-bounds "
                "and float-aligned");
    }
  }
  return ok;
}

bool embedded_esm2_8m_manifest_matches() noexcept {
  const WeightManifestView& manifest = default_esm2_8m_manifest();
  if (manifest.model_name != kDefaultEsm2_8mModelName ||
      manifest.model_family != kDefaultEsm2_8mModelFamily ||
      manifest.hidden_dimension != kDefaultEsm2_8mHiddenDim ||
      manifest.layer_count != kDefaultEsm2_8mAttentionLayerCount ||
      manifest.tensor_schema != generated::kTensorSchema ||
      manifest.dtype != std::string_view{"float32"} ||
      manifest.checksum != generated::kSafetensorsBlobSha256 ||
      manifest.checksum_algorithm != std::string_view{"sha256"} ||
      manifest.source_checkpoint_checksum !=
          generated::kSourceArtifactSha256 ||
      manifest.gap_parameter_family != kHardSwGapFamily ||
      manifest.gap_open != kHardSwEsm2_8mGapOpen ||
      manifest.gap_extension != kHardSwEsm2_8mGapExtension ||
      manifest.soft_gap_parameter_family != kSoftSwGapFamily ||
      manifest.soft_gap_open != kSoftSwEsm2_8mGapOpen ||
      manifest.soft_gap_extension != kSoftSwEsm2_8mGapExtension ||
      manifest.similarity != std::string_view{"raw_dot_product"} ||
      manifest.tensors.size != generated::kRuntimeTensorCount ||
      esm2_weights_view().tensors.size != generated::kRuntimeTensorCount) {
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

namespace hikoboshi::weights {

const WeightManifestView& default_esm2_8m_manifest() noexcept {
  return ::hikoboshi::weights::detail::esm2_manifest_storage();
}

}  // namespace hikoboshi::weights

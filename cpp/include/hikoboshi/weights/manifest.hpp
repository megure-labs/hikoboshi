#ifndef HIKOBOSHI_WEIGHTS_MANIFEST_HPP
#define HIKOBOSHI_WEIGHTS_MANIFEST_HPP

/// @file
/// Public manifest view for the compiled hikoboshi-mpnn-d64 package.

#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::weights {

/// Sentinel used when a manifest/provenance field is intentionally unavailable.
inline constexpr std::string_view kPendingProvenanceSentinel{"pending"};
/// Canonical package id for the compiled hikoboshi-mpnn-d64 package.
inline constexpr std::string_view kDefaultMpnnD64ModelName{"hikoboshi-mpnn-d64"};
/// Backward-compatible C++ symbol for callers compiled against the old name.
inline constexpr std::string_view kDefaultMpnn64ModelName{
    kDefaultMpnnD64ModelName};
/// User-facing model family for the compiled default package.
inline constexpr std::string_view kDefaultMpnn64ModelFamily{"Hikoboshi-MPNN"};
/// Registered architecture id implemented by the default compiled package.
inline constexpr std::string_view kDefaultMpnn64ArchitectureId{
    "hikoboshi_mpnn_v1"};
/// Default package embedding dimension.
inline constexpr std::size_t kDefaultMpnn64HiddenDim = 64;
/// K-nearest-neighbor count used by hikoboshi-mpnn-d64 preprocessing.
inline constexpr std::size_t kDefaultMpnn64NeighborCount = 64;
/// Radial-basis feature count used in edge construction.
inline constexpr std::size_t kDefaultMpnn64RbfCount = 16;
/// Message-passing layer count in hikoboshi-mpnn-d64.
inline constexpr std::size_t kDefaultMpnn64LayerCount = 3;
/// Training-time message scale used by the compiled architecture
/// (PyTorch `EncLayer.scale` source default).
inline constexpr float kDefaultMpnn64MessageScale = 30.0F;
/// Manifest label for the edge radial-basis feature order.
inline constexpr std::string_view kDefaultMpnn64RbfFeatureOrder{
    "atom_pair_distance_rbf"};
/// Tensor schema identifier reported by the manifest.
inline constexpr std::string_view kDefaultMpnn64TensorSchema{"pending"};
/// Hard local affine SW family name preserved as the Hikoboshi 0.1.0 default.
inline constexpr std::string_view kHardSwGapFamily{"Hikoboshi 0.1.0 hard-SW"};
/// Soft local affine SW benchmark companion family name.
inline constexpr std::string_view kSoftSwGapFamily{"Hikoboshi 0.1.0 soft-SW"};
/// Hard local affine SW gap opening default calibrated for raw-dot scores.
inline constexpr float kHardSwDefaultGapOpen = -1.40000F;
/// Hard local affine SW gap extension default calibrated for raw-dot scores.
inline constexpr float kHardSwDefaultGapExtension = -0.150000F;
/// Soft local affine SW gap opening default calibrated for hikoboshi-mpnn-d64.
inline constexpr float kSoftSwMpnn64GapOpen = -3.21337F;
/// Soft local affine SW gap extension default calibrated for hikoboshi-mpnn-d64.
inline constexpr float kSoftSwMpnn64GapExtension = -0.111704F;

/// Canonical package id for the compiled hikoboshi-esm2-8m package.
inline constexpr std::string_view kDefaultEsm2_8mModelName{"hikoboshi-esm2-8m"};
/// User-facing model family for the compiled ESM2-8M package.
inline constexpr std::string_view kDefaultEsm2_8mModelFamily{"Hikoboshi-ESM2"};
/// Registered architecture id implemented by the ESM2-8M package.
inline constexpr std::string_view kDefaultEsm2_8mArchitectureId{
    "hikoboshi_esm2_v1"};
/// ESM2-8M embedding dimension reported by the descriptor.
inline constexpr std::size_t kDefaultEsm2_8mHiddenDim = 320;
/// ESM2-8M transformer attention layer count.
inline constexpr std::size_t kDefaultEsm2_8mAttentionLayerCount = 6;
/// ESM2-8M transformer attention head count.
inline constexpr std::size_t kDefaultEsm2_8mAttentionHeadCount = 20;
/// ESM2-8M feedforward intermediate dimension.
inline constexpr std::size_t kDefaultEsm2_8mFfnHiddenDim = 1280;
/// ESM2-8M token vocabulary size for Casey's compacted local alphabet
/// (20 canonical AAs + 5 non-standard AAs + PAD/CLS/EOS/MASK specials).
inline constexpr std::size_t kDefaultEsm2_8mVocabSize = 29;
/// ESM2-8M maximum supported sequence length.
inline constexpr std::size_t kDefaultEsm2_8mMaxSequenceLength = 65536;
/// Hard local affine SW gap opening default calibrated for hikoboshi-esm2-8m.
inline constexpr float kHardSwEsm2_8mGapOpen = -1.01982F;
/// Hard local affine SW gap extension default calibrated for hikoboshi-esm2-8m.
inline constexpr float kHardSwEsm2_8mGapExtension = +0.225736F;
/// Soft local affine SW gap opening default recovered from the ESM2 checkpoint.
inline constexpr float kSoftSwEsm2_8mGapOpen = -6.72805F;
/// Soft local affine SW gap extension default recovered from the ESM2 checkpoint.
inline constexpr float kSoftSwEsm2_8mGapExtension = -0.0159468F;

/// Canonical package id for the embedded vanilla ProteinMPNN v_48_020 package.
inline constexpr std::string_view kDefaultProteinMpnnV48Eps020ModelName{
    "proteinmpnn-v48-eps020"};
/// Backward-compatible C++ symbol for callers compiled against the old name.
inline constexpr std::string_view kDefaultProteinMpnnV48020ModelName{
    kDefaultProteinMpnnV48Eps020ModelName};
/// User-facing model family for the embedded ProteinMPNN inverse-folding package.
inline constexpr std::string_view kDefaultProteinMpnnV48020ModelFamily{
    "ProteinMPNN"};
/// Internal architecture id reserved for the ProteinMPNN inverse-folding path.
inline constexpr std::string_view kDefaultProteinMpnnV48Eps020ArchitectureId{
    "proteinmpnn_v48_eps020"};
/// Backward-compatible C++ symbol for callers compiled against the old name.
inline constexpr std::string_view kDefaultProteinMpnnV48020ArchitectureId{
    kDefaultProteinMpnnV48Eps020ArchitectureId};
/// ProteinMPNN v_48_020 hidden dimension.
inline constexpr std::size_t kDefaultProteinMpnnV48020HiddenDim = 128;
/// ProteinMPNN v_48_020 K-nearest-neighbor count.
inline constexpr std::size_t kDefaultProteinMpnnV48020NeighborCount = 48;
/// ProteinMPNN v_48_020 radial-basis feature count.
inline constexpr std::size_t kDefaultProteinMpnnV48020RbfCount = 16;
/// ProteinMPNN v_48_020 encoder and decoder layer count.
inline constexpr std::size_t kDefaultProteinMpnnV48020LayerCount = 3;
/// ProteinMPNN v_48_020 message scale from the reference layers.
inline constexpr float kDefaultProteinMpnnV48020MessageScale = 30.0F;
/// Sentinel family for schema-required gap fields on inverse-folding packages.
inline constexpr std::string_view kInverseFoldingGapFamily{
    "n/a inverse-folding"};
/// Sentinel score/similarity field for inverse-folding packages.
inline constexpr std::string_view kInverseFoldingSimilarity{
    "n/a inverse-folding"};

/// Manifest row for one tensor in the compiled package.
struct TensorManifestView {
  std::string_view name;
  universal::Span<const std::size_t> shape;
  std::string_view dtype;
  std::string_view checksum;
};

/// Public read-only view of a compiled Hikoboshi weight manifest.
///
/// The manifest records package identity, architecture dimensions, tensor
/// inventory, checksums, generation metadata, scoring method, hard-SW gap
/// defaults, and soft-SW benchmark gap defaults. All strings and tensor spans
/// are owned by the weights library and remain valid for the lifetime of the
/// process.
struct WeightManifestView {
  std::string_view schema_version;
  std::string_view model_name;
  std::string_view model_family;
  std::string_view model_version;
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
  std::size_t rbf_count;
  std::string_view rbf_feature_order;
  std::size_t layer_count;
  std::string_view tensor_schema;
  float message_scale;
  std::string_view dtype;
  universal::Span<const TensorManifestView> tensors;
  std::string_view checksum;
  std::string_view checksum_algorithm;
  std::string_view source_checkpoint;
  std::string_view source_checkpoint_checksum;
  std::string_view generation_tool;
  std::string_view generation_tool_version;
  std::string_view generation_date;
  std::string_view gap_parameter_family;
  float gap_open;
  float gap_extension;
  std::string_view soft_gap_parameter_family;
  float soft_gap_open;
  float soft_gap_extension;
  std::string_view similarity;
  std::string_view validation_status;
  std::string_view provenance_status;
};

/// Return the manifest for the compiled hikoboshi-mpnn-d64 package.
const WeightManifestView& default_mpnn_d64_manifest() noexcept;

/// Backward-compatible manifest accessor for the compiled hikoboshi-mpnn-d64 package.
const WeightManifestView& default_mpnn64_manifest() noexcept;

/// Return the manifest for the compiled hikoboshi-esm2-8m package.
///
/// Most numeric and checksum fields carry `kPendingProvenanceSentinel`
/// until the `esm2-8m-weights-package` packet calibrates them against a
/// concrete safetensors blob. Architecture identity, hidden dimension,
/// scoring family, and alignment algorithm strings are stable.
const WeightManifestView& default_esm2_8m_manifest() noexcept;

/// Return the manifest for the embedded vanilla ProteinMPNN v_48_020 package.
///
/// ProteinMPNN is an inverse-folding package, not an aligner. Its gap fields
/// are explicit non-aligner sentinels retained only because the current
/// manifest view shape carries package-wide gap metadata.
const WeightManifestView& default_proteinmpnn_v48_eps020_manifest() noexcept;

/// Backward-compatible manifest accessor for proteinmpnn-v48-eps020.
const WeightManifestView& default_proteinmpnn_v48_020_manifest() noexcept;

}  // namespace hikoboshi::weights

#endif  // HIKOBOSHI_WEIGHTS_MANIFEST_HPP

#ifndef HIKOBOSHI_API_REQUESTS_HPP
#define HIKOBOSHI_API_REQUESTS_HPP

/// @file
/// Public request records for encode, pairwise, all-vs-all, and pair-list
/// workflows.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::api {

/// Default hard local affine SW gap opening penalty for Hikoboshi 0.1.0.
inline constexpr float kDefaultGapOpen = -1.40000F;
/// Default hard local affine SW gap extension penalty for Hikoboshi 0.1.0.
inline constexpr float kDefaultGapExtension = -0.150000F;

/// Sentinel that signals "use the resolved package descriptor's calibrated
/// affine gap value" when carried through an `AlignmentOptions` field. The
/// engine substitutes this sentinel with the package descriptor's gap on
/// entry to the sequence routes so per-package calibrations such as
/// hikoboshi-esm2-8m's hard `-1.01982 / +0.225736` pair propagate without each
/// adapter having to know them.
inline constexpr float kPackageDefaultGapSentinel =
    std::numeric_limits<float>::quiet_NaN();

/// True iff `value` is the package-default sentinel (NaN).
inline bool is_package_default_gap(float value) noexcept {
  return std::isnan(value);
}

/// Default soft Smith-Waterman temperature for Soft and Both modes.
///
/// `1.0` is the empirically-best F1 from the training run and the safe
/// plateau choice for Hikoboshi 0.1.0; see packet a1c notes for the F1-vs-T
/// rationale. May be re-tuned in a follow-up packet (a1e).
inline constexpr float kDefaultSoftTemperature = 1.0F;

/// Selects the alignment branch for pairwise requests.
///
/// `Hard` runs the hard local affine Smith-Waterman path and is the
/// default in Hikoboshi 0.1.0; it is bit-equal to the pre-soft-SW behavior.
/// `Soft` runs the soft Smith-Waterman partition-function path at the
/// requested `temperature` and is available as an explicit non-default
/// opt-in for callers that prioritize quality over wall-clock. `Both`
/// runs both branches and reports the hard alignment/path as primary while
/// also reporting the soft score. Soft and both modes are roughly 6-10x
/// slower than hard mode at the same problem size.
enum class AlignmentMode {
  Soft,
  Hard,
  Both,
};

/// Options for embedding generation requests.
struct EncodeOptions {};

/// Alignment controls shared by pairwise and all-vs-all requests.
///
/// By default the engine resolves these fields from the selected package
/// descriptor. Hard-SW uses the package's `gaps` family and soft-SW uses its
/// `soft_gaps` family; a finite caller-supplied value overrides the matching
/// field for every branch selected by `AlignmentMode`.
struct AlignmentOptions {
  float gap_open = kPackageDefaultGapSentinel;
  float gap_extension = kPackageDefaultGapSentinel;
};

/// Borrowed canonical coordinate input.
///
/// `coordinates` is row-major `[residue_count, 5, 3]`; `atom_sources` is
/// row-major `[residue_count, 5]`. Residue codes and metadata are optional but
/// required for sequence-aware outputs and metrics.
struct CoordsInputView {
  std::size_t residue_count = 0;
  universal::Span<const float> coordinates{};
  universal::Span<const universal::AtomSource> atom_sources{};
  universal::Span<const char> residue_codes{};
  universal::Span<const universal::ResidueMetadataView> residues{};
};

/// Default compiled inverse-folding package.
inline constexpr std::string_view kDefaultInverseFoldPackage =
    "proteinmpnn-v48-eps020";

/// ProteinMPNN decode-order policy for inverse folding.
enum class InverseFoldDecodeOrder {
  Random,
  NToC,
};

/// Inverse-fold one normalized backbone into one or more designed sequences.
///
/// `structure` is the preferred public input. `coords` carries the same
/// canonical `[L, 5, 3]` backbone shape for callers that already hold parsed
/// coordinates. `pdb_path` is adapter provenance only; the API engine remains
/// in-memory and does not load files.
struct InverseFoldRequest {
  universal::StructureView structure{};
  CoordsInputView coords{};
  std::string pdb_path{};
  std::string package = "proteinmpnn-v48-eps020";
  float sampling_temp = 0.1F;
  std::size_t num_seqs = 1;
  std::uint64_t seed = 0;
  InverseFoldDecodeOrder decode_order = InverseFoldDecodeOrder::Random;
  float backbone_noise = 0.0F;
  /// Non-empty means callers want the per-position log-prob artifact. The API
  /// returns the artifact payload; adapters that own file IO write this path.
  std::string logprobs_out{};
};

/// Encode one normalized structure.
struct EncodeStructureRequest {
  universal::StructureView structure;
  EncodeOptions options{};
};

/// Encode one canonical coordinate view.
struct EncodeCoordsRequest {
  CoordsInputView coords;
  EncodeOptions options{};
};

/// Pairwise alignment request for normalized structures.
///
/// `mode` defaults to `AlignmentMode::Hard` (hard local affine SW) for
/// Hikoboshi 0.1.0; pass `AlignmentMode::Soft` to opt in to the
/// partition-function soft SW path at `temperature`, or
/// `AlignmentMode::Both` to run hard and soft in one request. `temperature`
/// is ignored when `mode == AlignmentMode::Hard`.
struct PairwiseStructureRequest {
  universal::StructureView query;
  universal::StructureView target;
  AlignmentOptions alignment{};
  AlignmentMode mode = AlignmentMode::Hard;
  float temperature = kDefaultSoftTemperature;
};

/// Pairwise alignment request for canonical coordinate views.
///
/// See `PairwiseStructureRequest` for `mode` / `temperature` semantics.
struct PairwiseCoordsRequest {
  CoordsInputView query;
  CoordsInputView target;
  AlignmentOptions alignment{};
  AlignmentMode mode = AlignmentMode::Hard;
  float temperature = kDefaultSoftTemperature;
};

/// Pairwise alignment request for precomputed residue embeddings.
///
/// This route skips package encoding but still runs raw-dot scoring through
/// the SW branch or branches selected by `mode`. It is not a public score-only
/// API. See `PairwiseStructureRequest` for `mode` / `temperature` semantics.
struct PairwiseEmbeddingRequest {
  universal::EmbeddingView query;
  universal::EmbeddingView target;
  AlignmentOptions alignment{};
  AlignmentMode mode = AlignmentMode::Hard;
  float temperature = kDefaultSoftTemperature;
};

/// All-vs-all enumeration options.
///
/// By default records are emitted for `i < j`. Set `include_self` to emit
/// `i <= j` while preserving lexicographic order. `mode` defaults to
/// `AlignmentMode::Hard` so existing all-vs-all callers see bit-identical
/// hard-SW output; pass `AlignmentMode::Soft` (with a positive
/// `temperature`) to opt in to soft Smith-Waterman, or `AlignmentMode::Both`
/// to run both branches and report both scores. `temperature` is ignored when
/// `mode == AlignmentMode::Hard`.
struct AllVsAllOptions {
  bool include_self = false;
  AlignmentOptions alignment{};
  AlignmentMode mode = AlignmentMode::Hard;
  float temperature = kDefaultSoftTemperature;
};

/// All-vs-all request over normalized structures.
struct AllVsAllStructureRequest {
  universal::Span<const universal::StructureView> structures{};
  AllVsAllOptions options{};
};

/// All-vs-all request over canonical coordinate views.
struct AllVsAllCoordsRequest {
  universal::Span<const CoordsInputView> coords{};
  AllVsAllOptions options{};
};

/// All-vs-all request over precomputed embeddings.
struct AllVsAllEmbeddingRequest {
  universal::Span<const universal::EmbeddingView> embeddings{};
  AllVsAllOptions options{};
};

/// Pair-list alignment runs a caller-supplied list of named (query, target)
/// pairs — the third alignment driver mode alongside pairwise and all-vs-all.
/// See `docs/charters/PAIR_LIST_CHARTER.md`.
///
/// Each `PairList*Request` mirrors the matching `AllVsAll*Request` source
/// input and adds a `pairs` field. `pairs` is the caller-supplied list of
/// `(query_id, target_id)` protein IDs; output records are emitted
/// one-per-pair in this exact input order (record index == input pair
/// index), and a duplicate input pair produces its own record. IDs are
/// case-sensitive and never normalized; a pair referencing an ID absent
/// from the route's source set is a fail-fast error, never a silent skip.
///
/// `AllVsAllOptions` is reused unchanged: pair-list needs the same alignment
/// gap parameters, SW `mode`, and `temperature`, and no genuinely new
/// option. Its `include_self` field is inert for pair-list — the caller
/// supplies the exact pair set, so the mode never generates an unlisted
/// pair.

/// Pair-list request over normalized structures.
struct PairListStructureRequest {
  universal::Span<const universal::StructureView> structures{};
  std::vector<std::pair<std::string, std::string>> pairs{};
  AllVsAllOptions options{};
};

/// Pair-list request over canonical coordinate views.
struct PairListCoordsRequest {
  universal::Span<const CoordsInputView> coords{};
  std::vector<std::pair<std::string, std::string>> pairs{};
  AllVsAllOptions options{};
};

/// Pair-list request over precomputed embeddings.
struct PairListEmbeddingRequest {
  universal::Span<const universal::EmbeddingView> embeddings{};
  std::vector<std::pair<std::string, std::string>> pairs{};
  AllVsAllOptions options{};
};

}  // namespace hikoboshi::api

#endif  // HIKOBOSHI_API_REQUESTS_HPP

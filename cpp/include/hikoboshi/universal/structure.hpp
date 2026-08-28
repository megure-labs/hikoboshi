#ifndef HIKOBOSHI_UNIVERSAL_STRUCTURE_HPP
#define HIKOBOSHI_UNIVERSAL_STRUCTURE_HPP

/// @file
/// Borrowed normalized structure views and residue metadata.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::universal {

/// Canonical Hikoboshi atom order for normalized backbone coordinates.
inline constexpr std::size_t kCanonicalAtomCount = 5;
/// Coordinate axis count for xyz triples.
inline constexpr std::size_t kCoordinateAxisCount = 3;

/// Canonical atom slots in a normalized residue coordinate row.
enum class CanonicalAtom : std::uint8_t {
  N = 0,
  CA = 1,
  C = 2,
  O = 3,
  CB = 4,
};

/// Provenance for each canonical atom coordinate.
///
/// Missing atoms are masked by this enum; zero coordinates are not semantic
/// missing-data markers. Glycine virtual CB atoms are marked `Virtual`.
enum class AtomSource : std::uint8_t {
  Missing = 0,
  Observed = 1,
  Inferred = 2,
  Virtual = 3,
};

/// Chain-break marker after a residue in the normalized sequence.
struct ChainBreakView {
  std::size_t after_residue_index = 0;
};

/// Borrowed residue identity and source-location metadata.
///
/// Residue codes use normalized one-letter amino-acid codes. Original names,
/// chain/model identifiers, numbering, insertion codes, and source positions
/// are retained when available so outputs can preserve biological context.
struct ResidueMetadataView {
  char residue_code;
  std::string_view original_residue_name;
  std::string_view chain_id;
  std::string_view model_id;
  std::int32_t model_index;
  std::int32_t residue_number;
  char insertion_code;
  std::string_view source_id;
  std::int64_t source_residue_index;
  std::string_view source_filename{};
  std::int64_t source_record_index = -1;
};

/// Borrowed normalized single-chain structure view.
///
/// Coordinates are row-major `[residue_count, kCanonicalAtomCount, 3]`.
/// `atom_sources` has `[residue_count, kCanonicalAtomCount]` entries and
/// determines which coordinates are observed, inferred, virtual, or missing.
/// Hikoboshi 0.1.0 aligns one selected polymer chain per input.
struct StructureView {
  std::size_t residue_count;
  Span<const float> coordinates;
  Span<const AtomSource> atom_sources;
  Span<const char> residue_codes;
  Span<const ResidueMetadataView> residues;
  std::string_view input_id{};
  std::string_view source_filename{};
  Span<const ChainBreakView> chain_breaks{};
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_STRUCTURE_HPP

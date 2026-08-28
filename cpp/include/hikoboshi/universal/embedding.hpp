#ifndef HIKOBOSHI_UNIVERSAL_EMBEDDING_HPP
#define HIKOBOSHI_UNIVERSAL_EMBEDDING_HPP

/// @file
/// Borrowed residue embedding views for public API requests.

#include <cstddef>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::universal {

/// Borrowed residue embedding matrix and optional residue metadata.
///
/// `values` is row-major `[residue_count, dimension]` float32 data. When
/// present, `residue_codes` and `residues` have `residue_count` entries and
/// carry the sequence/structure metadata needed for FASTA rendering and
/// metadata-dependent metrics. The caller owns all referenced storage.
struct EmbeddingView {
  std::size_t residue_count;
  std::size_t dimension;
  Span<const float> values;
  Span<const char> residue_codes;
  Span<const ResidueMetadataView> residues;
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_EMBEDDING_HPP

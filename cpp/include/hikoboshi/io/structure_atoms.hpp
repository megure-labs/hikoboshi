#ifndef HIKOBOSHI_IO_STRUCTURE_ATOMS_HPP
#define HIKOBOSHI_IO_STRUCTURE_ATOMS_HPP

#include <hikoboshi/io/structure_loader.hpp>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

struct CanonicalAtomPackRequest {
  universal::Span<const detail::NormalizedResidue> residues{};
};

struct CanonicalAtomPackOutput {
  universal::Span<float> coordinates{};  // row-major [L, 5, 3]
  universal::Span<universal::AtomSource> atom_sources{};  // row-major [L, 5]
};

[[nodiscard]] universal::Status atom_source_extract(
    universal::Span<const detail::NormalizedResidue> residues,
    universal::Span<universal::AtomSource> atom_sources) noexcept;

[[nodiscard]] universal::Status canonical_atom_pack(
    const CanonicalAtomPackRequest& request,
    const CanonicalAtomPackOutput& output) noexcept;

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_STRUCTURE_ATOMS_HPP

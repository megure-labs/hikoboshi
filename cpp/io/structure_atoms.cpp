#include <hikoboshi/io/structure_atoms.hpp>

#include <algorithm>
#include <cstddef>

#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {
namespace {

constexpr std::size_t kAtomCount = universal::kCanonicalAtomCount;
constexpr std::size_t kAxisCount = universal::kCoordinateAxisCount;

std::size_t coord_count(std::size_t residue_count) noexcept {
  return residue_count * kAtomCount * kAxisCount;
}

std::size_t source_count(std::size_t residue_count) noexcept {
  return residue_count * kAtomCount;
}

bool has_coordinate_capacity(universal::Span<float> values,
                             std::size_t residue_count) noexcept {
  return values.data != nullptr && values.size >= coord_count(residue_count);
}

bool has_source_capacity(universal::Span<universal::AtomSource> values,
                         std::size_t residue_count) noexcept {
  return values.data != nullptr && values.size >= source_count(residue_count);
}

}  // namespace

universal::Status atom_source_extract(
    universal::Span<const detail::NormalizedResidue> residues,
    universal::Span<universal::AtomSource> atom_sources) noexcept {
  if (residues.size == 0) {
    return universal::ok_status();
  }
  if (residues.data == nullptr || !has_source_capacity(atom_sources, residues.size)) {
    return universal::invalid_argument_status(
        "atom_source_extract requires residue records and [L,5] output");
  }
  for (std::size_t residue = 0; residue < residues.size; ++residue) {
    const detail::NormalizedResidue& slot = residues.data[residue];
    for (std::size_t atom = 0; atom < kAtomCount; ++atom) {
      atom_sources.data[residue * kAtomCount + atom] = slot.atom_sources[atom];
    }
  }
  return universal::ok_status();
}

universal::Status canonical_atom_pack(
    const CanonicalAtomPackRequest& request,
    const CanonicalAtomPackOutput& output) noexcept {
  const std::size_t residue_count = request.residues.size;
  if (residue_count == 0) {
    return universal::ok_status();
  }
  if (request.residues.data == nullptr ||
      !has_coordinate_capacity(output.coordinates, residue_count) ||
      !has_source_capacity(output.atom_sources, residue_count)) {
    return universal::invalid_argument_status(
        "canonical_atom_pack requires residue records and [L,5,3]/[L,5] outputs");
  }

  std::fill_n(output.coordinates.data, coord_count(residue_count), 0.0F);
  const universal::Status source_status =
      atom_source_extract(request.residues, output.atom_sources);
  if (!source_status.ok()) {
    return source_status;
  }

  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    const detail::NormalizedResidue& slot = request.residues.data[residue];
    for (std::size_t atom = 0; atom < kAtomCount; ++atom) {
      if (slot.atom_sources[atom] != universal::AtomSource::Observed) {
        continue;
      }
      const std::size_t offset =
          (residue * kAtomCount + atom) * kAxisCount;
      for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
        output.coordinates.data[offset + axis] = slot.coords[atom][axis];
      }
    }
  }

  return universal::ok_status();
}

}  // namespace hikoboshi::io

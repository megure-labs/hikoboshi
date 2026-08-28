#include <hikoboshi/primitives/compute/atom_pair_distance.hpp>

#include <cstddef>
#include <cstdint>

namespace hikoboshi::primitives::compute {
namespace {

constexpr float kPairDistanceFloor = 1.0e-6F;

struct AtomPair {
  std::size_t query_atom;
  std::size_t neighbor_atom;
};

constexpr std::size_t atom_index(
    hikoboshi::universal::CanonicalAtom atom) noexcept {
  return static_cast<std::size_t>(atom);
}

constexpr AtomPair kAtomPairs[kAtomPairDistancePairCount] = {
    {atom_index(hikoboshi::universal::CanonicalAtom::CA),
     atom_index(hikoboshi::universal::CanonicalAtom::CA)},
    {atom_index(hikoboshi::universal::CanonicalAtom::N),
     atom_index(hikoboshi::universal::CanonicalAtom::N)},
    {atom_index(hikoboshi::universal::CanonicalAtom::C),
     atom_index(hikoboshi::universal::CanonicalAtom::C)},
    {atom_index(hikoboshi::universal::CanonicalAtom::O),
     atom_index(hikoboshi::universal::CanonicalAtom::O)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CB),
     atom_index(hikoboshi::universal::CanonicalAtom::CB)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CA),
     atom_index(hikoboshi::universal::CanonicalAtom::N)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CA),
     atom_index(hikoboshi::universal::CanonicalAtom::C)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CA),
     atom_index(hikoboshi::universal::CanonicalAtom::O)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CA),
     atom_index(hikoboshi::universal::CanonicalAtom::CB)},
    {atom_index(hikoboshi::universal::CanonicalAtom::N),
     atom_index(hikoboshi::universal::CanonicalAtom::C)},
    {atom_index(hikoboshi::universal::CanonicalAtom::N),
     atom_index(hikoboshi::universal::CanonicalAtom::O)},
    {atom_index(hikoboshi::universal::CanonicalAtom::N),
     atom_index(hikoboshi::universal::CanonicalAtom::CB)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CB),
     atom_index(hikoboshi::universal::CanonicalAtom::C)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CB),
     atom_index(hikoboshi::universal::CanonicalAtom::O)},
    {atom_index(hikoboshi::universal::CanonicalAtom::O),
     atom_index(hikoboshi::universal::CanonicalAtom::C)},
    {atom_index(hikoboshi::universal::CanonicalAtom::N),
     atom_index(hikoboshi::universal::CanonicalAtom::CA)},
    {atom_index(hikoboshi::universal::CanonicalAtom::C),
     atom_index(hikoboshi::universal::CanonicalAtom::CA)},
    {atom_index(hikoboshi::universal::CanonicalAtom::O),
     atom_index(hikoboshi::universal::CanonicalAtom::CA)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CB),
     atom_index(hikoboshi::universal::CanonicalAtom::CA)},
    {atom_index(hikoboshi::universal::CanonicalAtom::C),
     atom_index(hikoboshi::universal::CanonicalAtom::N)},
    {atom_index(hikoboshi::universal::CanonicalAtom::O),
     atom_index(hikoboshi::universal::CanonicalAtom::N)},
    {atom_index(hikoboshi::universal::CanonicalAtom::CB),
     atom_index(hikoboshi::universal::CanonicalAtom::N)},
    {atom_index(hikoboshi::universal::CanonicalAtom::C),
     atom_index(hikoboshi::universal::CanonicalAtom::CB)},
    {atom_index(hikoboshi::universal::CanonicalAtom::O),
     atom_index(hikoboshi::universal::CanonicalAtom::CB)},
    {atom_index(hikoboshi::universal::CanonicalAtom::C),
     atom_index(hikoboshi::universal::CanonicalAtom::O)},
};

static_assert(sizeof(kAtomPairs) / sizeof(kAtomPairs[0]) ==
                  kAtomPairDistancePairCount,
              "Hikoboshi-MPNN-64 atom-pair order must contain 25 groups");

std::size_t coord_offset(std::size_t residue,
                         std::size_t atom,
                         std::size_t axis) noexcept {
  return (residue * hikoboshi::universal::kCanonicalAtomCount + atom) *
             hikoboshi::universal::kCoordinateAxisCount +
         axis;
}

std::size_t atom_source_offset(std::size_t residue, std::size_t atom) noexcept {
  return residue * hikoboshi::universal::kCanonicalAtomCount + atom;
}

bool atom_missing(const AtomPairDistanceScalarRequest& request,
                  std::size_t residue,
                  std::size_t atom) noexcept {
  return request.atom_sources[atom_source_offset(residue, atom)] ==
         hikoboshi::universal::AtomSource::Missing;
}

bool valid_neighbor(const AtomPairDistanceScalarRequest& request,
                    std::int32_t neighbor) noexcept {
  return neighbor >= 0 &&
         static_cast<std::size_t>(neighbor) < request.residue_count;
}

float squared_distance(const AtomPairDistanceScalarRequest& request,
                       std::size_t query_residue,
                       std::size_t neighbor_residue,
                       AtomPair pair) noexcept {
  float squared = 0.0F;
  for (std::size_t axis = 0; axis < hikoboshi::universal::kCoordinateAxisCount;
       ++axis) {
    const float query =
        request.coordinates[coord_offset(query_residue, pair.query_atom, axis)];
    const float neighbor = request.coordinates[coord_offset(
        neighbor_residue, pair.neighbor_atom, axis)];
    const float delta = neighbor - query;
    squared += delta * delta;
  }
  return squared;
}

}  // namespace

void atom_pair_distance_scalar(
    const AtomPairDistanceScalarRequest& request,
    const AtomPairDistanceScalarOutput& output) noexcept {
  // ProteinMPNN applies sqrt(sum(dx^2) + 1e-6) for atom-pair distances.
  // This primitive stores the squared-distance input to rbf_scalar, so the
  // upstream floor is applied here before the downstream square root.
  if (request.residue_count == 0 || request.neighbor_count == 0) {
    return;
  }

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < request.neighbor_count; ++neighbor_slot) {
      const std::size_t slot =
          residue * request.neighbor_count + neighbor_slot;
      const std::int32_t neighbor = request.neighbor_indices[slot];
      float* row =
          output.squared_distances + slot * kAtomPairDistancePairCount;

      if (!valid_neighbor(request, neighbor)) {
        for (std::size_t pair_index = 0;
             pair_index < kAtomPairDistancePairCount; ++pair_index) {
          row[pair_index] = kAtomPairDistanceMissingSentinel;
        }
        continue;
      }

      const std::size_t neighbor_residue = static_cast<std::size_t>(neighbor);
      for (std::size_t pair_index = 0;
           pair_index < kAtomPairDistancePairCount; ++pair_index) {
        const AtomPair pair = kAtomPairs[pair_index];
        if (atom_missing(request, residue, pair.query_atom) ||
            atom_missing(request, neighbor_residue, pair.neighbor_atom)) {
          row[pair_index] = kAtomPairDistanceMissingSentinel;
          continue;
        }
        row[pair_index] =
            squared_distance(request, residue, neighbor_residue, pair) +
            kPairDistanceFloor;
      }
    }
  }
}

}  // namespace hikoboshi::primitives::compute

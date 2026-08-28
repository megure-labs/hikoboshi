#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_ATOM_PAIR_DISTANCE_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_ATOM_PAIR_DISTANCE_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::primitives::compute {

inline constexpr std::size_t kAtomPairDistancePairCount = 25;
inline constexpr float kAtomPairDistanceMissingSentinel = -1.0F;

struct AtomPairDistanceScalarRequest {
  const float* coordinates;  // row-major [L, 5, 3]
  const hikoboshi::universal::AtomSource* atom_sources;  // row-major [L, 5]
  const std::int32_t* neighbor_indices;  // row-major [L, K]
  std::size_t residue_count;
  std::size_t neighbor_count;
};

struct AtomPairDistanceScalarOutput {
  float* squared_distances;  // row-major [L, K, 25]
};

void atom_pair_distance_scalar(
    const AtomPairDistanceScalarRequest& request,
    const AtomPairDistanceScalarOutput& output) noexcept;

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_ATOM_PAIR_DISTANCE_HPP

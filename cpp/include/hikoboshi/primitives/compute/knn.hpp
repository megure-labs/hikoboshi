#ifndef HIKOBOSHI_PRIMITIVES_COMPUTE_KNN_HPP
#define HIKOBOSHI_PRIMITIVES_COMPUTE_KNN_HPP

#include <cstddef>
#include <cstdint>

namespace hikoboshi::primitives::compute {

// PyTorch ProteinMPNN _dist_and_neighbors semantics (hikoboshi_train/features.py):
//
//   mask_2D = mask_i * mask_j
//   D       = mask_2D * sqrt(sum(dx^2) + eps)        eps = 1e-6
//   D_max   = max over j of D[i, j]
//   D_adj   = D + (1 - mask_2D) * D_max
//   neighbors = topk(D_adj, k, largest=False)
//
// Effect: invalid (mask=0) pairs are promoted to D_max and land at the
// bottom of the rank, never displacing a real neighbor when at least k
// valid candidates exist. The sqrt(sum + 1e-6) floor is part of the
// trained-model semantics: Hikoboshi-MPNN-64 was trained against this
// distance, so Hikoboshi matches it for inference parity.
//
// Validity selection:
//   - When `query_validity_mask` / `target_validity_mask` is non-null,
//     each entry is 0 for invalid and non-zero for valid. The pointer
//     must reference query_count / target_count uint8 entries.
//   - When the pointer is null and `treat_zero_coords_as_invalid` is
//     true, knn_scalar treats any coordinate with all three components
//     exactly equal to 0.0F as invalid. This matches the Hikoboshi MPNN
//     convention from `cpp/modules/mpnn/mpnn_scalar.cpp:build_ca_coordinates`,
//     which writes (0, 0, 0) for residues whose CA atom is Missing.
//   - When both are null/false, every coordinate is treated as valid
//     (legacy behavior used by the `tests/cpp/primitive_compute_goldens.cpp`
//     unit tests that exercise (0, 0, 0) as a real coordinate).
struct KnnScalarRequest {
  const float* query_coordinates;
  const float* target_coordinates;
  const std::uint8_t* query_validity_mask;
  const std::uint8_t* target_validity_mask;
  std::size_t query_count;
  std::size_t target_count;
  std::size_t k;
  bool include_self;
  bool treat_zero_coords_as_invalid;
};

struct KnnScalarOutput {
  std::int32_t* neighbor_indices;
  // Stores `squared + 1e-6` for valid neighbors and `D_max^2` for
  // invalid ones / unfilled slots. sqrt(value) recovers the PyTorch
  // sqrt-space distance, matching `D_neighbors` from
  // `_dist_and_neighbors`.
  float* neighbor_squared_distances;
};

void knn_scalar(const KnnScalarRequest& request, const KnnScalarOutput& output);

}  // namespace hikoboshi::primitives::compute

#endif  // HIKOBOSHI_PRIMITIVES_COMPUTE_KNN_HPP

// Hikoboshi-MPNN-64 KNN graph for the scalar backend.
//
// The semantics match PyTorch ProteinMPNN's `_dist_and_neighbors`
// (hikoboshi_train/features.py:_dist_and_neighbors), which is the
// distance the production Hikoboshi-MPNN-64 weights were trained against:
//
//     mask_2D = mask_i * mask_j
//     D       = mask_2D * sqrt(sum(dx^2) + eps)         eps = 1.0e-6
//     D_max   = max over j of D[i, j]
//     D_adj   = D + (1 - mask_2D) * D_max
//     k_nn    = torch.topk(D_adj, k, largest=False, sorted=True)
//
// Invalid (mask=0) pairs are promoted to D_max so that they are sorted
// to the bottom of the rank and never displace a real neighbor unless
// the row has fewer than k valid candidates. The sqrt(sum + 1.0e-6)
// floor is part of the trained-model contract; without it the
// self-distance is 0 in Hikoboshi and 1.0e-3 in PyTorch (the
// `bench/MPNN64_REAL_PARITY_DIFFS_KABSCH.md` 1e-3 self-distance row).
//
// Storage convention: the output buffer keeps the
// `squared + 1.0e-6` form so that:
//   - downstream consumers that take sqrt(value) recover the PyTorch
//     sqrt-space distance,
//   - existing `tests/cpp/primitive_compute_goldens.cpp` assertions
//     against squared distances stay within their 1e-5 tolerance,
//   - ranking by `squared + eps` is monotone-equivalent to ranking by
//     `sqrt(squared + eps)` for non-negative distances, so the rank
//     order matches PyTorch.
//
// For invalid neighbor slots (when mask is provided and a target is
// invalid) the output buffer holds `D_max^2`, so sqrt(value) recovers
// `D_max` matching PyTorch's masked topk fill.

#include <hikoboshi/primitives/compute/knn.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace hikoboshi::primitives::compute {

namespace {

constexpr float kSquaredDistanceEpsilon = 1.0e-6F;

float squared_distance_with_floor(const float* a, const float* b) noexcept {
  const float dx = a[0] - b[0];
  const float dy = a[1] - b[1];
  const float dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz + kSquaredDistanceEpsilon;
}

bool coord_is_zero(const float* xyz) noexcept {
  return xyz[0] == 0.0F && xyz[1] == 0.0F && xyz[2] == 0.0F;
}

bool coordinate_valid(const KnnScalarRequest& request,
                      const std::uint8_t* mask,
                      const float* coordinates,
                      std::size_t index) noexcept {
  if (mask != nullptr) {
    return mask[index] != 0;
  }
  if (request.treat_zero_coords_as_invalid) {
    return !coord_is_zero(coordinates + index * 3);
  }
  return true;
}

}  // namespace

void knn_scalar(const KnnScalarRequest& request, const KnnScalarOutput& output) {
  if (request.k == 0 || request.query_count == 0) {
    return;
  }

  const std::size_t k = request.k;
  const float infinity = std::numeric_limits<float>::infinity();
  const bool same_buffer =
      request.query_coordinates == request.target_coordinates;

  for (std::size_t qi = 0; qi < request.query_count; ++qi) {
    std::int32_t* indices = output.neighbor_indices + qi * k;
    float* distances = output.neighbor_squared_distances + qi * k;
    for (std::size_t s = 0; s < k; ++s) {
      indices[s] = -1;
      distances[s] = infinity;
    }

    const float* query_xyz = request.query_coordinates + qi * 3;
    const bool query_valid =
        coordinate_valid(request, request.query_validity_mask,
                         request.query_coordinates, qi);

    // Pass 1: compute D_max^2 over valid (mask=1) targets that are
    // contributing to this row, matching PyTorch's row-wise max of
    // mask_2D * sqrt(sum + eps). When the query itself is invalid all
    // mask_2D entries in the row are 0, so D_max is 0 in PyTorch and
    // the row contains arbitrary indices; we mirror that by tracking
    // D_max^2 = eps so sqrt(eps) = 1e-3 for the dump.
    float row_max_squared = kSquaredDistanceEpsilon;
    if (query_valid) {
      for (std::size_t ti = 0; ti < request.target_count; ++ti) {
        if (!request.include_self && same_buffer && qi == ti) {
          continue;
        }
        if (!coordinate_valid(request, request.target_validity_mask,
                              request.target_coordinates, ti)) {
          continue;
        }
        const float* target_xyz = request.target_coordinates + ti * 3;
        const float candidate = squared_distance_with_floor(query_xyz, target_xyz);
        if (candidate > row_max_squared) {
          row_max_squared = candidate;
        }
      }
    }

    // Pass 2: rank and insert.
    for (std::size_t ti = 0; ti < request.target_count; ++ti) {
      if (!request.include_self && same_buffer && qi == ti) {
        continue;
      }

      float dist = 0.0F;
      const bool target_valid =
          coordinate_valid(request, request.target_validity_mask,
                           request.target_coordinates, ti);
      if (!query_valid || !target_valid) {
        // PyTorch promotes invalid rows/columns to D_max (sqrt-space),
        // which corresponds to row_max_squared in our squared+eps space.
        dist = row_max_squared;
      } else {
        const float* target_xyz = request.target_coordinates + ti * 3;
        dist = squared_distance_with_floor(query_xyz, target_xyz);
      }

      // Strict less-than: existing values survive ties so the earliest
      // target index wins, per canonical-map deterministic KNN tie behavior.
      std::size_t insert_at = k;
      for (std::size_t s = 0; s < k; ++s) {
        if (dist < distances[s]) {
          insert_at = s;
          break;
        }
      }
      if (insert_at == k) {
        continue;
      }
      for (std::size_t s = k - 1; s > insert_at; --s) {
        distances[s] = distances[s - 1];
        indices[s] = indices[s - 1];
      }
      distances[insert_at] = dist;
      indices[insert_at] = static_cast<std::int32_t>(ti);
    }

    // Replace any remaining `infinity` sentinels (rows where fewer than
    // k candidates were even considered) with row_max_squared so that
    // sqrt(value) recovers a finite D_max matching PyTorch's masked
    // topk fill behavior.
    for (std::size_t s = 0; s < k; ++s) {
      if (distances[s] == infinity) {
        distances[s] = row_max_squared;
      }
    }
  }
}

}  // namespace hikoboshi::primitives::compute

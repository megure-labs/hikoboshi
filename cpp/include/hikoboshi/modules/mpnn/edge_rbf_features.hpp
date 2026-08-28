#ifndef HIKOBOSHI_MODULES_MPNN_EDGE_RBF_FEATURES_HPP
#define HIKOBOSHI_MODULES_MPNN_EDGE_RBF_FEATURES_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::modules::mpnn {

struct EdgeRbfFeaturesRequest {
  const float* coordinates;  // row-major [L, 5, 3]
  const hikoboshi::universal::AtomSource* atom_sources;  // row-major [L, 5]
  const std::int32_t* neighbor_indices;  // row-major [L, K]
  const std::int32_t* residue_indices;  // optional [L], defaults to 0..L-1
  const std::int32_t* chain_labels;  // optional [L], defaults to one chain
  hikoboshi::universal::Span<const float> positional_weight;  // [16, 66]
  hikoboshi::universal::Span<const float> positional_bias;    // optional [16]
  std::size_t residue_count;
  std::size_t neighbor_count;
  std::size_t rbf_count;
};

struct EdgeRbfFeaturesOutput {
  float* atom_pair_squared_distances;  // scratch [L, K, 25]
  float* rbf_features;                 // row-major [L, K, 25 * R]
  float* edge_features;                // row-major [L, K, 16 + 25 * R]
};

void edge_rbf_features_scalar(const EdgeRbfFeaturesRequest& request,
                              const EdgeRbfFeaturesOutput& output) noexcept;

}  // namespace hikoboshi::modules::mpnn

#endif  // HIKOBOSHI_MODULES_MPNN_EDGE_RBF_FEATURES_HPP

#include <hikoboshi/modules/mpnn/edge_rbf_features.hpp>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hikoboshi::modules::mpnn {
namespace {

constexpr float kRbfCenterMin = 2.0F;
constexpr float kRbfCenterMax = 22.0F;
constexpr float kOneOverSqrtTwo = 0.7071067811865475F;

void fill_zero(float* values, std::size_t count) noexcept {
  std::fill_n(values, count, 0.0F);
}

void copy_values(const float* source, float* target, std::size_t count) noexcept {
  std::memcpy(target, source, count * sizeof(float));
}

bool valid_neighbor(const EdgeRbfFeaturesRequest& request,
                    std::int32_t neighbor) noexcept {
  return neighbor >= 0 &&
         static_cast<std::size_t>(neighbor) < request.residue_count;
}

std::int32_t residue_index(const EdgeRbfFeaturesRequest& request,
                           std::size_t residue) noexcept {
  return request.residue_indices != nullptr
             ? request.residue_indices[residue]
             : static_cast<std::int32_t>(residue);
}

std::int32_t chain_label(const EdgeRbfFeaturesRequest& request,
                         std::size_t residue) noexcept {
  return request.chain_labels != nullptr ? request.chain_labels[residue] : 0;
}

std::size_t positional_class(const EdgeRbfFeaturesRequest& request,
                             std::size_t residue,
                             std::size_t neighbor) noexcept {
  if (chain_label(request, residue) != chain_label(request, neighbor)) {
    return detail::kMpnn64PositionalClassCount - 1;
  }
  std::int32_t offset =
      residue_index(request, residue) - residue_index(request, neighbor);
  offset = std::max(-detail::kMpnn64MaxRelativePosition,
                    std::min(detail::kMpnn64MaxRelativePosition, offset));
  return static_cast<std::size_t>(offset + detail::kMpnn64MaxRelativePosition);
}

void write_positional_features(const EdgeRbfFeaturesRequest& request,
                               std::size_t residue,
                               std::int32_t neighbor,
                               float* output) noexcept {
  if (!valid_neighbor(request, neighbor)) {
    fill_zero(output, detail::kMpnn64PositionalFeatureCount);
    return;
  }
  const std::size_t class_index =
      positional_class(request, residue, static_cast<std::size_t>(neighbor));
  for (std::size_t d = 0; d < detail::kMpnn64PositionalFeatureCount; ++d) {
    output[d] =
        request.positional_weight.data[d * detail::kMpnn64PositionalClassCount +
                                       class_index] +
        (request.positional_bias.data != nullptr
             ? request.positional_bias.data[d]
             : 0.0F);
  }
}

void zero_missing_rbf_features(const EdgeRbfFeaturesRequest& request,
                               const EdgeRbfFeaturesOutput& output) noexcept {
  const std::size_t slot_count = request.residue_count * request.neighbor_count;
  const std::size_t pair_count =
      hikoboshi::primitives::compute::kAtomPairDistancePairCount;
  for (std::size_t value_index = 0; value_index < slot_count * pair_count;
       ++value_index) {
    if (output.atom_pair_squared_distances[value_index] ==
        hikoboshi::primitives::compute::kAtomPairDistanceMissingSentinel) {
      fill_zero(output.rbf_features + value_index * request.rbf_count,
                request.rbf_count);
    }
  }
}

void write_edge_features(const EdgeRbfFeaturesRequest& request,
                         const EdgeRbfFeaturesOutput& output) noexcept {
  const std::size_t edge_rbf_dim =
      hikoboshi::primitives::compute::kAtomPairDistancePairCount *
      request.rbf_count;
  const std::size_t edge_dim =
      detail::kMpnn64PositionalFeatureCount + edge_rbf_dim;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < request.neighbor_count; ++neighbor_slot) {
      const std::size_t slot =
          residue * request.neighbor_count + neighbor_slot;
      float* edge_row = output.edge_features + slot * edge_dim;
      write_positional_features(request, residue,
                                request.neighbor_indices[slot], edge_row);
      copy_values(output.rbf_features + slot * edge_rbf_dim,
                  edge_row + detail::kMpnn64PositionalFeatureCount,
                  edge_rbf_dim);
    }
  }
}

}  // namespace

void edge_rbf_features_scalar(const EdgeRbfFeaturesRequest& request,
                              const EdgeRbfFeaturesOutput& output) noexcept {
  if (request.residue_count == 0 || request.neighbor_count == 0 ||
      request.rbf_count == 0) {
    return;
  }

  hikoboshi::primitives::compute::AtomPairDistanceScalarRequest distance{};
  distance.coordinates = request.coordinates;
  distance.atom_sources = request.atom_sources;
  distance.neighbor_indices = request.neighbor_indices;
  distance.residue_count = request.residue_count;
  distance.neighbor_count = request.neighbor_count;
  hikoboshi::primitives::compute::AtomPairDistanceScalarOutput distance_output{};
  distance_output.squared_distances = output.atom_pair_squared_distances;
  hikoboshi::dispatch::atom_pair_distance_forward(
      hikoboshi::dispatch::ScalarTag{}, distance, distance_output);

  const float rbf_sigma =
      (kRbfCenterMax - kRbfCenterMin) /
      (static_cast<float>(request.rbf_count) / kOneOverSqrtTwo);
  hikoboshi::primitives::compute::RbfScalarRequest rbf{};
  rbf.squared_distances = output.atom_pair_squared_distances;
  rbf.value_count =
      request.residue_count * request.neighbor_count *
      hikoboshi::primitives::compute::kAtomPairDistancePairCount;
  rbf.feature_count = request.rbf_count;
  rbf.center_min = kRbfCenterMin;
  rbf.center_max = kRbfCenterMax;
  rbf.sigma = rbf_sigma;
  hikoboshi::dispatch::rbf_forward(hikoboshi::dispatch::ScalarTag{}, rbf,
                                 output.rbf_features);

  zero_missing_rbf_features(request, output);
  write_edge_features(request, output);
}

}  // namespace hikoboshi::modules::mpnn

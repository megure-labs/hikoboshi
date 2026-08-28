#include <hikoboshi/modules/mpnn.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko_m = hikoboshi::modules;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;

namespace {

bool nearly_equal(float a, float b, float tolerance = 1.0e-5F) {
  return std::fabs(a - b) <= tolerance;
}

void fail(const char* detail) {
  std::fprintf(stderr, "mpnn_input_edge_prep_tests: %s\n", detail);
  std::exit(1);
}

hiko_u::Span<const float> span(const std::vector<float>& values) {
  return {values.data(), values.size()};
}

struct OwnedWorkspace {
  std::vector<float> ca_coordinates;
  std::vector<float> edge_inputs;
  std::vector<std::int32_t> neighbor_indices;
  std::vector<float> neighbor_squared_distances;
  std::vector<float> rbf_features;
  std::vector<float> residue_state;
  std::vector<float> gathered_state;
  std::vector<float> edge_state;
  std::vector<float> message_state;
  std::vector<float> projected_message_state;
  std::vector<float> residue_scratch;
  std::vector<float> ffn_hidden;
  hiko_d::Mpnn64Workspace view{};
};

OwnedWorkspace make_workspace(const hiko_d::Mpnn64MemoryPlan& plan) {
  OwnedWorkspace owned{};
  owned.ca_coordinates.resize(hiko_d::mpnn64_ca_coordinate_count(plan));
  owned.edge_inputs.resize(hiko_d::mpnn64_neighbor_edge_feature_count(plan));
  owned.neighbor_indices.resize(hiko_d::mpnn64_neighbor_slot_count(plan));
  owned.neighbor_squared_distances.resize(hiko_d::mpnn64_neighbor_slot_count(plan));
  owned.rbf_features.resize(hiko_d::mpnn64_neighbor_rbf_count(plan));
  owned.residue_state.resize(hiko_d::mpnn64_residue_hidden_count(plan));
  owned.gathered_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.edge_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.message_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.projected_message_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.residue_scratch.resize(hiko_d::mpnn64_residue_hidden_count(plan));
  owned.ffn_hidden.resize(hiko_d::mpnn64_residue_hidden_count(plan));
  owned.view = {
      plan,
      {owned.ca_coordinates.data(), owned.ca_coordinates.size()},
      {owned.edge_inputs.data(), owned.edge_inputs.size()},
      {owned.neighbor_indices.data(), owned.neighbor_indices.size()},
      {owned.neighbor_squared_distances.data(),
       owned.neighbor_squared_distances.size()},
      {owned.rbf_features.data(), owned.rbf_features.size()},
      {owned.residue_state.data(), owned.residue_state.size()},
      {owned.gathered_state.data(), owned.gathered_state.size()},
      {owned.edge_state.data(), owned.edge_state.size()},
      {owned.message_state.data(), owned.message_state.size()},
      {owned.projected_message_state.data(),
       owned.projected_message_state.size()},
      {owned.residue_scratch.data(), owned.residue_scratch.size()},
      {owned.ffn_hidden.data(), owned.ffn_hidden.size()},
  };
  return owned;
}

struct SyntheticWeights {
  std::vector<float> W_e_weight;
  std::vector<float> W_e_bias;
  std::vector<float> edge_embedding_weight;
  std::vector<float> edge_embedding_bias;
  std::vector<float> edge_norm_weight;
  std::vector<float> edge_norm_bias;
  std::vector<float> positional_weight;
  std::vector<float> positional_bias;
  hiko_d::Mpnn64Weights view{};
};

SyntheticWeights make_weights(const hiko_m::Mpnn64Descriptor& descriptor) {
  const hiko_d::Mpnn64MemoryPlan plan{1,
                                   descriptor.hidden_dimension,
                                   descriptor.neighbor_count,
                                   descriptor.rbf_count,
                                   descriptor.layer_count};
  SyntheticWeights weights{};
  weights.W_e_weight.resize(descriptor.hidden_dimension *
                                descriptor.hidden_dimension,
                            0.0F);
  weights.W_e_bias.resize(descriptor.hidden_dimension, 0.0F);
  for (std::size_t h = 0; h < descriptor.hidden_dimension; ++h) {
    weights.W_e_weight[h * descriptor.hidden_dimension + h] = 1.0F;
  }
  weights.edge_embedding_weight.resize(descriptor.hidden_dimension *
                                           hiko_d::mpnn64_edge_feature_dimension(plan),
                                       0.0F);
  weights.edge_embedding_bias.resize(descriptor.hidden_dimension, 0.0F);
  weights.edge_norm_weight.resize(descriptor.hidden_dimension, 1.0F);
  weights.edge_norm_bias.resize(descriptor.hidden_dimension, 0.0F);
  weights.positional_weight.resize(hiko_d::kMpnn64PositionalFeatureCount *
                                       hiko_d::kMpnn64PositionalClassCount,
                                   0.0F);
  weights.positional_bias.resize(hiko_d::kMpnn64PositionalFeatureCount, 0.0F);
  weights.view.W_e = {span(weights.W_e_weight), span(weights.W_e_bias)};
  weights.view.edge_embedding = {
      {span(weights.edge_embedding_weight), span(weights.edge_embedding_bias)},
      {span(weights.edge_norm_weight), span(weights.edge_norm_bias)}};
  weights.view.positional_encoding =
      {span(weights.positional_weight), span(weights.positional_bias)};
  weights.view.layer_count = descriptor.layer_count;
  return weights;
}

hiko_m::Mpnn64Descriptor descriptor(std::size_t neighbor_count) {
  return {hiko_d::kMpnn64HiddenDimension, neighbor_count, 16, 0, 30.0F};
}

std::vector<float> coordinates_from_ca(
    const std::vector<std::array<float, 3>>& ca_points) {
  std::vector<float> coordinates(ca_points.size() * hiko_d::kMpnn64AtomCount *
                                     hiko_d::kMpnn64AxisCount,
                                 0.0F);
  for (std::size_t residue = 0; residue < ca_points.size(); ++residue) {
    for (std::size_t atom = 0; atom < hiko_d::kMpnn64AtomCount; ++atom) {
      for (std::size_t axis = 0; axis < hiko_d::kMpnn64AxisCount; ++axis) {
        coordinates[(residue * hiko_d::kMpnn64AtomCount + atom) *
                        hiko_d::kMpnn64AxisCount +
                    axis] = ca_points[residue][axis];
      }
    }
  }
  return coordinates;
}

hiko_u::Status run_forward(const std::vector<float>& coordinates,
                        const hiko_m::Mpnn64Descriptor& desc,
                        SyntheticWeights& weights,
                        OwnedWorkspace& workspace) {
  const std::size_t residue_count =
      coordinates.size() / (hiko_d::kMpnn64AtomCount * hiko_d::kMpnn64AxisCount);
  std::vector<hiko_u::AtomSource> atom_sources(residue_count *
                                                hiko_d::kMpnn64AtomCount,
                                            hiko_u::AtomSource::Observed);
  std::vector<float> embeddings(residue_count * desc.hidden_dimension, -1.0F);
  return hiko_m::mpnn64_forward_scalar(
      {coordinates.data(), atom_sources.data(), residue_count, desc,
       &weights.view, &workspace.view},
      {embeddings.data(), residue_count, desc.hidden_dimension});
}

void require_ok(hiko_u::Status status) {
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("MPNN forward returned non-ok");
  }
}

void test_knn_ties_by_index() {
  const hiko_m::Mpnn64Descriptor desc = descriptor(3);
  const std::vector<float> coordinates =
      coordinates_from_ca({{{0.0F, 0.0F, 0.0F}},
                           {{1.0F, 0.0F, 0.0F}},
                           {{-1.0F, 0.0F, 0.0F}}});
  const hiko_d::Mpnn64MemoryPlan plan{3,
                                   desc.hidden_dimension,
                                   desc.neighbor_count,
                                   desc.rbf_count,
                                   desc.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  SyntheticWeights weights = make_weights(desc);
  require_ok(run_forward(coordinates, desc, weights, workspace));

  if (workspace.neighbor_indices[0] != 0 ||
      workspace.neighbor_indices[1] != 1 ||
      workspace.neighbor_indices[2] != 2) {
    fail("KNN tie between equal-distance residues must keep lower index first");
  }
}

void set_atom(std::vector<float>& coordinates,
              std::size_t residue,
              hiko_u::CanonicalAtom atom,
              float x,
              float y,
              float z) {
  const std::size_t atom_index = static_cast<std::size_t>(atom);
  const std::size_t offset =
      (residue * hiko_d::kMpnn64AtomCount + atom_index) *
      hiko_d::kMpnn64AxisCount;
  coordinates[offset + 0] = x;
  coordinates[offset + 1] = y;
  coordinates[offset + 2] = z;
}

void test_rbf_feature_ordering() {
  const hiko_m::Mpnn64Descriptor desc = descriptor(2);
  std::vector<float> coordinates(2 * hiko_d::kMpnn64AtomCount *
                                     hiko_d::kMpnn64AxisCount,
                                 0.0F);
  set_atom(coordinates, 0, hiko_u::CanonicalAtom::CA, 0.0F, 0.0F, 0.0F);
  set_atom(coordinates, 0, hiko_u::CanonicalAtom::N, 0.0F, 0.0F, 0.0F);
  set_atom(coordinates, 1, hiko_u::CanonicalAtom::CA, 2.0F, 0.0F, 0.0F);
  set_atom(coordinates, 1, hiko_u::CanonicalAtom::N, 22.0F, 0.0F, 0.0F);

  const hiko_d::Mpnn64MemoryPlan plan{2,
                                   desc.hidden_dimension,
                                   desc.neighbor_count,
                                   desc.rbf_count,
                                   desc.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  SyntheticWeights weights = make_weights(desc);
  require_ok(run_forward(coordinates, desc, weights, workspace));

  const std::size_t slot = 1;  // residue 0 -> residue 1, after self edge
  const std::size_t rbf_row_offset =
      slot * hiko_d::kMpnn64AtomPairCount * desc.rbf_count;
  const float ca_ca_first_bin = workspace.rbf_features[rbf_row_offset + 0];
  const float n_n_last_bin =
      workspace.rbf_features[rbf_row_offset + desc.rbf_count + 15];
  if (!nearly_equal(ca_ca_first_bin, 1.0F) ||
      !nearly_equal(n_n_last_bin, 1.0F)) {
    fail("RBF groups must follow ProteinMPNN Ca-Ca, N-N ordering");
  }
}

void test_positional_feature_shape_and_class() {
  const hiko_m::Mpnn64Descriptor desc = descriptor(2);
  const std::vector<float> coordinates =
      coordinates_from_ca({{{0.0F, 0.0F, 0.0F}},
                           {{2.0F, 0.0F, 0.0F}}});
  const hiko_d::Mpnn64MemoryPlan plan{2,
                                   desc.hidden_dimension,
                                   desc.neighbor_count,
                                   desc.rbf_count,
                                   desc.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  SyntheticWeights weights = make_weights(desc);
  for (std::size_t d = 0; d < hiko_d::kMpnn64PositionalFeatureCount; ++d) {
    for (std::size_t c = 0; c < hiko_d::kMpnn64PositionalClassCount; ++c) {
      weights.positional_weight[d * hiko_d::kMpnn64PositionalClassCount + c] =
          static_cast<float>(1000 * d + c);
    }
  }
  weights.view.positional_encoding.weight = span(weights.positional_weight);
  require_ok(run_forward(coordinates, desc, weights, workspace));

  const std::size_t edge_dim = hiko_d::mpnn64_edge_feature_dimension(plan);
  const float* self_edge = workspace.edge_inputs.data();
  const float* neighbor_edge = workspace.edge_inputs.data() + edge_dim;
  if (!nearly_equal(self_edge[3], 3032.0F)) {
    fail("self positional class must be offset class 32");
  }
  if (!nearly_equal(neighbor_edge[3], 3031.0F)) {
    fail("residue 0 to residue 1 positional class must be offset class 31");
  }
  if (edge_dim != hiko_d::kMpnn64EdgeFeatureCount ||
      hiko_d::kMpnn64PositionalFeatureCount + 25 * desc.rbf_count != edge_dim) {
    fail("edge feature shape must be 16 positional + 25*R RBF = 416");
  }
}

void test_edge_embedding_projection_norm_and_validation() {
  const hiko_m::Mpnn64Descriptor desc = descriptor(2);
  const std::vector<float> coordinates =
      coordinates_from_ca({{{0.0F, 0.0F, 0.0F}},
                           {{2.0F, 0.0F, 0.0F}}});
  const hiko_d::Mpnn64MemoryPlan plan{2,
                                   desc.hidden_dimension,
                                   desc.neighbor_count,
                                   desc.rbf_count,
                                   desc.layer_count};

  OwnedWorkspace workspace = make_workspace(plan);
  SyntheticWeights weights = make_weights(desc);
  for (std::size_t h = 0; h < desc.hidden_dimension; ++h) {
    weights.edge_embedding_bias[h] = static_cast<float>(h);
  }
  weights.view.edge_embedding.linear.bias = span(weights.edge_embedding_bias);
  require_ok(run_forward(coordinates, desc, weights, workspace));

  float mean = 0.0F;
  for (std::size_t h = 0; h < desc.hidden_dimension; ++h) {
    mean += static_cast<float>(h);
  }
  mean /= static_cast<float>(desc.hidden_dimension);
  float variance = 0.0F;
  for (std::size_t h = 0; h < desc.hidden_dimension; ++h) {
    const float delta = static_cast<float>(h) - mean;
    variance += delta * delta;
  }
  variance /= static_cast<float>(desc.hidden_dimension);
  const float expected0 = (0.0F - mean) /
                          std::sqrt(variance + hiko_d::kMpnn64LayerNormEpsilon);
  const float expected63 = (63.0F - mean) /
                           std::sqrt(variance + hiko_d::kMpnn64LayerNormEpsilon);
  if (!nearly_equal(workspace.edge_state[0], expected0) ||
      !nearly_equal(workspace.edge_state[63], expected63)) {
    fail("edge embedding must apply linear bias, layer norm, and W_e");
  }

  SyntheticWeights invalid = make_weights(desc);
  invalid.view.edge_embedding.linear.weight =
      {invalid.edge_embedding_weight.data(),
       invalid.edge_embedding_weight.size() - 1};
  OwnedWorkspace invalid_workspace = make_workspace(plan);
  const hiko_u::Status status =
      run_forward(coordinates, desc, invalid, invalid_workspace);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("wrong edge embedding shape must be rejected");
  }
}

}  // namespace

int main() {
  test_knn_ties_by_index();
  test_rbf_feature_ordering();
  test_positional_feature_shape_and_class();
  test_edge_embedding_projection_norm_and_validation();
  return 0;
}

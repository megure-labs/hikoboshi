#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko_m = hikoboshi::modules;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

bool nearly_equal(float a, float b, float tolerance = 1.0e-5F) {
  return std::fabs(a - b) <= tolerance;
}

void fail(const char* detail) {
  std::fprintf(stderr, "mpnn_module_goldens: %s\n", detail);
  std::exit(1);
}

hiko_u::Span<const float> span(const std::vector<float>& values) {
  return {values.data(), values.size()};
}

struct OwnedWorkspace {
  std::vector<float> ca_coordinates;
  std::vector<float> residue_features;
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
  owned.residue_features.resize(hiko_d::mpnn64_residue_feature_count(plan));
  owned.neighbor_indices.resize(hiko_d::mpnn64_neighbor_slot_count(plan));
  owned.neighbor_squared_distances.resize(hiko_d::mpnn64_neighbor_slot_count(plan));
  owned.rbf_features.resize(hiko_d::mpnn64_neighbor_rbf_count(plan));
  owned.residue_state.resize(hiko_d::mpnn64_residue_hidden_count(plan));
  owned.gathered_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.edge_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.message_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.projected_message_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.residue_scratch.resize(hiko_d::mpnn64_residue_hidden_count(plan));
  owned.ffn_hidden.resize(hiko_d::mpnn64_ffn_hidden_count(plan));
  owned.view = {
      plan,
      {owned.ca_coordinates.data(), owned.ca_coordinates.size()},
      {owned.residue_features.data(), owned.residue_features.size()},
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
  struct LayerStorage {
    std::vector<float> W1_weight;
    std::vector<float> W1_bias;
    std::vector<float> W11_weight;
    std::vector<float> W11_bias;
    std::vector<float> W12_weight;
    std::vector<float> W12_bias;
    std::vector<float> W13_weight;
    std::vector<float> W13_bias;
    std::vector<float> W2_weight;
    std::vector<float> W2_bias;
    std::vector<float> W3_weight;
    std::vector<float> W3_bias;
    std::vector<float> ffn_W_in_weight;
    std::vector<float> ffn_W_in_bias;
    std::vector<float> ffn_W_out_weight;
    std::vector<float> ffn_W_out_bias;
    std::vector<float> norm1_weight;
    std::vector<float> norm1_bias;
    std::vector<float> norm2_weight;
    std::vector<float> norm2_bias;
    std::vector<float> norm3_weight;
    std::vector<float> norm3_bias;
  };

  std::vector<float> W_e_weight;
  std::vector<float> W_e_bias;
  std::vector<float> edge_embedding_weight;
  std::vector<float> edge_embedding_bias;
  std::vector<float> edge_norm_weight;
  std::vector<float> edge_norm_bias;
  std::vector<float> positional_weight;
  std::vector<float> positional_bias;
  std::vector<LayerStorage> layer_storage;
  std::vector<hiko_d::Mpnn64LayerWeights> layers;
  hiko_d::Mpnn64Weights view{};
};

// SyntheticWeights is retained as scaffold coverage for workspace sizing,
// KNN padding, and zero-message behavior. It is not an authoritative
// Hikoboshi-MPNN-64 parity fixture.
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
  weights.layer_storage.resize(descriptor.layer_count);
  weights.layers.resize(descriptor.layer_count);
  for (std::size_t layer = 0; layer < descriptor.layer_count; ++layer) {
    SyntheticWeights::LayerStorage& storage = weights.layer_storage[layer];
    storage.W1_weight.resize(descriptor.hidden_dimension *
                                 hiko_d::kMpnn64MessageInputDimension,
                             0.0F);
    storage.W1_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.W11_weight.resize(descriptor.hidden_dimension *
                                  hiko_d::kMpnn64MessageInputDimension,
                              0.0F);
    storage.W11_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.W12_weight.resize(descriptor.hidden_dimension *
                                  descriptor.hidden_dimension,
                              0.0F);
    storage.W12_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.W13_weight.resize(descriptor.hidden_dimension *
                                  descriptor.hidden_dimension,
                              0.0F);
    storage.W13_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.W2_weight.resize(descriptor.hidden_dimension *
                                 descriptor.hidden_dimension,
                             0.0F);
    storage.W2_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.W3_weight.resize(descriptor.hidden_dimension *
                                 descriptor.hidden_dimension,
                             0.0F);
    storage.W3_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.ffn_W_in_weight.resize(hiko_d::kMpnn64FfnHiddenDimension *
                                       descriptor.hidden_dimension,
                                   0.0F);
    storage.ffn_W_in_bias.resize(hiko_d::kMpnn64FfnHiddenDimension, 0.0F);
    storage.ffn_W_out_weight.resize(descriptor.hidden_dimension *
                                        hiko_d::kMpnn64FfnHiddenDimension,
                                    0.0F);
    storage.ffn_W_out_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.norm1_weight.resize(descriptor.hidden_dimension, 1.0F);
    storage.norm1_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.norm2_weight.resize(descriptor.hidden_dimension, 1.0F);
    storage.norm2_bias.resize(descriptor.hidden_dimension, 0.0F);
    storage.norm3_weight.resize(descriptor.hidden_dimension, 1.0F);
    storage.norm3_bias.resize(descriptor.hidden_dimension, 0.0F);

    hiko_d::Mpnn64LayerWeights& layer_view = weights.layers[layer];
    layer_view.W1 = {span(storage.W1_weight), span(storage.W1_bias)};
    layer_view.W11 = {span(storage.W11_weight), span(storage.W11_bias)};
    layer_view.W12 = {span(storage.W12_weight), span(storage.W12_bias)};
    layer_view.W13 = {span(storage.W13_weight), span(storage.W13_bias)};
    layer_view.W2 = {span(storage.W2_weight), span(storage.W2_bias)};
    layer_view.W3 = {span(storage.W3_weight), span(storage.W3_bias)};
    layer_view.ffn = {{span(storage.ffn_W_in_weight),
                       span(storage.ffn_W_in_bias)},
                      {span(storage.ffn_W_out_weight),
                       span(storage.ffn_W_out_bias)}};
    layer_view.norm1 = {span(storage.norm1_weight), span(storage.norm1_bias)};
    layer_view.norm2 = {span(storage.norm2_weight), span(storage.norm2_bias)};
    layer_view.norm3 = {span(storage.norm3_weight), span(storage.norm3_bias)};
  }
  weights.view.layers = weights.layers.data();
  weights.view.W_e = {span(weights.W_e_weight), span(weights.W_e_bias)};
  weights.view.edge_embedding = {
      {span(weights.edge_embedding_weight), span(weights.edge_embedding_bias)},
      {span(weights.edge_norm_weight), span(weights.edge_norm_bias)}};
  weights.view.positional_encoding =
      {span(weights.positional_weight), span(weights.positional_bias)};
  weights.view.layer_count = weights.layers.size();
  return weights;
}

hiko_m::Mpnn64Descriptor manifest_descriptor() {
  const auto& manifest = hiko_w::default_mpnn_d64_manifest();
  return {manifest.hidden_dimension,
          manifest.neighbor_count,
          manifest.rbf_count,
          manifest.layer_count,
          manifest.message_scale};
}

void test_synthetic_scaffold_edge_workspace_outputs() {
  const hiko_m::Mpnn64Descriptor descriptor = manifest_descriptor();
  if (descriptor.hidden_dimension != 64 || descriptor.neighbor_count != 64 ||
      descriptor.rbf_count != 16 || descriptor.layer_count != 3) {
    fail("manifest-backed MPNN architecture metadata mismatch");
  }

  constexpr std::size_t kResidues = 2;
  const std::array<float, kResidues * 5 * 3> coordinates = {
      0.0F, 0.0F, 0.0F,  // residue 0 N
      1.0F, 0.0F, 0.0F,  // residue 0 CA
      2.0F, 0.0F, 0.0F,  // residue 0 C
      2.0F, 1.0F, 0.0F,  // residue 0 O
      1.0F, 1.0F, 0.0F,  // residue 0 CB
      0.0F, 0.0F, 1.0F,  // residue 1 N
      1.0F, 0.0F, 1.0F,  // residue 1 CA
      2.0F, 0.0F, 1.0F,  // residue 1 C
      2.0F, 1.0F, 1.0F,  // residue 1 O
      1.0F, 1.0F, 1.0F,  // residue 1 CB
  };
  std::array<hiko_u::AtomSource, kResidues * 5> atom_sources{};
  atom_sources.fill(hiko_u::AtomSource::Observed);

  const hiko_d::Mpnn64MemoryPlan plan{kResidues,
                                   descriptor.hidden_dimension,
                                   descriptor.neighbor_count,
                                   descriptor.rbf_count,
                                   descriptor.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  SyntheticWeights weights = make_weights(descriptor);
  std::vector<float> embeddings(kResidues * descriptor.hidden_dimension, -1.0F);

  const hiko_u::Status status = hiko_m::mpnn64_forward_scalar(
      {coordinates.data(), atom_sources.data(), kResidues, descriptor,
       &weights.view, &workspace.view},
      {embeddings.data(), kResidues, descriptor.hidden_dimension});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("MPNN scalar forward returned non-ok for synthetic scaffold fixture");
  }

  if (workspace.residue_features.size() !=
      hiko_d::mpnn64_neighbor_edge_feature_count(plan)) {
    fail("MPNN workspace must reserve real 416-d edge input rows");
  }
  if (workspace.neighbor_indices[0] != 0 || workspace.neighbor_indices[1] != 1 ||
      workspace.neighbor_indices[descriptor.neighbor_count] != 1 ||
      workspace.neighbor_indices[descriptor.neighbor_count + 1] != 0) {
    fail("MPNN KNN graph should include self first and tie by index");
  }
  if (workspace.neighbor_indices[2] != -1 ||
      workspace.neighbor_indices[descriptor.neighbor_count + 2] != -1) {
    fail("MPNN KNN graph should pad k>N slots with -1");
  }
  for (float value : embeddings) {
    if (!nearly_equal(value, 0.0F)) {
      fail("zero synthetic message weights should keep embeddings at zero");
    }
  }
}

void test_workspace_capacity_is_validated() {
  const hiko_m::Mpnn64Descriptor descriptor = manifest_descriptor();
  constexpr std::size_t kResidues = 1;
  std::array<float, kResidues * 5 * 3> coordinates{};
  std::array<hiko_u::AtomSource, kResidues * 5> atom_sources{};
  atom_sources.fill(hiko_u::AtomSource::Observed);
  const hiko_d::Mpnn64MemoryPlan plan{0,
                                   descriptor.hidden_dimension,
                                   descriptor.neighbor_count,
                                   descriptor.rbf_count,
                                   descriptor.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  SyntheticWeights weights = make_weights(descriptor);
  std::vector<float> embeddings(kResidues * descriptor.hidden_dimension, 0.0F);
  const hiko_u::Status status = hiko_m::mpnn64_forward_scalar(
      {coordinates.data(), atom_sources.data(), kResidues, descriptor,
       &weights.view, &workspace.view},
      {embeddings.data(), kResidues, descriptor.hidden_dimension});
  if (status.code != hiko_u::StatusCode::FailedPrecondition) {
    fail("undersized MPNN workspace must return FailedPrecondition");
  }
}

void test_real_package_tiny_structure_golden() {
  const hiko_m::Mpnn64Descriptor descriptor = manifest_descriptor();
  constexpr std::size_t kResidues = 3;
  const std::array<float, kResidues * 5 * 3> coordinates = {
      0.0F, 0.0F, 0.0F,  1.0F, 0.0F, 0.0F,  2.0F, 0.0F, 0.0F,
      2.0F, 1.0F, 0.0F,  1.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.5F,  1.0F, 0.0F, 1.5F,  2.0F, 0.0F, 1.5F,
      2.0F, 1.0F, 1.5F,  1.0F, 1.0F, 1.5F,
      0.0F, 0.0F, 3.0F,  1.0F, 0.0F, 3.0F,  2.0F, 0.0F, 3.0F,
      2.0F, 1.0F, 3.0F,  1.0F, 1.0F, 3.0F,
  };
  std::array<hiko_u::AtomSource, kResidues * 5> atom_sources{};
  atom_sources.fill(hiko_u::AtomSource::Observed);

  const auto weights_result = hiko_w::default_mpnn_d64();
  if (weights_result.status.code != hiko_u::StatusCode::Ok) {
    fail("default Hikoboshi-MPNN-64 weights must prepare for real smoke");
  }
  const auto* prepared =
      static_cast<const hiko_d::Mpnn64Weights*>(weights_result.value.opaque);
  if (prepared == nullptr) {
    fail("default Hikoboshi-MPNN-64 opaque state must be prepared weights");
  }

  const hiko_d::Mpnn64MemoryPlan plan{kResidues,
                                   descriptor.hidden_dimension,
                                   descriptor.neighbor_count,
                                   descriptor.rbf_count,
                                   descriptor.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  std::vector<float> first(kResidues * descriptor.hidden_dimension, 0.0F);
  std::vector<float> second(kResidues * descriptor.hidden_dimension, 0.0F);

  hiko_u::Status status = hiko_m::mpnn64_forward_scalar(
      {coordinates.data(), atom_sources.data(), kResidues, descriptor, prepared,
       &workspace.view},
      {first.data(), kResidues, descriptor.hidden_dimension});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("real Hikoboshi-MPNN-64 scalar forward returned non-ok");
  }
  status = hiko_m::mpnn64_forward_scalar(
      {coordinates.data(), atom_sources.data(), kResidues, descriptor, prepared,
       &workspace.view},
      {second.data(), kResidues, descriptor.hidden_dimension});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("real Hikoboshi-MPNN-64 scalar forward returned non-ok on reuse");
  }

  bool saw_nonzero = false;
  for (std::size_t i = 0; i < first.size(); ++i) {
    if (!std::isfinite(first[i]) || !std::isfinite(second[i])) {
      fail("real Hikoboshi-MPNN-64 embeddings must be finite");
    }
    if (!nearly_equal(first[i], second[i], 1.0e-6F)) {
      fail("real Hikoboshi-MPNN-64 embeddings must be deterministic");
    }
    saw_nonzero = saw_nonzero || std::fabs(first[i]) > 1.0e-6F;
  }
  if (!saw_nonzero) {
    fail("real Hikoboshi-MPNN-64 embeddings should not be all zero");
  }

  // Real-package golden generated by running hiko_w::default_mpnn_d64() through
  // mpnn64_forward_scalar on the 3-residue structure above, after the MF
  // wave (mf1 KNN+vCb policy, mf2 distance floor, mf3 W_e GEMM accumulation
  // order, and per-residue mask gate). The selected-value and aggregate
  // checks below use tight numerical tolerances so the test remains portable
  // across conforming C++ compilers and CPU architectures.
  struct ExpectedValue {
    std::size_t index;
    float value;
  };
  constexpr ExpectedValue kExpectedValues[] = {
      {0, -0.5113158226F},  {1, 0.5963224769F},
      {2, 0.1502234489F},   {3, 0.4265324175F},
      {63, -0.4969113171F}, {64, -0.4483822882F},
      {65, 0.5867809653F},  {127, -0.5039344430F},
      {128, -0.1966642290F}, {191, -0.3205783367F},
  };
  for (const ExpectedValue& expected : kExpectedValues) {
    if (!nearly_equal(first[expected.index], expected.value, 2.0e-6F)) {
      fail("real Hikoboshi-MPNN-64 selected embedding golden drifted");
    }
  }
  double sum = 0.0;
  for (float value : first) {
    sum += value;
  }
  constexpr double kExpectedEmbeddingSum = -1.718477795;
  if (std::fabs(sum - kExpectedEmbeddingSum) > 5.0e-6) {
    std::fprintf(stderr,
                 "mpnn_module_goldens: embedding sum %.12f differs from "
                 "%.12f by %.12g\n",
                 sum, kExpectedEmbeddingSum,
                 std::fabs(sum - kExpectedEmbeddingSum));
    std::exit(1);
  }
}

}  // namespace

int main() {
  // Applicable parity mode: strict.
  //
  // Force the strict path regardless of the build-time
  // `hikoboshi_gemm_parity_mode` default so these numerical goldens continue
  // to verify the strict reduction-tree contract.
  setenv("HIKOBOSHI_GEMM_PARITY_MODE", "strict", 1);
  test_synthetic_scaffold_edge_workspace_outputs();
  test_workspace_capacity_is_validated();
  test_real_package_tiny_structure_golden();
  return 0;
}

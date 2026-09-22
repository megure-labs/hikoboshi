#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/algorithms/detail/mpnn_workspace_storage.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string_view>

namespace hiko_m = hikoboshi::modules;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* detail) {
  std::fprintf(stderr, "mpnn_message_passing_tests: %s\n", detail);
  std::exit(1);
}

bool nearly_equal(float a, float b, float tolerance = 2.0e-5F) {
  return std::fabs(a - b) <= tolerance;
}

hiko_u::Span<const float> span(const std::vector<float>& values) {
  return {values.data(), values.size()};
}

float fast_erf(float value) {
  float sign = 1.0F;
  if (value < 0.0F) {
    sign = -1.0F;
    value = -value;
  }
  if (value > 4.0F) {
    return sign;
  }
  constexpr float kA1 = 0.254829592F;
  constexpr float kA2 = -0.284496736F;
  constexpr float kA3 = 1.421413741F;
  constexpr float kA4 = -1.453152027F;
  constexpr float kA5 = 1.061405429F;
  constexpr float kP = 0.3275911F;
  const float t = 1.0F / (1.0F + kP * value);
  const float polynomial =
      (((((kA5 * t + kA4) * t) + kA3) * t + kA2) * t + kA1) * t;
  return sign * (1.0F - polynomial * std::exp(-(value * value)));
}

float gelu(float value) {
  constexpr float kOneOverSqrtTwo = 0.7071067811865475F;
  return 0.5F * value * (1.0F + fast_erf(value * kOneOverSqrtTwo));
}

std::vector<float> layer_norm(const std::vector<float>& input) {
  float mean = 0.0F;
  for (float value : input) {
    mean += value;
  }
  mean /= static_cast<float>(input.size());
  float variance = 0.0F;
  for (float value : input) {
    const float delta = value - mean;
    variance += delta * delta;
  }
  variance /= static_cast<float>(input.size());
  const float inv_std =
      1.0F / std::sqrt(variance + hiko_d::kMpnn64LayerNormEpsilon);
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = (input[i] - mean) * inv_std;
  }
  return output;
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
  std::vector<float> W_e_weight;
  std::vector<float> W_e_bias;
  std::vector<float> edge_embedding_weight;
  std::vector<float> edge_norm_weight;
  std::vector<float> edge_norm_bias;
  std::vector<float> positional_weight;
  std::vector<float> positional_bias;
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
  std::array<hiko_d::Mpnn64LayerWeights, 3> layers{};
  hiko_d::Mpnn64Weights view{};
};

void set_identity(std::vector<float>& weight,
                  std::size_t rows,
                  std::size_t cols) {
  for (std::size_t i = 0; i < rows && i < cols; ++i) {
    weight[i * cols + i] = 1.0F;
  }
}

SyntheticWeights make_weights(const std::vector<float>& edge_pattern,
                              const std::vector<float>& ffn_bias) {
  SyntheticWeights weights{};
  constexpr std::size_t H = hiko_d::kMpnn64HiddenDimension;
  constexpr std::size_t E = hiko_d::kMpnn64EdgeFeatureCount;
  constexpr std::size_t I = hiko_d::kMpnn64MessageInputDimension;
  constexpr std::size_t F = hiko_d::kMpnn64FfnHiddenDimension;

  weights.W_e_weight.resize(H * H, 0.0F);
  weights.W_e_bias.resize(H, 0.0F);
  set_identity(weights.W_e_weight, H, H);
  weights.edge_embedding_weight.resize(H * E, 0.0F);
  weights.edge_norm_weight.resize(H, 0.0F);
  weights.edge_norm_bias = edge_pattern;
  weights.positional_weight.resize(hiko_d::kMpnn64PositionalFeatureCount *
                                       hiko_d::kMpnn64PositionalClassCount,
                                   0.0F);
  weights.positional_bias.resize(hiko_d::kMpnn64PositionalFeatureCount, 0.0F);

  weights.W1_weight.resize(H * I, 0.0F);
  weights.W11_weight.resize(H * I, 0.0F);
  for (std::size_t d = 0; d < H; ++d) {
    weights.W1_weight[d * I + H + d] = 1.0F;
    weights.W11_weight[d * I + H + d] = 1.0F;
  }
  weights.W1_bias.resize(H, 0.0F);
  weights.W11_bias.resize(H, 0.0F);
  weights.W12_weight.resize(H * H, 0.0F);
  weights.W13_weight.resize(H * H, 0.0F);
  weights.W2_weight.resize(H * H, 0.0F);
  weights.W3_weight.resize(H * H, 0.0F);
  set_identity(weights.W12_weight, H, H);
  set_identity(weights.W13_weight, H, H);
  set_identity(weights.W2_weight, H, H);
  set_identity(weights.W3_weight, H, H);
  weights.W12_bias.resize(H, 0.0F);
  weights.W13_bias.resize(H, 0.0F);
  weights.W2_bias.resize(H, 0.0F);
  weights.W3_bias.resize(H, 0.0F);

  weights.ffn_W_in_weight.resize(F * H, 0.0F);
  weights.ffn_W_in_bias = ffn_bias;
  weights.ffn_W_out_weight.resize(H * F, 0.0F);
  for (std::size_t d = 0; d < H; ++d) {
    weights.ffn_W_out_weight[d * F + d] = 1.0F;
  }
  weights.ffn_W_out_bias.resize(H, 0.0F);
  weights.norm1_weight.resize(H, 1.0F);
  weights.norm1_bias.resize(H, 0.0F);
  weights.norm2_weight.resize(H, 1.0F);
  weights.norm2_bias.resize(H, 0.0F);
  weights.norm3_weight.resize(H, 1.0F);
  weights.norm3_bias.resize(H, 0.0F);

  hiko_d::Mpnn64LayerWeights& layer = weights.layers[0];
  layer.W1 = {span(weights.W1_weight), span(weights.W1_bias)};
  layer.W11 = {span(weights.W11_weight), span(weights.W11_bias)};
  layer.W12 = {span(weights.W12_weight), span(weights.W12_bias)};
  layer.W13 = {span(weights.W13_weight), span(weights.W13_bias)};
  layer.W2 = {span(weights.W2_weight), span(weights.W2_bias)};
  layer.W3 = {span(weights.W3_weight), span(weights.W3_bias)};
  layer.ffn = {{span(weights.ffn_W_in_weight), span(weights.ffn_W_in_bias)},
               {span(weights.ffn_W_out_weight), span(weights.ffn_W_out_bias)}};
  layer.norm1 = {span(weights.norm1_weight), span(weights.norm1_bias)};
  layer.norm2 = {span(weights.norm2_weight), span(weights.norm2_bias)};
  layer.norm3 = {span(weights.norm3_weight), span(weights.norm3_bias)};

  weights.view.W_e = {span(weights.W_e_weight), span(weights.W_e_bias)};
  weights.view.edge_embedding = {
      {span(weights.edge_embedding_weight), {nullptr, 0}},
      {span(weights.edge_norm_weight), span(weights.edge_norm_bias)}};
  weights.view.positional_encoding =
      {span(weights.positional_weight), span(weights.positional_bias)};
  weights.layers.fill(layer);
  weights.view.layers = weights.layers.data();
  weights.view.layer_count = weights.layers.size();
  return weights;
}

hiko_u::Status run_forward(SyntheticWeights& weights,
                        hiko_d::Mpnn64Workspace& workspace,
                        const hiko_m::Mpnn64Descriptor& descriptor,
                        std::vector<float>& embeddings,
                        hiko_m::Mpnn64DebugCapture* capture = nullptr
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
                        , hiko_m::Mpnn64IntermediateDumper dumper = {}
#endif
                        ) {
  constexpr std::size_t kResidues = 1;
  std::array<float, kResidues * hiko_d::kMpnn64AtomCount *
                        hiko_d::kMpnn64AxisCount>
      coordinates{};
  std::array<hiko_u::AtomSource, kResidues * hiko_d::kMpnn64AtomCount> atom_sources{};
  atom_sources.fill(hiko_u::AtomSource::Observed);
  hiko_m::Mpnn64ForwardRequest request{
      coordinates.data(), atom_sources.data(), kResidues, descriptor,
      &weights.view, &workspace};
  request.debug_capture = capture;
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  request.intermediate_dumper = dumper;
#endif
  return hiko_m::mpnn64_forward_scalar(
      request, {embeddings.data(), kResidues, descriptor.hidden_dimension});
}

#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
struct DumpCapture {
  std::size_t edge_updates = 0;
  std::vector<float> last_edges;
};
void capture_dump_edges(const hiko_m::Mpnn64IntermediateTensor& tensor,
                        void* user_data) noexcept {
  constexpr std::string_view suffix{".post_edge_update_residual_norm3"};
  const std::string_view name{tensor.name};
  if (name.size() < suffix.size() ||
      name.substr(name.size() - suffix.size()) != suffix)
    return;
  auto& capture = *static_cast<DumpCapture*>(user_data);
  ++capture.edge_updates;
  std::size_t count = 1;
  for (std::size_t i = 0; i < tensor.rank; ++i) count *= tensor.shape[i];
  const auto* values = static_cast<const float*>(tensor.data);
  capture.last_edges.assign(values, values + count);
}
#endif

void test_layer_algebra_message_scaling_and_invalid_neighbor_mask(
    std::size_t layer_count, std::size_t neighbor_count, bool compact) {
  constexpr std::size_t H = hiko_d::kMpnn64HiddenDimension;
  constexpr std::size_t F = hiko_d::kMpnn64FfnHiddenDimension;
  std::vector<float> edge_pattern(H);
  std::vector<float> ffn_bias(F, 0.0F);
  for (std::size_t d = 0; d < H; ++d) {
    edge_pattern[d] = 0.05F + 0.01F * static_cast<float>(d);
    ffn_bias[d] = -0.2F + 0.007F * static_cast<float>(d);
  }
  hiko_m::Mpnn64Descriptor descriptor{H, neighbor_count, 16, layer_count, 2.0F};
  const hiko_d::Mpnn64MemoryPlan plan{1, H, neighbor_count, 16, layer_count};
  OwnedWorkspace separate = make_workspace(plan);
  hikoboshi::algorithms::detail::Mpnn64WorkspaceStorage shared;
  shared.prepare(plan);
  auto& workspace = compact ? shared.workspace : separate.view;
  SyntheticWeights weights = make_weights(edge_pattern, ffn_bias);
  std::vector<float> embeddings(H, 0.0F), initial_edges(neighbor_count * H);
  std::array<std::vector<float>, 3> node_captures;
  for (auto& values : node_captures) values.resize(H);
  std::vector<float> final_capture(H);
  hiko_m::Mpnn64DebugCapture capture{
      initial_edges.data(), node_captures[0].data(), node_captures[1].data(),
      node_captures[2].data(), final_capture.data()};
  if (run_forward(weights, workspace, descriptor, embeddings, &capture).code !=
      hiko_u::StatusCode::Ok) fail("synthetic message-passing forward returned non-ok");
  if (workspace.neighbor_indices.data[0] != 0 ||
      (neighbor_count > 1 && workspace.neighbor_indices.data[1] != -1))
    fail("KNN must expose self edge and mask any padded edge");

  // Independent scalar reference: only an earlier layer's updated edges can
  // affect a node embedding. Also retain the hypothetical final edges to check
  // the diagnostic path, which still promises all intermediate tensors.
  std::vector<float> node(H, 0.0F), edge = edge_pattern;
  std::vector<float> expected_workspace_edge = edge;
  for (std::size_t layer = 0; layer < layer_count; ++layer) {
    std::vector<float> message(H);
    for (std::size_t d = 0; d < H; ++d) {
      message[d] = gelu(gelu(edge[d]));
      node[d] += message[d] / descriptor.message_scale;
    }
    node = layer_norm(node);
    for (std::size_t d = 0; d < H; ++d) node[d] += gelu(ffn_bias[d]);
    node = layer_norm(node);
    for (std::size_t d = 0; d < H; ++d)
      if (!nearly_equal(node_captures[layer][d], node[d]))
        fail("debug capture must preserve every node layer boundary");
    expected_workspace_edge = edge;
    for (std::size_t d = 0; d < H; ++d) edge[d] += message[d];
    edge = layer_norm(edge);
  }
  for (std::size_t d = 0; d < H; ++d) {
    if (!nearly_equal(embeddings[d], node[d]))
      fail("embedding must preserve multilayer message/FFN algebra");
    if (!nearly_equal(workspace.edge_state.data[d], expected_workspace_edge[d]))
      fail("normal forward must stop edge updates after the penultimate layer");
    if (neighbor_count > 1 && !nearly_equal(workspace.edge_state.data[H+d], edge_pattern[d]))
      fail("invalid padded edge must remain unchanged");
    if (!nearly_equal(initial_edges[d], edge_pattern[d]))
      fail("initial edge debug capture must remain unchanged");
  }
  if (std::memcmp(final_capture.data(), embeddings.data(), H * sizeof(float)))
    fail("final debug capture must equal returned embeddings");
  std::vector<float> repeated(H);
  if (run_forward(weights, workspace, descriptor, repeated).code != hiko_u::StatusCode::Ok ||
      std::memcmp(repeated.data(), embeddings.data(), H * sizeof(float)))
    fail("workspace reuse must not consume stale terminal edge/scratch state");
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  DumpCapture dumped;
  std::vector<float> diagnostic(H);
  if (run_forward(weights, workspace, descriptor, diagnostic, nullptr,
                  {capture_dump_edges, &dumped}).code != hiko_u::StatusCode::Ok)
    fail("diagnostic forward returned non-ok");
  if (dumped.edge_updates != layer_count)
    fail("diagnostic forward must still emit every edge update");
  for (std::size_t d = 0; d < H; ++d) {
    if (!nearly_equal(diagnostic[d], embeddings[d]))
      fail("diagnostic path must preserve node embeddings");
    if (layer_count && !nearly_equal(dumped.last_edges[d], edge[d]))
      fail("diagnostic terminal edge tensor must retain its original algebra");
  }
#endif
}

void test_layer_shape_validation_rejects_missing_tensor() {
  constexpr std::size_t H = hiko_d::kMpnn64HiddenDimension;
  constexpr std::size_t F = hiko_d::kMpnn64FfnHiddenDimension;
  std::vector<float> edge_pattern(H, 0.0F);
  std::vector<float> ffn_bias(F, 0.0F);
  SyntheticWeights weights = make_weights(edge_pattern, ffn_bias);
  weights.layers[0].W13.weight = {weights.W13_weight.data(),
                                  weights.W13_weight.size() - 1U};

  hiko_m::Mpnn64Descriptor descriptor{H, 1, 16, 1, 1.0F};
  const hiko_d::Mpnn64MemoryPlan plan{1, H, descriptor.neighbor_count,
                                   descriptor.rbf_count,
                                   descriptor.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  std::vector<float> embeddings(H, 0.0F);
  const hiko_u::Status status =
      run_forward(weights, workspace.view, descriptor, embeddings);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("wrong W13 tensor shape must be rejected");
  }
}

}  // namespace

int main() {
  for (std::size_t layers : {0U, 1U, 2U, 3U}) {
    for (std::size_t neighbors : {1U, 2U})
      for (bool compact : {false, true})
        test_layer_algebra_message_scaling_and_invalid_neighbor_mask(layers, neighbors, compact);
  }
  test_layer_shape_validation_rejects_missing_tensor();
  return 0;
}

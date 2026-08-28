#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/modules/mpnn/proteinmpnn_encoder.hpp>
#include <hikoboshi/primitives/compute/gelu.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace hiko_m = hikoboshi::modules::mpnn;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace pmp = hikoboshi::universal::detail;

namespace {

constexpr std::size_t kResidues = 5;
constexpr std::size_t kHidden = pmp::kProteinMpnnV48020Hidden;
constexpr std::size_t kNeighbors = pmp::kProteinMpnnV48020KNeighbors;
constexpr std::size_t kLayers = 1;
constexpr float kMessageScale = pmp::kProteinMpnnV48020MessageScale;
constexpr float kTolerance = 1.5e-5F;

static_assert(kHidden == 128, "ProteinMPNN v48 hidden dimension changed");
static_assert(kNeighbors == 48, "ProteinMPNN v48 neighbor count changed");

[[noreturn]] void fail(const std::string& detail) {
  std::cerr << "proteinmpnn_encoder_layer_parity_test: " << detail << "\n";
  std::exit(1);
}

hiko_u::Span<const float> span(const std::vector<float>& values) noexcept {
  return {values.data(), values.size()};
}

std::uint64_t splitmix(std::size_t index, std::uint64_t salt) noexcept {
  std::uint64_t x = 0x9E3779B97F4A7C15ull;
  x += (static_cast<std::uint64_t>(index) + 1ull) * 0xBF58476D1CE4E5B9ull;
  x += (salt + 1ull) * 0x94D049BB133111EBull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return x;
}

float pseudo_value(std::size_t index, std::uint64_t salt, float scale) noexcept {
  const std::uint64_t bits = splitmix(index, salt) >> 40;
  const float unit =
      static_cast<float>(bits) * (1.0F / 8388608.0F) - 1.0F;
  return unit * scale;
}

void fill_pseudo(std::vector<float>& values,
                 std::uint64_t salt,
                 float scale) noexcept {
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = pseudo_value(index, salt, scale);
  }
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
  owned.residue_features.resize(
      std::max(hiko_d::mpnn64_residue_feature_count(plan),
               hiko_d::mpnn64_neighbor_slot_count(plan) * 3 * plan.hidden_dimension));
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

struct LayerStorage {
  std::vector<float> W1_weight;
  std::vector<float> W1_bias;
  std::vector<float> W2_weight;
  std::vector<float> W2_bias;
  std::vector<float> W3_weight;
  std::vector<float> W3_bias;
  std::vector<float> W11_weight;
  std::vector<float> W11_bias;
  std::vector<float> W12_weight;
  std::vector<float> W12_bias;
  std::vector<float> W13_weight;
  std::vector<float> W13_bias;
  std::vector<float> dense_W_in_weight;
  std::vector<float> dense_W_in_bias;
  std::vector<float> dense_W_out_weight;
  std::vector<float> dense_W_out_bias;
  std::vector<float> norm1_weight;
  std::vector<float> norm1_bias;
  std::vector<float> norm2_weight;
  std::vector<float> norm2_bias;
  std::vector<float> norm3_weight;
  std::vector<float> norm3_bias;
};

struct Fixture {
  std::vector<float> input_nodes;
  std::vector<float> input_edges;
  std::vector<std::int32_t> edge_indices;
  std::vector<float> mask_v;
  std::vector<float> mask_attend;
  LayerStorage layer;
  pmp::ProteinMpnnV48020Weights weights{};
};

void fill_norm(std::vector<float>& weight,
               std::vector<float>& bias,
               std::uint64_t salt) noexcept {
  weight.resize(kHidden);
  bias.resize(kHidden);
  for (std::size_t index = 0; index < kHidden; ++index) {
    weight[index] = 0.9F + pseudo_value(index, salt, 0.08F);
    bias[index] = pseudo_value(index, salt + 17, 0.03F);
  }
}

Fixture make_fixture() {
  constexpr std::size_t message_input = 3 * kHidden;
  constexpr std::size_t ffn_hidden = 4 * kHidden;
  constexpr std::size_t slots = kResidues * kNeighbors;
  Fixture fixture{};
  fixture.input_nodes.resize(kResidues * kHidden);
  fixture.input_edges.resize(slots * kHidden);
  fixture.edge_indices.resize(slots);
  fixture.mask_v = {1.0F, 1.0F, 0.0F, 1.0F, 1.0F};
  fixture.mask_attend.resize(slots, 0.0F);
  fill_pseudo(fixture.input_nodes, 1, 0.35F);
  fill_pseudo(fixture.input_edges, 2, 0.25F);

  for (std::size_t residue = 0; residue < kResidues; ++residue) {
    for (std::size_t slot = 0; slot < kNeighbors; ++slot) {
      const std::size_t index = residue * kNeighbors + slot;
      if (slot < kResidues) {
        const std::int32_t neighbor =
            static_cast<std::int32_t>((residue + slot) % kResidues);
        fixture.edge_indices[index] = neighbor;
        fixture.mask_attend[index] =
            fixture.mask_v[residue] * fixture.mask_v[static_cast<std::size_t>(neighbor)];
      } else {
        fixture.edge_indices[index] = -1;
      }
    }
  }

  LayerStorage& layer = fixture.layer;
  layer.W1_weight.resize(kHidden * message_input);
  layer.W1_bias.resize(kHidden);
  layer.W2_weight.resize(kHidden * kHidden);
  layer.W2_bias.resize(kHidden);
  layer.W3_weight.resize(kHidden * kHidden);
  layer.W3_bias.resize(kHidden);
  layer.W11_weight.resize(kHidden * message_input);
  layer.W11_bias.resize(kHidden);
  layer.W12_weight.resize(kHidden * kHidden);
  layer.W12_bias.resize(kHidden);
  layer.W13_weight.resize(kHidden * kHidden);
  layer.W13_bias.resize(kHidden);
  layer.dense_W_in_weight.resize(ffn_hidden * kHidden);
  layer.dense_W_in_bias.resize(ffn_hidden);
  layer.dense_W_out_weight.resize(kHidden * ffn_hidden);
  layer.dense_W_out_bias.resize(kHidden);

  fill_pseudo(layer.W1_weight, 10, 0.018F);
  fill_pseudo(layer.W1_bias, 11, 0.012F);
  fill_pseudo(layer.W2_weight, 12, 0.015F);
  fill_pseudo(layer.W2_bias, 13, 0.010F);
  fill_pseudo(layer.W3_weight, 14, 0.015F);
  fill_pseudo(layer.W3_bias, 15, 0.010F);
  fill_pseudo(layer.W11_weight, 16, 0.018F);
  fill_pseudo(layer.W11_bias, 17, 0.012F);
  fill_pseudo(layer.W12_weight, 18, 0.015F);
  fill_pseudo(layer.W12_bias, 19, 0.010F);
  fill_pseudo(layer.W13_weight, 20, 0.015F);
  fill_pseudo(layer.W13_bias, 21, 0.010F);
  fill_pseudo(layer.dense_W_in_weight, 22, 0.014F);
  fill_pseudo(layer.dense_W_in_bias, 23, 0.010F);
  fill_pseudo(layer.dense_W_out_weight, 24, 0.014F);
  fill_pseudo(layer.dense_W_out_bias, 25, 0.010F);
  fill_norm(layer.norm1_weight, layer.norm1_bias, 30);
  fill_norm(layer.norm2_weight, layer.norm2_bias, 40);
  fill_norm(layer.norm3_weight, layer.norm3_bias, 50);

  pmp::ProteinMpnnV48020EncoderLayerWeights& view =
      fixture.weights.encoder_layers[0];
  view.W1 = {span(layer.W1_weight), span(layer.W1_bias)};
  view.W2 = {span(layer.W2_weight), span(layer.W2_bias)};
  view.W3 = {span(layer.W3_weight), span(layer.W3_bias)};
  view.W11 = {span(layer.W11_weight), span(layer.W11_bias)};
  view.W12 = {span(layer.W12_weight), span(layer.W12_bias)};
  view.W13 = {span(layer.W13_weight), span(layer.W13_bias)};
  view.dense = {{span(layer.dense_W_in_weight), span(layer.dense_W_in_bias)},
                {span(layer.dense_W_out_weight), span(layer.dense_W_out_bias)}};
  view.norm1 = {span(layer.norm1_weight), span(layer.norm1_bias)};
  view.norm2 = {span(layer.norm2_weight), span(layer.norm2_bias)};
  view.norm3 = {span(layer.norm3_weight), span(layer.norm3_bias)};
  return fixture;
}

bool valid_neighbor(std::int32_t neighbor) noexcept {
  return neighbor >= 0 && static_cast<std::size_t>(neighbor) < kResidues;
}

float gelu(float value) noexcept {
  return hikoboshi::primitives::compute::detail::gelu_exact_scalar(value);
}

void linear_nt(const std::vector<float>& input,
               const std::vector<float>& weight,
               const std::vector<float>& bias,
               std::size_t row_count,
               std::size_t output_dimension,
               std::size_t input_dimension,
               std::vector<float>& output) {
  output.assign(row_count * output_dimension, 0.0F);
  for (std::size_t row = 0; row < row_count; ++row) {
    for (std::size_t out = 0; out < output_dimension; ++out) {
      float acc = 0.0F;
      for (std::size_t dim = 0; dim < input_dimension; ++dim) {
        acc += input[row * input_dimension + dim] *
               weight[out * input_dimension + dim];
      }
      output[row * output_dimension + out] = acc + bias[out];
    }
  }
}

void gelu_inplace(std::vector<float>& values) noexcept {
  for (float& value : values) {
    value = gelu(value);
  }
}

std::vector<float> build_message_inputs(const std::vector<float>& nodes,
                                        const std::vector<float>& edges,
                                        const std::vector<std::int32_t>& idx) {
  constexpr std::size_t input_dim = 3 * kHidden;
  std::vector<float> input(kResidues * kNeighbors * input_dim, 0.0F);
  for (std::size_t residue = 0; residue < kResidues; ++residue) {
    for (std::size_t slot = 0; slot < kNeighbors; ++slot) {
      const std::size_t edge_slot = residue * kNeighbors + slot;
      float* row = input.data() + edge_slot * input_dim;
      std::memcpy(row, nodes.data() + residue * kHidden, kHidden * sizeof(float));
      std::memcpy(row + kHidden, edges.data() + edge_slot * kHidden,
                  kHidden * sizeof(float));
      const std::int32_t neighbor = idx[edge_slot];
      if (valid_neighbor(neighbor)) {
        std::memcpy(row + 2 * kHidden,
                    nodes.data() + static_cast<std::size_t>(neighbor) * kHidden,
                    kHidden * sizeof(float));
      }
    }
  }
  return input;
}

void layer_norm_row(const float* input,
                    const std::vector<float>& gamma,
                    const std::vector<float>& beta,
                    float* output) {
  float mean = 0.0F;
  float m2 = 0.0F;
  for (std::size_t dim = 0; dim < kHidden; ++dim) {
    const float value = input[dim];
    const float count = static_cast<float>(dim + 1);
    const float delta = value - mean;
    mean += delta / count;
    const float delta2 = value - mean;
    m2 += delta * delta2;
  }
  const float variance = m2 / static_cast<float>(kHidden);
  const float inv_std =
      1.0F / std::sqrt(variance + hiko_d::kMpnn64LayerNormEpsilon);
  for (std::size_t dim = 0; dim < kHidden; ++dim) {
    output[dim] = ((input[dim] - mean) * inv_std) * gamma[dim] + beta[dim];
  }
}

void layer_norm_residual(const std::vector<float>& update,
                         const std::vector<float>& residual,
                         const std::vector<float>& gamma,
                         const std::vector<float>& beta,
                         std::size_t row_count,
                         std::vector<float>& output) {
  output.resize(row_count * kHidden);
  std::vector<float> row(kHidden);
  for (std::size_t r = 0; r < row_count; ++r) {
    for (std::size_t dim = 0; dim < kHidden; ++dim) {
      row[dim] = update[r * kHidden + dim] + residual[r * kHidden + dim];
    }
    layer_norm_row(row.data(), gamma, beta, output.data() + r * kHidden);
  }
}

void reference_layer(const Fixture& fixture,
                     std::vector<float>& nodes,
                     std::vector<float>& edges) {
  constexpr std::size_t slots = kResidues * kNeighbors;
  constexpr std::size_t input_dim = 3 * kHidden;
  constexpr std::size_t ffn_hidden = 4 * kHidden;
  const LayerStorage& layer = fixture.layer;

  std::vector<float> input =
      build_message_inputs(nodes, edges, fixture.edge_indices);
  std::vector<float> message;
  std::vector<float> projected;
  linear_nt(input, layer.W1_weight, layer.W1_bias, slots, kHidden, input_dim,
            message);
  gelu_inplace(message);
  linear_nt(message, layer.W2_weight, layer.W2_bias, slots, kHidden, kHidden,
            projected);
  gelu_inplace(projected);
  linear_nt(projected, layer.W3_weight, layer.W3_bias, slots, kHidden, kHidden,
            message);

  std::vector<float> node_update(kResidues * kHidden, 0.0F);
  for (std::size_t residue = 0; residue < kResidues; ++residue) {
    for (std::size_t slot = 0; slot < kNeighbors; ++slot) {
      const std::size_t edge_slot = residue * kNeighbors + slot;
      if (!valid_neighbor(fixture.edge_indices[edge_slot])) {
        continue;
      }
      const float alpha = fixture.mask_attend[edge_slot] / kMessageScale;
      for (std::size_t dim = 0; dim < kHidden; ++dim) {
        node_update[residue * kHidden + dim] +=
            alpha * message[edge_slot * kHidden + dim];
      }
    }
  }
  std::vector<float> after_norm1;
  layer_norm_residual(node_update, nodes, layer.norm1_weight, layer.norm1_bias,
                      kResidues, after_norm1);

  std::vector<float> ffn_mid;
  std::vector<float> ffn_out;
  linear_nt(after_norm1, layer.dense_W_in_weight, layer.dense_W_in_bias,
            kResidues, ffn_hidden, kHidden, ffn_mid);
  gelu_inplace(ffn_mid);
  linear_nt(ffn_mid, layer.dense_W_out_weight, layer.dense_W_out_bias,
            kResidues, kHidden, ffn_hidden, ffn_out);
  layer_norm_residual(ffn_out, after_norm1, layer.norm2_weight,
                      layer.norm2_bias, kResidues, nodes);

  for (std::size_t residue = 0; residue < kResidues; ++residue) {
    for (std::size_t dim = 0; dim < kHidden; ++dim) {
      nodes[residue * kHidden + dim] *= fixture.mask_v[residue];
    }
  }

  input = build_message_inputs(nodes, edges, fixture.edge_indices);
  linear_nt(input, layer.W11_weight, layer.W11_bias, slots, kHidden, input_dim,
            message);
  gelu_inplace(message);
  linear_nt(message, layer.W12_weight, layer.W12_bias, slots, kHidden, kHidden,
            projected);
  gelu_inplace(projected);
  linear_nt(projected, layer.W13_weight, layer.W13_bias, slots, kHidden,
            kHidden, message);

  std::vector<float> row(kHidden);
  for (std::size_t slot = 0; slot < slots; ++slot) {
    if (!valid_neighbor(fixture.edge_indices[slot])) {
      continue;
    }
    for (std::size_t dim = 0; dim < kHidden; ++dim) {
      row[dim] = message[slot * kHidden + dim] + edges[slot * kHidden + dim];
    }
    layer_norm_row(row.data(), layer.norm3_weight, layer.norm3_bias,
                   edges.data() + slot * kHidden);
  }
}

struct NativeOutput {
  std::vector<float> nodes;
  std::vector<float> edges;
  std::vector<std::int32_t> edge_indices;
};

NativeOutput run_native(Fixture& fixture) {
  const hiko_d::Mpnn64MemoryPlan plan{kResidues, kHidden, kNeighbors,
                                   pmp::kProteinMpnnV48020RbfCount, kLayers};
  OwnedWorkspace workspace = make_workspace(plan);
  NativeOutput output{};
  output.nodes.resize(kResidues * kHidden);
  output.edges.resize(kResidues * kNeighbors * kHidden);
  output.edge_indices.resize(kResidues * kNeighbors);

  const hiko_m::ProteinMpnnEncoderDescriptor descriptor{
      kHidden, kNeighbors, kLayers, kMessageScale};
  const hiko_m::ProteinMpnnEncoderRequest request{
      fixture.input_nodes.data(),
      fixture.input_edges.data(),
      fixture.edge_indices.data(),
      fixture.mask_v.data(),
      fixture.mask_attend.data(),
      &fixture.weights,
      &workspace.view,
      kResidues,
      descriptor,
  };
  const hiko_m::ProteinMpnnEncoderOutput out{
      output.nodes.data(),
      output.edges.data(),
      output.edge_indices.data(),
      kResidues,
      kHidden,
      kNeighbors,
  };
  const hiko_u::Status status = hiko_m::proteinmpnn_encoder_scalar(request, out);
  if (!status.ok()) {
    fail(std::string("native encoder returned non-ok: ") + status.detail);
  }
  return output;
}

float max_abs_diff(const std::vector<float>& a,
                   const std::vector<float>& b) noexcept {
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < a.size(); ++index) {
    max_abs = std::max(max_abs, std::fabs(a[index] - b[index]));
  }
  return max_abs;
}

void print_float_array(const char* name, const std::vector<float>& values) {
  std::cout << "\"" << name << "\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << std::setprecision(9) << values[index];
  }
  std::cout << "]";
}

void print_int_array(const char* name,
                     const std::vector<std::int32_t>& values) {
  std::cout << "\"" << name << "\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << values[index];
  }
  std::cout << "]";
}

void dump_json(Fixture& fixture) {
  NativeOutput native = run_native(fixture);
  std::cout << "{\"residue_count\":" << kResidues
            << ",\"hidden_dimension\":" << kHidden
            << ",\"neighbor_count\":" << kNeighbors << ",";
  print_float_array("h_v", native.nodes);
  std::cout << ",";
  print_float_array("h_e", native.edges);
  std::cout << ",";
  print_int_array("e_idx", native.edge_indices);
  std::cout << "}\n";
}

void run_parity() {
  Fixture fixture = make_fixture();
  NativeOutput native = run_native(fixture);
  std::vector<float> ref_nodes = fixture.input_nodes;
  std::vector<float> ref_edges = fixture.input_edges;
  reference_layer(fixture, ref_nodes, ref_edges);

  if (native.edge_indices != fixture.edge_indices) {
    fail("E_idx output must retain the input neighbor indices");
  }
  const float node_max_abs = max_abs_diff(native.nodes, ref_nodes);
  const float edge_max_abs = max_abs_diff(native.edges, ref_edges);
  if (node_max_abs > kTolerance || edge_max_abs > kTolerance) {
    fail("native encoder layer drift exceeds tolerance: h_V max_abs=" +
         std::to_string(node_max_abs) + " h_E max_abs=" +
         std::to_string(edge_max_abs));
  }
}

}  // namespace

int main(int argc, char** argv) {
  Fixture fixture = make_fixture();
  if (argc == 2 && std::strcmp(argv[1], "--dump-json") == 0) {
    dump_json(fixture);
    return 0;
  }
  run_parity();
  return 0;
}

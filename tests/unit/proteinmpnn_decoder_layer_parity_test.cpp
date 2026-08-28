#include <hikoboshi/modules/mpnn/proteinmpnn_decoder.hpp>
#include <hikoboshi/primitives/compute/gelu.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>
#include <hikoboshi/universal/span.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace hiko_m = hikoboshi::modules::mpnn;
namespace hiko_p = hikoboshi::primitives::compute;
namespace hiko_s = hikoboshi::universal::detail;
namespace hiko_u = hikoboshi::universal;

namespace {

constexpr float kDefaultTolerance = 1.5e-5F;

void fail(const char* detail) {
  std::fprintf(stderr, "proteinmpnn_decoder_layer_parity_test: %s\n", detail);
  std::exit(1);
}

hiko_u::Span<const float> span(const std::vector<float>& values) {
  return {values.data(), values.size()};
}

float uniform(std::mt19937& rng, float low, float high) {
  std::uniform_real_distribution<float> dist(low, high);
  return dist(rng);
}

void fill_random(std::vector<float>& values,
                 std::mt19937& rng,
                 float low,
                 float high) {
  for (float& value : values) {
    value = uniform(rng, low, high);
  }
}

float parse_tolerance_payload(const std::string& payload) {
  const std::string key = "\"proteinmpnn_layer_abs\"";
  const std::size_t key_pos = payload.find(key);
  if (key_pos == std::string::npos) {
    return kDefaultTolerance;
  }
  const std::size_t colon = payload.find(':', key_pos + key.size());
  if (colon == std::string::npos) {
    return kDefaultTolerance;
  }
  char* end = nullptr;
  const float parsed =
      std::strtof(payload.c_str() + static_cast<std::ptrdiff_t>(colon + 1),
                  &end);
  if (end == payload.c_str() + static_cast<std::ptrdiff_t>(colon + 1) ||
      !std::isfinite(parsed) || parsed <= 0.0F) {
    return kDefaultTolerance;
  }
  return std::min(parsed, kDefaultTolerance);
}

float load_tolerance() {
  std::array<std::string, 3> candidates{
      "bench/numerical_tolerance.json",
      "../bench/numerical_tolerance.json",
      "../../bench/numerical_tolerance.json",
  };
  const char* source_root = std::getenv("HIKOBOSHI_SOURCE_ROOT");
  if (source_root != nullptr && source_root[0] != '\0') {
    candidates[0] = std::string(source_root) + "/bench/numerical_tolerance.json";
  }

  for (const std::string& path : candidates) {
    std::ifstream input(path);
    if (!input) {
      continue;
    }
    const std::string payload((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
    return parse_tolerance_payload(payload);
  }
  return kDefaultTolerance;
}

struct OwnedWorkspace {
  std::vector<float> message_input;
  std::vector<float> message_state;
  std::vector<float> projected_message_state;
  std::vector<float> residue_scratch;
  std::vector<float> ffn_hidden;
  hiko_m::ProteinMpnnDecoderWorkspace view{};
};

OwnedWorkspace make_workspace(const hiko_m::ProteinMpnnDecoderMemoryPlan& plan) {
  OwnedWorkspace owned{};
  owned.message_input.resize(hiko_m::proteinmpnn_decoder_message_input_count(plan));
  owned.message_state.resize(
      hiko_m::proteinmpnn_decoder_neighbor_hidden_count(plan));
  owned.projected_message_state.resize(
      hiko_m::proteinmpnn_decoder_neighbor_hidden_count(plan));
  owned.residue_scratch.resize(
      hiko_m::proteinmpnn_decoder_residue_hidden_count(plan));
  owned.ffn_hidden.resize(hiko_m::proteinmpnn_decoder_ffn_hidden_count(plan));
  owned.view = {
      plan,
      {owned.message_input.data(), owned.message_input.size()},
      {owned.message_state.data(), owned.message_state.size()},
      {owned.projected_message_state.data(),
       owned.projected_message_state.size()},
      {owned.residue_scratch.data(), owned.residue_scratch.size()},
      {owned.ffn_hidden.data(), owned.ffn_hidden.size()},
  };
  return owned;
}

struct SyntheticWeights {
  std::vector<float> W1_weight;
  std::vector<float> W1_bias;
  std::vector<float> W2_weight;
  std::vector<float> W2_bias;
  std::vector<float> W3_weight;
  std::vector<float> W3_bias;
  std::vector<float> dense_W_in_weight;
  std::vector<float> dense_W_in_bias;
  std::vector<float> dense_W_out_weight;
  std::vector<float> dense_W_out_bias;
  std::vector<float> norm1_weight;
  std::vector<float> norm1_bias;
  std::vector<float> norm2_weight;
  std::vector<float> norm2_bias;
  std::vector<float> W_out_weight;
  std::vector<float> W_out_bias;
  std::vector<float> W_s_weight;
  hiko_s::ProteinMpnnV48020DecoderLayerWeights layer{};
  hiko_s::ProteinMpnnV48020LinearWeights W_out{};
  hiko_s::ProteinMpnnV48020EmbeddingWeights W_s{};
};

SyntheticWeights make_weights(std::size_t hidden,
                              std::size_t decoder_input,
                              std::size_t ffn_hidden,
                              std::size_t vocab) {
  std::mt19937 rng(0x02CDEC0Du);
  SyntheticWeights weights{};
  const std::size_t input_dim = hidden + decoder_input;

  weights.W1_weight.resize(hidden * input_dim);
  weights.W1_bias.resize(hidden);
  weights.W2_weight.resize(hidden * hidden);
  weights.W2_bias.resize(hidden);
  weights.W3_weight.resize(hidden * hidden);
  weights.W3_bias.resize(hidden);
  weights.dense_W_in_weight.resize(ffn_hidden * hidden);
  weights.dense_W_in_bias.resize(ffn_hidden);
  weights.dense_W_out_weight.resize(hidden * ffn_hidden);
  weights.dense_W_out_bias.resize(hidden);
  weights.norm1_weight.resize(hidden);
  weights.norm1_bias.resize(hidden);
  weights.norm2_weight.resize(hidden);
  weights.norm2_bias.resize(hidden);
  weights.W_out_weight.resize(vocab * hidden);
  weights.W_out_bias.resize(vocab);
  weights.W_s_weight.resize(vocab * hidden);

  fill_random(weights.W1_weight, rng, -0.035F, 0.035F);
  fill_random(weights.W1_bias, rng, -0.020F, 0.020F);
  fill_random(weights.W2_weight, rng, -0.045F, 0.045F);
  fill_random(weights.W2_bias, rng, -0.020F, 0.020F);
  fill_random(weights.W3_weight, rng, -0.045F, 0.045F);
  fill_random(weights.W3_bias, rng, -0.020F, 0.020F);
  fill_random(weights.dense_W_in_weight, rng, -0.040F, 0.040F);
  fill_random(weights.dense_W_in_bias, rng, -0.020F, 0.020F);
  fill_random(weights.dense_W_out_weight, rng, -0.040F, 0.040F);
  fill_random(weights.dense_W_out_bias, rng, -0.020F, 0.020F);
  fill_random(weights.norm1_weight, rng, 0.85F, 1.15F);
  fill_random(weights.norm1_bias, rng, -0.050F, 0.050F);
  fill_random(weights.norm2_weight, rng, 0.85F, 1.15F);
  fill_random(weights.norm2_bias, rng, -0.050F, 0.050F);
  fill_random(weights.W_out_weight, rng, -0.050F, 0.050F);
  fill_random(weights.W_out_bias, rng, -0.030F, 0.030F);
  fill_random(weights.W_s_weight, rng, -0.10F, 0.10F);

  weights.layer.W1 = {span(weights.W1_weight), span(weights.W1_bias)};
  weights.layer.W2 = {span(weights.W2_weight), span(weights.W2_bias)};
  weights.layer.W3 = {span(weights.W3_weight), span(weights.W3_bias)};
  weights.layer.norm1 = {span(weights.norm1_weight), span(weights.norm1_bias)};
  weights.layer.norm2 = {span(weights.norm2_weight), span(weights.norm2_bias)};
  weights.layer.dense = {
      {span(weights.dense_W_in_weight), span(weights.dense_W_in_bias)},
      {span(weights.dense_W_out_weight), span(weights.dense_W_out_bias)}};
  weights.W_out = {span(weights.W_out_weight), span(weights.W_out_bias)};
  weights.W_s = {span(weights.W_s_weight)};
  return weights;
}

void linear_nt(const std::vector<float>& input,
               hiko_u::Span<const float> weight,
               hiko_u::Span<const float> bias,
               std::size_t row_count,
               std::size_t output_dimension,
               std::size_t input_dimension,
               std::vector<float>& output) {
  output.assign(row_count * output_dimension, 0.0F);
  for (std::size_t row = 0; row < row_count; ++row) {
    const float* row_in = input.data() + row * input_dimension;
    float* row_out = output.data() + row * output_dimension;
    for (std::size_t out = 0; out < output_dimension; ++out) {
      const float* weight_row = weight.data + out * input_dimension;
      float acc = 0.0F;
      for (std::size_t in = 0; in < input_dimension; ++in) {
        acc += row_in[in] * weight_row[in];
      }
      if (bias.data != nullptr && bias.size != 0U) {
        acc += bias.data[out];
      }
      row_out[out] = acc;
    }
  }
}

void gelu_inplace(std::vector<float>& values) {
  for (float& value : values) {
    value = hiko_p::detail::gelu_exact_scalar(value);
  }
}

void layer_norm_residual(const std::vector<float>& input,
                         const std::vector<float>& residual,
                         hiko_u::Span<const float> gamma,
                         hiko_u::Span<const float> beta,
                         std::size_t row_count,
                         std::size_t dimension,
                         std::vector<float>& output) {
  output.assign(row_count * dimension, 0.0F);
  for (std::size_t row = 0; row < row_count; ++row) {
    std::vector<float> scratch(dimension);
    for (std::size_t d = 0; d < dimension; ++d) {
      const std::size_t index = row * dimension + d;
      scratch[d] = input[index] + residual[index];
    }

    float mean = 0.0F;
    float m2 = 0.0F;
    for (std::size_t d = 0; d < dimension; ++d) {
      const float x = scratch[d];
      const float count = static_cast<float>(d + 1);
      const float delta = x - mean;
      mean += delta / count;
      const float delta2 = x - mean;
      m2 += delta * delta2;
    }
    const float variance = m2 / static_cast<float>(dimension);
    const float inv_std = 1.0F / std::sqrt(variance + 1.0e-5F);

    for (std::size_t d = 0; d < dimension; ++d) {
      const float normalized = (scratch[d] - mean) * inv_std;
      output[row * dimension + d] = normalized * gamma.data[d] + beta.data[d];
    }
  }
}

std::vector<float> reference_decoder(
    const std::vector<float>& node_embeddings,
    const std::vector<float>& edge_context,
    const std::vector<float>& attention_mask,
    const std::vector<float>& residue_mask,
    const SyntheticWeights& weights,
    std::size_t residue_count,
    std::size_t hidden,
    std::size_t neighbor_count,
    std::size_t decoder_input,
    std::size_t ffn_hidden,
    float message_scale) {
  const std::size_t input_dim = hidden + decoder_input;
  const std::size_t slot_count = residue_count * neighbor_count;
  std::vector<float> message_input(slot_count * input_dim);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    for (std::size_t neighbor = 0; neighbor < neighbor_count; ++neighbor) {
      const std::size_t slot = residue * neighbor_count + neighbor;
      float* dst = message_input.data() + slot * input_dim;
      std::copy_n(node_embeddings.data() + residue * hidden, hidden, dst);
      std::copy_n(edge_context.data() + slot * decoder_input, decoder_input,
                  dst + hidden);
    }
  }

  std::vector<float> message;
  std::vector<float> projected;
  linear_nt(message_input, span(weights.W1_weight), span(weights.W1_bias),
            slot_count, hidden, input_dim, message);
  gelu_inplace(message);
  linear_nt(message, span(weights.W2_weight), span(weights.W2_bias), slot_count,
            hidden, hidden, projected);
  gelu_inplace(projected);
  linear_nt(projected, span(weights.W3_weight), span(weights.W3_bias),
            slot_count, hidden, hidden, message);

  std::vector<float> update(residue_count * hidden, 0.0F);
  const float scale = 1.0F / message_scale;
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    float* row = update.data() + residue * hidden;
    for (std::size_t neighbor = 0; neighbor < neighbor_count; ++neighbor) {
      const std::size_t slot = residue * neighbor_count + neighbor;
      const float alpha = attention_mask[slot] * scale;
      for (std::size_t d = 0; d < hidden; ++d) {
        row[d] += alpha * message[slot * hidden + d];
      }
    }
  }

  std::vector<float> after_norm1;
  layer_norm_residual(update, node_embeddings, span(weights.norm1_weight),
                      span(weights.norm1_bias), residue_count, hidden,
                      after_norm1);

  std::vector<float> ffn_mid;
  std::vector<float> ffn_out;
  linear_nt(after_norm1, span(weights.dense_W_in_weight),
            span(weights.dense_W_in_bias), residue_count, ffn_hidden, hidden,
            ffn_mid);
  gelu_inplace(ffn_mid);
  linear_nt(ffn_mid, span(weights.dense_W_out_weight),
            span(weights.dense_W_out_bias), residue_count, hidden, ffn_hidden,
            ffn_out);

  std::vector<float> after_norm2;
  layer_norm_residual(ffn_out, after_norm1, span(weights.norm2_weight),
                      span(weights.norm2_bias), residue_count, hidden,
                      after_norm2);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    for (std::size_t d = 0; d < hidden; ++d) {
      after_norm2[residue * hidden + d] *= residue_mask[residue];
    }
  }
  return after_norm2;
}

void assert_close(const std::vector<float>& actual,
                  const std::vector<float>& expected,
                  float tolerance,
                  const char* label) {
  if (actual.size() != expected.size()) {
    fail("shape mismatch");
  }
  float max_abs = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
  }
  if (max_abs > tolerance) {
    std::fprintf(stderr,
                 "proteinmpnn_decoder_layer_parity_test: %s max_abs=%g "
                 "tolerance=%g\n",
                 label, static_cast<double>(max_abs),
                 static_cast<double>(tolerance));
    std::exit(1);
  }
}

void run_decoder_layer_test(float tolerance) {
  constexpr std::size_t kResidues = 4;
  constexpr std::size_t kHidden = hiko_s::kProteinMpnnV48020Hidden;
  constexpr std::size_t kNeighbors = 5;
  constexpr std::size_t kDecoderInput = 3 * kHidden;
  constexpr std::size_t kFfnHidden = 4 * kHidden;
  constexpr std::size_t kVocab = hiko_s::kProteinMpnnV48020Vocab;

  std::mt19937 rng(0x48DEC02Cu);
  std::vector<float> node_embeddings(kResidues * kHidden);
  std::vector<float> edge_context(kResidues * kNeighbors * kDecoderInput);
  std::vector<float> attention_mask(kResidues * kNeighbors);
  std::vector<float> residue_mask(kResidues);
  fill_random(node_embeddings, rng, -0.20F, 0.20F);
  fill_random(edge_context, rng, -0.12F, 0.12F);
  for (std::size_t i = 0; i < attention_mask.size(); ++i) {
    attention_mask[i] = (i % 3U) == 1U ? 0.0F : 1.0F;
  }
  residue_mask = {1.0F, 0.0F, 1.0F, 1.0F};

  SyntheticWeights weights =
      make_weights(kHidden, kDecoderInput, kFfnHidden, kVocab);
  hiko_m::ProteinMpnnDecoderMemoryPlan plan{kResidues, kHidden, kNeighbors,
                                          kDecoderInput, kFfnHidden};
  OwnedWorkspace workspace = make_workspace(plan);
  std::vector<float> actual(kResidues * kHidden, 0.0F);

  hiko_m::proteinmpnn_decoder_layer_scalar(
      {node_embeddings.data(),
       edge_context.data(),
       attention_mask.data(),
       residue_mask.data(),
       &weights.layer,
       &workspace.view,
       kResidues,
       kHidden,
       kNeighbors,
       kDecoderInput,
       kFfnHidden,
       hiko_s::kProteinMpnnV48020MessageScale},
      {actual.data()});
  const std::vector<float> expected =
      reference_decoder(node_embeddings, edge_context, attention_mask,
                        residue_mask, weights, kResidues, kHidden, kNeighbors,
                        kDecoderInput, kFfnHidden,
                        hiko_s::kProteinMpnnV48020MessageScale);
  assert_close(actual, expected, tolerance, "decoder layer");

  std::vector<float> logits(kResidues * kVocab, 0.0F);
  hiko_m::proteinmpnn_logits_head_scalar(
      {actual.data(), &weights.W_out, kResidues, kHidden, kVocab},
      {logits.data()});
  std::vector<float> expected_logits;
  linear_nt(actual, span(weights.W_out_weight), span(weights.W_out_bias),
            kResidues, kVocab, kHidden, expected_logits);
  assert_close(logits, expected_logits, tolerance, "logits head");

  std::array<std::int32_t, 5> token_ids{0, 7, 20, -1, 21};
  std::vector<float> embeddings(token_ids.size() * kHidden, 1.0F);
  hiko_m::proteinmpnn_sequence_embedding_scalar(
      {token_ids.data(), &weights.W_s, token_ids.size(), kVocab, kHidden},
      {embeddings.data()});
  std::vector<float> expected_embeddings(token_ids.size() * kHidden, 0.0F);
  for (std::size_t row = 0; row < token_ids.size(); ++row) {
    const std::int32_t token = token_ids[row];
    if (token < 0 || static_cast<std::size_t>(token) >= kVocab) {
      continue;
    }
    std::copy_n(weights.W_s_weight.data() +
                    static_cast<std::size_t>(token) * kHidden,
                kHidden, expected_embeddings.data() + row * kHidden);
  }
  assert_close(embeddings, expected_embeddings, 0.0F, "W_s embedding");
}

}  // namespace

int main() {
  run_decoder_layer_test(load_tolerance());
  return 0;
}

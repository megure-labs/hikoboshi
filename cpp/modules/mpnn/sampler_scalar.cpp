#include <hikoboshi/modules/mpnn/inverse_fold.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace hikoboshi::modules::mpnn {
namespace {

namespace hiko_u = hikoboshi::universal;

constexpr hiko_u::Status kOk{hiko_u::StatusCode::Ok, ""};

hiko_u::Status invalid(const char* detail) noexcept {
  return hiko_u::invalid_argument_status(detail);
}

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  state += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = state;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31U);
}

float scaled_logit(const ProteinMpnnSampleRequest& request,
                   std::size_t index) noexcept {
  const float temperature =
      request.temperature > 0.0F && std::isfinite(request.temperature)
          ? request.temperature
          : 1.0F;
  return request.logits[index] / temperature;
}

std::int32_t argmax_token(const ProteinMpnnSampleRequest& request) noexcept {
  std::size_t best = 0;
  float best_value = scaled_logit(request, 0);
  for (std::size_t index = 1; index < request.vocab_size; ++index) {
    const float value = scaled_logit(request, index);
    if (value > best_value) {
      best = index;
      best_value = value;
    }
  }
  return static_cast<std::int32_t>(best);
}

void compute_distribution(const ProteinMpnnSampleRequest& request,
                          float* probabilities,
                          float* log_probs) noexcept {
  float row_max = -INFINITY;
  for (std::size_t index = 0; index < request.vocab_size; ++index) {
    row_max = std::max(row_max, scaled_logit(request, index));
  }

  float row_sum = 0.0F;
  for (std::size_t index = 0; index < request.vocab_size; ++index) {
    const float shifted = scaled_logit(request, index) - row_max;
    const float value = std::exp(shifted);
    if (probabilities != nullptr) {
      probabilities[index] = value;
    }
    row_sum += value;
  }

  const float log_sum = std::log(row_sum);
  const float inv_sum = 1.0F / row_sum;
  for (std::size_t index = 0; index < request.vocab_size; ++index) {
    const float shifted = scaled_logit(request, index) - row_max;
    if (probabilities != nullptr) {
      probabilities[index] *= inv_sum;
    }
    if (log_probs != nullptr) {
      log_probs[index] = shifted - log_sum;
    }
  }
}

std::int32_t multinomial(const float* probabilities,
                         std::size_t vocab_size,
                         ProteinMpnnHostRng& rng) noexcept {
  const double draw = proteinmpnn_host_rng_uniform01(rng);
  double cumulative = 0.0;
  for (std::size_t index = 0; index < vocab_size; ++index) {
    cumulative += static_cast<double>(probabilities[index]);
    if (draw < cumulative) {
      return static_cast<std::int32_t>(index);
    }
  }
  return static_cast<std::int32_t>(vocab_size - 1U);
}

}  // namespace

void proteinmpnn_host_rng_seed(ProteinMpnnHostRng& rng,
                               std::uint64_t seed) noexcept {
  rng.state = seed;
}

double proteinmpnn_host_rng_uniform01(ProteinMpnnHostRng& rng) noexcept {
  return static_cast<double>(splitmix64(rng.state) >> 11U) *
         (1.0 / 9007199254740992.0);
}

hiko_u::Status proteinmpnn_sample_scalar(
    const ProteinMpnnSampleRequest& request,
    ProteinMpnnHostRng& rng,
    const ProteinMpnnSampleOutput& output) noexcept {
  if (request.vocab_size == 0) {
    return invalid("ProteinMPNN sampler vocab_size must be non-zero");
  }
  if (request.logits == nullptr || output.token_id == nullptr) {
    return invalid("ProteinMPNN sampler logits and token output must be non-null");
  }
  if (!request.greedy &&
      (request.temperature <= 0.0F || !std::isfinite(request.temperature))) {
    return invalid("ProteinMPNN sampler temperature must be finite and positive");
  }

  compute_distribution(request, output.probabilities, output.log_probs);
  if (request.greedy || request.temperature <= 0.0F) {
    *output.token_id = argmax_token(request);
    return kOk;
  }
  if (output.probabilities == nullptr) {
    return invalid("ProteinMPNN stochastic sampler requires probabilities scratch");
  }
  *output.token_id = multinomial(output.probabilities, request.vocab_size, rng);
  return kOk;
}

}  // namespace hikoboshi::modules::mpnn

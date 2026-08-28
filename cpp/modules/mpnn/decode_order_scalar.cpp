#include <hikoboshi/modules/mpnn/inverse_fold.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hikoboshi::modules::mpnn {
namespace {

namespace hiko_u = hikoboshi::universal;

constexpr hiko_u::Status kOk{hiko_u::StatusCode::Ok, ""};
constexpr double kTwoPi = 6.283185307179586476925286766559;

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

double uniform01(std::uint64_t& state) noexcept {
  return static_cast<double>(splitmix64(state) >> 11U) *
         (1.0 / 9007199254740992.0);
}

float normal_abs(std::uint64_t& state) noexcept {
  double u1 = uniform01(state);
  if (u1 <= 0.0) {
    u1 = 1.0 / 9007199254740992.0;
  }
  const double u2 = uniform01(state);
  const double radius = std::sqrt(-2.0 * std::log(u1));
  return static_cast<float>(std::fabs(radius * std::cos(kTwoPi * u2)));
}

float chain_value(const ProteinMpnnDecodeOrderRequest& request,
                  std::size_t residue) noexcept {
  const float chain =
      request.chain_mask != nullptr ? request.chain_mask[residue] : 1.0F;
  const float mask =
      request.residue_mask != nullptr ? request.residue_mask[residue] : 1.0F;
  return chain * mask;
}

bool valid_residue(const ProteinMpnnCausalMaskRequest& request,
                   std::int32_t residue) noexcept {
  return residue >= 0 &&
         static_cast<std::size_t>(residue) < request.residue_count;
}

}  // namespace

hiko_u::Status proteinmpnn_decode_order_scalar(
    const ProteinMpnnDecodeOrderRequest& request,
    const ProteinMpnnDecodeOrderOutput& output) noexcept {
  if (request.residue_count > 0 && output.decoding_order == nullptr) {
    return invalid("ProteinMPNN decode order output pointer is null");
  }

  if (request.use_input_decoding_order) {
    std::vector<unsigned char> seen(request.residue_count, 0U);
    for (std::size_t position = 0; position < request.residue_count; ++position) {
      const std::int32_t residue =
          request.input_order != nullptr
              ? request.input_order[position]
              : static_cast<std::int32_t>(position);
      if (residue < 0 ||
          static_cast<std::size_t>(residue) >= request.residue_count ||
          seen[static_cast<std::size_t>(residue)] != 0U) {
        return invalid("ProteinMPNN input decoding_order must be a permutation");
      }
      seen[static_cast<std::size_t>(residue)] = 1U;
      output.decoding_order[position] = residue;
      if (output.order_scores != nullptr) {
        output.order_scores[position] = static_cast<float>(position);
      }
    }
    return kOk;
  }

  std::vector<std::int32_t> order(request.residue_count, 0);
  std::vector<float> scores(request.residue_count, 0.0F);
  std::uint64_t state = request.seed;
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    order[residue] = static_cast<std::int32_t>(residue);
    scores[residue] = (chain_value(request, residue) + 0.0001F) *
                      normal_abs(state);
  }
  std::stable_sort(order.begin(), order.end(),
                   [&scores](std::int32_t lhs, std::int32_t rhs) {
                     const float lhs_score =
                         scores[static_cast<std::size_t>(lhs)];
                     const float rhs_score =
                         scores[static_cast<std::size_t>(rhs)];
                     if (lhs_score == rhs_score) {
                       return lhs < rhs;
                     }
                     return lhs_score < rhs_score;
                   });
  for (std::size_t position = 0; position < request.residue_count; ++position) {
    output.decoding_order[position] = order[position];
    if (output.order_scores != nullptr) {
      output.order_scores[position] =
          scores[static_cast<std::size_t>(order[position])];
    }
  }
  return kOk;
}

hiko_u::Status proteinmpnn_causal_masks_scalar(
    const ProteinMpnnCausalMaskRequest& request,
    const ProteinMpnnCausalMaskOutput& output) noexcept {
  if (request.residue_count > 0 &&
      (request.decoding_order == nullptr || request.edge_indices == nullptr ||
       output.mask_bw == nullptr || output.mask_fw == nullptr)) {
    return invalid("ProteinMPNN causal mask pointers must be non-null");
  }

  std::vector<std::int32_t> order_position(request.residue_count, -1);
  for (std::size_t position = 0; position < request.residue_count; ++position) {
    const std::int32_t residue = request.decoding_order[position];
    if (residue < 0 ||
        static_cast<std::size_t>(residue) >= request.residue_count ||
        order_position[static_cast<std::size_t>(residue)] >= 0) {
      return invalid("ProteinMPNN decoding_order must be a permutation");
    }
    order_position[static_cast<std::size_t>(residue)] =
        static_cast<std::int32_t>(position);
  }

  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    const float row_mask =
        request.residue_mask != nullptr ? request.residue_mask[residue] : 1.0F;
    const std::int32_t row_position =
        order_position[residue];
    for (std::size_t neighbor_slot = 0;
         neighbor_slot < request.neighbor_count; ++neighbor_slot) {
      const std::size_t slot = residue * request.neighbor_count + neighbor_slot;
      const std::int32_t neighbor = request.edge_indices[slot];
      float backward = 0.0F;
      float forward = 0.0F;
      if (valid_residue(request, neighbor)) {
        const std::int32_t neighbor_position =
            order_position[static_cast<std::size_t>(neighbor)];
        if (neighbor_position < row_position) {
          backward = row_mask;
        } else {
          forward = row_mask;
        }
      }
      output.mask_bw[slot] = backward;
      output.mask_fw[slot] = forward;
    }
  }
  return kOk;
}

}  // namespace hikoboshi::modules::mpnn

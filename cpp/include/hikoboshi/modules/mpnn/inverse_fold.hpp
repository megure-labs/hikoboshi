#ifndef HIKOBOSHI_MODULES_MPNN_INVERSE_FOLD_HPP
#define HIKOBOSHI_MODULES_MPNN_INVERSE_FOLD_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/modules/mpnn/proteinmpnn_decoder.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::modules::mpnn {

inline constexpr char kProteinMpnnV48020Alphabet[] =
    "ACDEFGHIKLMNPQRSTVWYX";

struct ProteinMpnnHostRng {
  std::uint64_t state = 0;
};

void proteinmpnn_host_rng_seed(ProteinMpnnHostRng& rng,
                               std::uint64_t seed) noexcept;

double proteinmpnn_host_rng_uniform01(ProteinMpnnHostRng& rng) noexcept;

struct ProteinMpnnSampleRequest {
  const float* logits;  // [vocab_size], unnormalized logits
  std::size_t vocab_size;
  float temperature;
  bool greedy;
};

struct ProteinMpnnSampleOutput {
  std::int32_t* token_id;  // scalar
  float* probabilities;   // optional [vocab_size]
  float* log_probs;       // optional [vocab_size]
};

hikoboshi::universal::Status proteinmpnn_sample_scalar(
    const ProteinMpnnSampleRequest& request,
    ProteinMpnnHostRng& rng,
    const ProteinMpnnSampleOutput& output) noexcept;

struct ProteinMpnnDecodeOrderRequest {
  const float* chain_mask;            // optional [L], defaults to ones
  const float* residue_mask;          // optional [L], defaults to ones
  const std::int32_t* input_order;    // optional [L]
  std::size_t residue_count;
  std::uint64_t seed;
  bool use_input_decoding_order;
};

struct ProteinMpnnDecodeOrderOutput {
  std::int32_t* decoding_order;  // [L]
  float* order_scores;           // optional [L]
};

hikoboshi::universal::Status proteinmpnn_decode_order_scalar(
    const ProteinMpnnDecodeOrderRequest& request,
    const ProteinMpnnDecodeOrderOutput& output) noexcept;

struct ProteinMpnnCausalMaskRequest {
  const std::int32_t* decoding_order;  // [L]
  const std::int32_t* edge_indices;    // [L, K]
  const float* residue_mask;           // optional [L], defaults to ones
  std::size_t residue_count;
  std::size_t neighbor_count;
};

struct ProteinMpnnCausalMaskOutput {
  float* mask_bw;  // [L, K]
  float* mask_fw;  // [L, K]
};

hikoboshi::universal::Status proteinmpnn_causal_masks_scalar(
    const ProteinMpnnCausalMaskRequest& request,
    const ProteinMpnnCausalMaskOutput& output) noexcept;

struct ProteinMpnnInverseFoldOptions {
  std::uint64_t seed = 0x48020D3C0D3ull;
  std::uint64_t decode_order_seed = 0x48020D3C0D3ull;
  float temperature = 0.1F;
  bool greedy = false;
  bool use_input_decoding_order = false;
};

struct ProteinMpnnInverseFoldRequest {
  const float* coordinates;  // row-major [L, 5, 3]
  const hikoboshi::universal::AtomSource* atom_sources;  // row-major [L, 5]
  const hikoboshi::universal::detail::ProteinMpnnV48020Weights* weights;
  hikoboshi::modules::detail::Mpnn64Workspace* encoder_workspace;
  ProteinMpnnDecoderWorkspace* decoder_workspace;
  std::size_t residue_count;
  const std::int32_t* residue_indices = nullptr;  // optional [L], defaults 0..L-1
  const std::int32_t* chain_labels = nullptr;     // optional [L], defaults one chain
  const float* chain_mask = nullptr;              // optional [L], defaults ones
  const std::int32_t* input_decoding_order = nullptr;  // optional [L]
  const std::int32_t* true_token_ids = nullptr;        // optional [L]
  ProteinMpnnInverseFoldOptions options{};
};

struct ProteinMpnnInverseFoldOutput {
  std::int32_t* token_ids;       // [L]
  char* sequence;                // optional [L + 1], null-terminated when present
  float* log_probs;              // [L, 21]
  std::int32_t* decoding_order;  // [L]
  float* sequence_score;         // optional scalar, average NLL over active mask
  std::size_t residue_count;
  std::size_t vocab_size;
};

hikoboshi::universal::Status proteinmpnn_inverse_fold_scalar(
    const ProteinMpnnInverseFoldRequest& request,
    const ProteinMpnnInverseFoldOutput& output);

struct ProteinMpnnTeacherForcedRequest {
  const float* coordinates;  // row-major [L, 5, 3]
  const hikoboshi::universal::AtomSource* atom_sources;  // row-major [L, 5]
  const std::int32_t* token_ids;  // [L]
  const hikoboshi::universal::detail::ProteinMpnnV48020Weights* weights;
  hikoboshi::modules::detail::Mpnn64Workspace* encoder_workspace;
  ProteinMpnnDecoderWorkspace* decoder_workspace;
  std::size_t residue_count;
  const std::int32_t* residue_indices = nullptr;  // optional [L], defaults 0..L-1
  const std::int32_t* chain_labels = nullptr;     // optional [L], defaults one chain
  const float* chain_mask = nullptr;              // optional [L], defaults ones
  const std::int32_t* input_decoding_order = nullptr;  // optional [L]
  ProteinMpnnInverseFoldOptions options{};
};

struct ProteinMpnnTeacherForcedOutput {
  float* log_probs;              // [L, 21]
  std::int32_t* decoding_order;  // optional [L]
  float* sequence_score;         // optional scalar, average NLL over active mask
  std::size_t residue_count;
  std::size_t vocab_size;
};

hikoboshi::universal::Status proteinmpnn_teacher_forced_forward_scalar(
    const ProteinMpnnTeacherForcedRequest& request,
    const ProteinMpnnTeacherForcedOutput& output);

std::size_t proteinmpnn_v48_020_effective_neighbor_count(
    std::size_t residue_count) noexcept;

char proteinmpnn_v48_020_token_to_aa(std::int32_t token_id) noexcept;

}  // namespace hikoboshi::modules::mpnn

#endif  // HIKOBOSHI_MODULES_MPNN_INVERSE_FOLD_HPP

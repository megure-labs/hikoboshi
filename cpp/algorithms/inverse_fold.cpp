#include <hikoboshi/algorithms/inverse_fold.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/modules/mpnn/inverse_fold.hpp>
#include <hikoboshi/modules/mpnn/proteinmpnn_decoder.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

namespace hikoboshi::algorithms {
namespace {

namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_m = hikoboshi::modules::mpnn;
namespace hiko_u = hikoboshi::universal;
namespace pmp = hikoboshi::universal::detail;

constexpr hiko_u::Status ok() noexcept { return hiko_u::ok_status(); }

constexpr hiko_u::Status invalid(const char* detail) noexcept {
  return hiko_u::invalid_argument_status(detail);
}

constexpr hiko_u::Status failed_precondition(const char* detail) noexcept {
  return hiko_u::failed_precondition_status(detail);
}

template <typename T>
hiko_u::Span<T> span(std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size()};
}

struct EncoderWorkspaceStorage {
  explicit EncoderWorkspaceStorage(std::size_t residue_count,
                                   std::size_t neighbor_count)
      : plan{residue_count,
             pmp::kProteinMpnnV48020Hidden,
             neighbor_count,
             pmp::kProteinMpnnV48020RbfCount,
             pmp::kProteinMpnnV48020NumEncoderLayers},
        ca_coordinates(hiko_d::mpnn64_ca_coordinate_count(plan)),
        residue_features(std::max(
            hiko_d::mpnn64_residue_feature_count(plan),
            hiko_d::mpnn64_neighbor_slot_count(plan) * 3U *
                pmp::kProteinMpnnV48020Hidden)),
        neighbor_indices(hiko_d::mpnn64_neighbor_slot_count(plan)),
        neighbor_squared_distances(hiko_d::mpnn64_neighbor_slot_count(plan)),
        rbf_features(hiko_d::mpnn64_neighbor_rbf_count(plan)),
        residue_state(hiko_d::mpnn64_residue_hidden_count(plan)),
        gathered_state(hiko_d::mpnn64_neighbor_hidden_count(plan)),
        edge_state(hiko_d::mpnn64_neighbor_hidden_count(plan)),
        message_state(hiko_d::mpnn64_neighbor_hidden_count(plan)),
        projected_message_state(hiko_d::mpnn64_neighbor_hidden_count(plan)),
        residue_scratch(hiko_d::mpnn64_residue_hidden_count(plan)),
        ffn_hidden(hiko_d::mpnn64_ffn_hidden_count(plan)) {}

  hiko_d::Mpnn64Workspace view() noexcept {
    return {plan,
            span(ca_coordinates),
            span(residue_features),
            span(neighbor_indices),
            span(neighbor_squared_distances),
            span(rbf_features),
            span(residue_state),
            span(gathered_state),
            span(edge_state),
            span(message_state),
            span(projected_message_state),
            span(residue_scratch),
            span(ffn_hidden)};
  }

  hiko_d::Mpnn64MemoryPlan plan;
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
};

struct DecoderWorkspaceStorage {
  explicit DecoderWorkspaceStorage(std::size_t residue_count,
                                   std::size_t neighbor_count)
      : plan{std::max<std::size_t>(residue_count, 1U),
             pmp::kProteinMpnnV48020Hidden,
             neighbor_count,
             pmp::kProteinMpnnV48020DecoderNumInDimension,
             pmp::kProteinMpnnV48020FfnHidden},
        message_input(hiko_m::proteinmpnn_decoder_message_input_count(plan)),
        message_state(hiko_m::proteinmpnn_decoder_neighbor_hidden_count(plan)),
        projected_message_state(
            hiko_m::proteinmpnn_decoder_neighbor_hidden_count(plan)),
        residue_scratch(hiko_m::proteinmpnn_decoder_residue_hidden_count(plan)),
        ffn_hidden(hiko_m::proteinmpnn_decoder_ffn_hidden_count(plan)) {}

  hiko_m::ProteinMpnnDecoderWorkspace view() noexcept {
    return {plan,
            span(message_input),
            span(message_state),
            span(projected_message_state),
            span(residue_scratch),
            span(ffn_hidden)};
  }

  hiko_m::ProteinMpnnDecoderMemoryPlan plan;
  std::vector<float> message_input;
  std::vector<float> message_state;
  std::vector<float> projected_message_state;
  std::vector<float> residue_scratch;
  std::vector<float> ffn_hidden;
};

char ascii_upper(char value) noexcept {
  return value >= 'a' && value <= 'z'
             ? static_cast<char>(value - 'a' + 'A')
             : value;
}

bool is_standard_aa(char value) noexcept {
  switch (ascii_upper(value)) {
    case 'A':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'P':
    case 'Q':
    case 'R':
    case 'S':
    case 'T':
    case 'V':
    case 'W':
    case 'Y':
      return true;
    default:
      return false;
  }
}

hiko_u::MetricValue recovery_vs_native(std::string_view sequence,
                                    hiko_u::Span<const char> residue_codes,
                                    std::size_t residue_count) {
  if (residue_codes.data == nullptr || residue_codes.size < residue_count) {
    return {0.0, false, hiko_u::MetricInvalidReason::MissingStructureMetadata};
  }
  std::size_t compared = 0;
  std::size_t matches = 0;
  const std::size_t limit = std::min(sequence.size(), residue_count);
  for (std::size_t index = 0; index < limit; ++index) {
    const char native = ascii_upper(residue_codes[index]);
    if (!is_standard_aa(native)) {
      continue;
    }
    ++compared;
    if (ascii_upper(sequence[index]) == native) {
      ++matches;
    }
  }
  if (compared == 0) {
    return {0.0, false, hiko_u::MetricInvalidReason::MissingStructureMetadata};
  }
  return {static_cast<double>(matches) / static_cast<double>(compared), true,
          hiko_u::MetricInvalidReason::None};
}

std::uint64_t sample_seed(std::uint64_t seed,
                          std::size_t sample_index) noexcept {
  return seed + static_cast<std::uint64_t>(sample_index);
}

void apply_backbone_noise(std::vector<float>& coordinates,
                          hiko_u::Span<const hiko_u::AtomSource> atom_sources,
                          float sigma,
                          std::uint64_t seed) {
  std::mt19937_64 rng(seed ^ 0x4D504E4E76343875ull);
  std::normal_distribution<float> normal(0.0F, sigma);
  for (std::size_t atom_index = 0; atom_index < atom_sources.size;
       ++atom_index) {
    if (atom_sources[atom_index] == hiko_u::AtomSource::Missing) {
      continue;
    }
    const std::size_t coordinate_index = atom_index * 3U;
    coordinates[coordinate_index + 0U] += normal(rng);
    coordinates[coordinate_index + 1U] += normal(rng);
    coordinates[coordinate_index + 2U] += normal(rng);
  }
}

InverseFoldDecodeOrder normalize_decode_order(
    InverseFoldDecodeOrder order) noexcept {
  switch (order) {
    case InverseFoldDecodeOrder::Random:
    case InverseFoldDecodeOrder::NToC:
      return order;
  }
  return InverseFoldDecodeOrder::Random;
}

}  // namespace

universal::Result<InverseFoldResult> inverse_fold_proteinmpnn_v48_020(
    const InverseFoldRequest& request) {
  InverseFoldResult result{};
  const hiko_u::StructureView& structure = request.structure;
  const std::size_t residue_count = structure.residue_count;
  constexpr std::size_t kAtomCount = hiko_u::kCanonicalAtomCount;
  constexpr std::size_t kAxes = hiko_u::kCoordinateAxisCount;
  constexpr std::size_t kVocab = pmp::kProteinMpnnV48020Vocab;
  const std::size_t coordinate_count = residue_count * kAtomCount * kAxes;
  const std::size_t atom_source_count = residue_count * kAtomCount;

  if (request.weights == nullptr) {
    return {failed_precondition(
                "ProteinMPNN inverse-folding weights are not available"),
            result};
  }
  if (residue_count == 0) {
    return {invalid("inverse-fold structure must contain at least one residue"),
            result};
  }
  if (structure.coordinates.data == nullptr ||
      structure.coordinates.size < coordinate_count ||
      structure.atom_sources.data == nullptr ||
      structure.atom_sources.size < atom_source_count) {
    return {invalid(
                "inverse-fold structure must provide canonical backbone atoms"),
            result};
  }
  if (request.options.num_seqs == 0) {
    return {invalid("inverse-fold num_seqs must be greater than zero"), result};
  }
  if (!std::isfinite(request.options.sampling_temp) ||
      request.options.sampling_temp < 0.0F) {
    return {invalid("inverse-fold sampling_temp must be finite and >= 0"),
            result};
  }
  if (!std::isfinite(request.options.backbone_noise) ||
      request.options.backbone_noise < 0.0F) {
    return {invalid("inverse-fold backbone_noise must be finite and >= 0"),
            result};
  }

  const std::size_t neighbor_count =
      hiko_m::proteinmpnn_v48_020_effective_neighbor_count(residue_count);
  EncoderWorkspaceStorage encoder_storage(residue_count, neighbor_count);
  DecoderWorkspaceStorage decoder_storage(residue_count, neighbor_count);
  hiko_d::Mpnn64Workspace encoder_workspace = encoder_storage.view();
  hiko_m::ProteinMpnnDecoderWorkspace decoder_workspace = decoder_storage.view();

  std::vector<std::int32_t> input_order;
  const InverseFoldDecodeOrder decode_order =
      normalize_decode_order(request.options.decode_order);
  if (decode_order == InverseFoldDecodeOrder::NToC) {
    input_order.resize(residue_count);
    std::iota(input_order.begin(), input_order.end(), 0);
  }

  result.sequences.reserve(request.options.num_seqs);
  if (request.options.keep_log_probs) {
    result.logprobs.num_seqs = request.options.num_seqs;
    result.logprobs.residue_count = residue_count;
    result.logprobs.vocab_size = kVocab;
    result.logprobs.values.reserve(request.options.num_seqs * residue_count *
                                   kVocab);
  }

  std::vector<std::int32_t> token_ids(residue_count);
  std::vector<char> sequence(residue_count + 1U, '\0');
  std::vector<float> log_probs(residue_count * kVocab);
  std::vector<std::int32_t> module_order(residue_count);

  for (std::size_t index = 0; index < request.options.num_seqs; ++index) {
    const std::uint64_t seed = sample_seed(request.options.seed, index);
    const float* coordinates = structure.coordinates.data;
    std::vector<float> noisy_coordinates;
    if (request.options.backbone_noise > 0.0F) {
      noisy_coordinates.assign(structure.coordinates.data,
                               structure.coordinates.data + coordinate_count);
      apply_backbone_noise(noisy_coordinates, structure.atom_sources,
                           request.options.backbone_noise, seed);
      coordinates = noisy_coordinates.data();
    }

    hiko_m::ProteinMpnnInverseFoldOptions options{};
    options.seed = seed;
    options.decode_order_seed = seed;
    options.temperature = request.options.sampling_temp;
    options.greedy = request.options.sampling_temp <= 0.0F;
    options.use_input_decoding_order =
        decode_order == InverseFoldDecodeOrder::NToC;

    float sequence_score = 0.0F;
    hiko_m::ProteinMpnnInverseFoldRequest module_request{};
    module_request.coordinates = coordinates;
    module_request.atom_sources = structure.atom_sources.data;
    module_request.weights = request.weights;
    module_request.encoder_workspace = &encoder_workspace;
    module_request.decoder_workspace = &decoder_workspace;
    module_request.residue_count = residue_count;
    module_request.input_decoding_order =
        input_order.empty() ? nullptr : input_order.data();
    module_request.options = options;

    hiko_m::ProteinMpnnInverseFoldOutput module_output{};
    module_output.token_ids = token_ids.data();
    module_output.sequence = sequence.data();
    module_output.log_probs = log_probs.data();
    module_output.decoding_order = module_order.data();
    module_output.sequence_score = &sequence_score;
    module_output.residue_count = residue_count;
    module_output.vocab_size = kVocab;

    const hiko_u::Status status =
        hiko_m::proteinmpnn_inverse_fold_scalar(module_request, module_output);
    if (!status.ok()) {
      return {status, result};
    }

    InverseFoldSequence entry{};
    entry.sequence.assign(sequence.data(), residue_count);
    entry.score = static_cast<double>(sequence_score);
    entry.recovery_vs_native =
        recovery_vs_native(entry.sequence, structure.residue_codes,
                           residue_count);
    entry.seed = seed;
    entry.decode_order = decode_order;
    result.sequences.push_back(std::move(entry));

    if (request.options.keep_log_probs) {
      result.logprobs.values.insert(result.logprobs.values.end(),
                                    log_probs.begin(), log_probs.end());
    }
  }

  return {ok(), std::move(result)};
}

}  // namespace hikoboshi::algorithms

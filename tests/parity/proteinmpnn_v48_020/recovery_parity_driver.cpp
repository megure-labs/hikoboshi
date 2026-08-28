#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/structure_loader.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/modules/mpnn/inverse_fold.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_m = hikoboshi::modules::mpnn;
namespace hiko_u = hikoboshi::universal;
namespace pmp = hikoboshi::universal::detail;
namespace hiko_w = hikoboshi::weights;

namespace {

constexpr std::size_t kHidden = pmp::kProteinMpnnV48020Hidden;
constexpr std::size_t kVocab = pmp::kProteinMpnnV48020Vocab;
constexpr std::size_t kRbf = pmp::kProteinMpnnV48020RbfCount;
constexpr std::size_t kDecoderInput =
    pmp::kProteinMpnnV48020DecoderNumInDimension;
constexpr std::size_t kFfnHidden = pmp::kProteinMpnnV48020FfnHidden;
constexpr std::string_view kAlphabet = "ACDEFGHIKLMNPQRSTVWYX";

[[noreturn]] void fail(const std::string& detail) {
  std::cerr << "proteinmpnn_recovery_parity_driver: " << detail << "\n";
  std::exit(1);
}

template <typename T>
hiko_u::Span<T> span(std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size()};
}

struct EncoderWorkspaceStorage {
  explicit EncoderWorkspaceStorage(std::size_t residue_count,
                                   std::size_t neighbor_count)
      : plan{residue_count,
             kHidden,
             neighbor_count,
             kRbf,
             pmp::kProteinMpnnV48020NumEncoderLayers},
        ca_coordinates(hiko_d::mpnn64_ca_coordinate_count(plan)),
        residue_features(std::max(
            hiko_d::mpnn64_residue_feature_count(plan),
            hiko_d::mpnn64_neighbor_slot_count(plan) * 3U * kHidden)),
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
             kHidden,
             neighbor_count,
             kDecoderInput,
             kFfnHidden},
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

bool status_ok(const hiko_u::Status status) noexcept {
  return status.code == hiko_u::StatusCode::Ok;
}

std::string status_detail(const hiko_u::Status status) {
  return status.detail != nullptr ? std::string(status.detail)
                                  : std::string("non-ok status");
}

char ascii_upper(char value) noexcept {
  return value >= 'a' && value <= 'z'
             ? static_cast<char>(value - 'a' + 'A')
             : value;
}

std::int32_t token_for_aa(const char residue) {
  const char aa = ascii_upper(residue);
  const std::size_t index = kAlphabet.find(aa);
  if (index == std::string_view::npos) {
    fail(std::string("fixture contains unsupported residue code: ") + aa);
  }
  return static_cast<std::int32_t>(index);
}

char aa_for_token(const std::size_t token) noexcept {
  return token < kAlphabet.size() ? kAlphabet[token] : 'X';
}

std::string native_sequence(const hiko_u::StructureView& structure) {
  if (structure.residue_codes.data == nullptr ||
      structure.residue_codes.size < structure.residue_count) {
    fail("fixture structure has no residue codes");
  }
  std::string sequence;
  sequence.reserve(structure.residue_count);
  for (std::size_t index = 0; index < structure.residue_count; ++index) {
    sequence.push_back(ascii_upper(structure.residue_codes.data[index]));
  }
  return sequence;
}

std::vector<std::int32_t> tokens_for_sequence(std::string_view sequence) {
  std::vector<std::int32_t> tokens;
  tokens.reserve(sequence.size());
  for (const char residue : sequence) {
    tokens.push_back(token_for_aa(residue));
  }
  return tokens;
}

std::string argmax_sequence(const std::vector<float>& log_probs,
                            const std::size_t residue_count) {
  std::string sequence;
  sequence.reserve(residue_count);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    const float* row = log_probs.data() + residue * kVocab;
    const float* best = std::max_element(row, row + kVocab);
    sequence.push_back(aa_for_token(static_cast<std::size_t>(best - row)));
  }
  return sequence;
}

double recovery(std::string_view designed, std::string_view native) noexcept {
  if (designed.size() != native.size() || designed.empty()) {
    return 0.0;
  }
  std::size_t matches = 0;
  for (std::size_t index = 0; index < native.size(); ++index) {
    if (ascii_upper(designed[index]) == ascii_upper(native[index])) {
      ++matches;
    }
  }
  return static_cast<double>(matches) / static_cast<double>(native.size());
}

const pmp::ProteinMpnnV48020Weights* resolve_weights(
    hiko_u::PackageHandle package) {
  if (package.descriptor == nullptr) {
    fail("ProteinMPNN package descriptor is null");
  }
  const hiko_u::WeightsHandle handle =
      package.descriptor->compatibility_views.weights;
  if (handle.opaque == nullptr) {
    fail("ProteinMPNN package has no prepared weight view");
  }
  return static_cast<const pmp::ProteinMpnnV48020Weights*>(handle.opaque);
}

hiko::Engine make_engine(hiko_u::PackageHandle package) {
  hiko::EngineConfig config{};
  config.package = package;
  config.weights = package.descriptor->compatibility_views.weights;
  config.execution.backend = hiko_u::Backend::Auto;
  return hiko::Engine{config};
}

struct TeacherForcedResult {
  std::vector<float> log_probs;
  std::string argmax;
  float score = 0.0F;
};

TeacherForcedResult run_teacher_forced(
    const hiko_u::StructureView& structure,
    const pmp::ProteinMpnnV48020Weights* weights,
    const std::vector<std::int32_t>& tokens,
    const std::uint64_t seed,
    const float sampling_temp) {
  const std::size_t residue_count = structure.residue_count;
  const std::size_t neighbor_count =
      hiko_m::proteinmpnn_v48_020_effective_neighbor_count(residue_count);
  EncoderWorkspaceStorage encoder_storage(residue_count, neighbor_count);
  DecoderWorkspaceStorage decoder_storage(residue_count, neighbor_count);
  hiko_d::Mpnn64Workspace encoder_workspace = encoder_storage.view();
  hiko_m::ProteinMpnnDecoderWorkspace decoder_workspace = decoder_storage.view();
  std::vector<std::int32_t> input_order(residue_count);
  std::iota(input_order.begin(), input_order.end(), 0);

  TeacherForcedResult result{};
  result.log_probs.assign(residue_count * kVocab, 0.0F);
  std::vector<std::int32_t> output_order(residue_count, -1);

  hiko_m::ProteinMpnnInverseFoldOptions options{};
  options.seed = seed;
  options.decode_order_seed = seed;
  options.temperature = sampling_temp;
  options.use_input_decoding_order = true;

  hiko_m::ProteinMpnnTeacherForcedRequest request{};
  request.coordinates = structure.coordinates.data;
  request.atom_sources = structure.atom_sources.data;
  request.token_ids = tokens.data();
  request.weights = weights;
  request.encoder_workspace = &encoder_workspace;
  request.decoder_workspace = &decoder_workspace;
  request.residue_count = residue_count;
  request.input_decoding_order = input_order.data();
  request.options = options;

  hiko_m::ProteinMpnnTeacherForcedOutput output{};
  output.log_probs = result.log_probs.data();
  output.decoding_order = output_order.data();
  output.sequence_score = &result.score;
  output.residue_count = residue_count;
  output.vocab_size = kVocab;

  const hiko_u::Status status =
      hiko_m::proteinmpnn_teacher_forced_forward_scalar(request, output);
  if (!status_ok(status)) {
    fail(status_detail(status));
  }
  result.argmax = argmax_sequence(result.log_probs, residue_count);
  return result;
}

hiko::InverseFoldSequenceResult run_design(const hiko::Engine& engine,
                                          const hiko_u::StructureView& structure,
                                          const std::uint64_t seed,
                                          const float sampling_temp) {
  hiko::InverseFoldRequest request{};
  request.structure = structure;
  request.package = std::string(hiko_w::kDefaultProteinMpnnV48Eps020ModelName);
  request.sampling_temp = sampling_temp;
  request.num_seqs = 1;
  request.seed = seed;
  request.decode_order = hiko::InverseFoldDecodeOrder::NToC;
  request.backbone_noise = 0.0F;
  const auto result = engine.inverse_fold(request);
  if (!status_ok(result.status)) {
    fail(status_detail(result.status));
  }
  if (result.value.sequences.size() != 1U) {
    fail("inverse_fold returned an unexpected sequence count");
  }
  return result.value.sequences.front();
}

void write_json_string(std::ostream& out, std::string_view value) {
  out << '"';
  for (const char c : value) {
    if (c == '"' || c == '\\') {
      out << '\\' << c;
    } else if (c == '\n') {
      out << "\\n";
    } else {
      out << c;
    }
  }
  out << '"';
}

void write_float_array(std::ostream& out, const std::vector<float>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << std::setprecision(10) << values[index];
  }
  out << ']';
}

std::uint64_t parse_seed(const char* text) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    fail("seed must be an unsigned integer");
  }
  return static_cast<std::uint64_t>(parsed);
}

float parse_float(const char* text) {
  char* end = nullptr;
  const float parsed = std::strtof(text, &end);
  if (end == text || *end != '\0') {
    fail("sampling temperature must be a finite float");
  }
  return parsed;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    fail("usage: recovery_parity_driver FIXTURE.pdb SEED SAMPLING_TEMP");
  }
  const std::string pdb_path = argv[1];
  const std::uint64_t seed = parse_seed(argv[2]);
  const float sampling_temp = parse_float(argv[3]);

  const auto loaded = hikoboshi::io::load_structure_from_file(pdb_path);
  if (!status_ok(loaded.status)) {
    fail(status_detail(loaded.status));
  }
  const hiko_u::StructureView structure = loaded.value.view();
  const std::string native = native_sequence(structure);
  const std::vector<std::int32_t> tokens = tokens_for_sequence(native);

  const auto package =
      hiko_w::default_package(hiko_w::kDefaultProteinMpnnV48Eps020ModelName);
  if (!status_ok(package.status)) {
    fail(status_detail(package.status));
  }
  const pmp::ProteinMpnnV48020Weights* weights =
      resolve_weights(package.value);
  const hiko::Engine engine = make_engine(package.value);

  const TeacherForcedResult teacher =
      run_teacher_forced(structure, weights, tokens, seed, sampling_temp);
  const hiko::InverseFoldSequenceResult sample =
      run_design(engine, structure, seed, sampling_temp);
  const hiko::InverseFoldSequenceResult greedy =
      run_design(engine, structure, seed, 0.0F);

  std::cout << '{';
  std::cout << "\"residue_count\":" << structure.residue_count;
  std::cout << ",\"native_sequence\":";
  write_json_string(std::cout, native);
  std::cout << ",\"teacher_log_probs\":";
  write_float_array(std::cout, teacher.log_probs);
  std::cout << ",\"teacher_argmax_sequence\":";
  write_json_string(std::cout, teacher.argmax);
  std::cout << ",\"teacher_score\":" << std::setprecision(10)
            << teacher.score;
  std::cout << ",\"teacher_argmax_recovery\":" << std::setprecision(10)
            << recovery(teacher.argmax, native);
  std::cout << ",\"sample_sequence\":";
  write_json_string(std::cout, sample.sequence);
  std::cout << ",\"sample_recovery\":" << std::setprecision(10)
            << recovery(sample.sequence, native);
  std::cout << ",\"greedy_sequence\":";
  write_json_string(std::cout, greedy.sequence);
  std::cout << ",\"greedy_recovery\":" << std::setprecision(10)
            << recovery(greedy.sequence, native);
  std::cout << "}\n";
  return 0;
}

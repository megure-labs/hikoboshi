#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/algorithms/inverse_fold.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>

#include <cstddef>
#include <string_view>
#include <utility>

namespace hikoboshi::api {
namespace {

namespace hiko = hikoboshi::algorithms;
namespace hiko_u = hikoboshi::universal;
namespace pmp = hikoboshi::universal::detail;

constexpr std::string_view kProteinMpnnPackageId =
    "proteinmpnn-v48-eps020";
constexpr std::string_view kProteinMpnnArchitectureId =
    "proteinmpnn_v48_eps020";

constexpr hiko_u::Status invalid(const char* detail) noexcept {
  return hiko_u::invalid_argument_status(detail);
}

constexpr hiko_u::Status failed_precondition(const char* detail) noexcept {
  return hiko_u::failed_precondition_status(detail);
}

bool ascii_case_equal(std::string_view lhs, std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    char left = lhs[index];
    char right = rhs[index];
    if (left >= 'A' && left <= 'Z') {
      left = static_cast<char>(left - 'A' + 'a');
    }
    if (right >= 'A' && right <= 'Z') {
      right = static_cast<char>(right - 'A' + 'a');
    }
    if (left != right) {
      return false;
    }
  }
  return true;
}

bool package_name_matches(const hiko_u::PackageDescriptor& descriptor,
                          std::string_view requested) noexcept {
  if (requested.empty() ||
      ascii_case_equal(descriptor.identity.package_id, requested)) {
    return true;
  }
  for (const std::string_view alias : descriptor.identity.aliases) {
    if (ascii_case_equal(alias, requested)) {
      return true;
    }
  }
  return false;
}

hiko::InverseFoldDecodeOrder to_algorithm_order(
    InverseFoldDecodeOrder order) noexcept {
  switch (order) {
    case InverseFoldDecodeOrder::Random:
      return hiko::InverseFoldDecodeOrder::Random;
    case InverseFoldDecodeOrder::NToC:
      return hiko::InverseFoldDecodeOrder::NToC;
  }
  return hiko::InverseFoldDecodeOrder::Random;
}

InverseFoldDecodeOrder to_api_order(hiko::InverseFoldDecodeOrder order) noexcept {
  switch (order) {
    case hiko::InverseFoldDecodeOrder::Random:
      return InverseFoldDecodeOrder::Random;
    case hiko::InverseFoldDecodeOrder::NToC:
      return InverseFoldDecodeOrder::NToC;
  }
  return InverseFoldDecodeOrder::Random;
}

hiko_u::StructureView structure_from_coords(const InverseFoldRequest& request) {
  hiko_u::StructureView view{};
  view.residue_count = request.coords.residue_count;
  view.coordinates = request.coords.coordinates;
  view.atom_sources = request.coords.atom_sources;
  view.residue_codes = request.coords.residue_codes;
  view.residues = request.coords.residues;
  view.input_id = request.pdb_path;
  view.source_filename = request.pdb_path;
  return view;
}

hiko_u::Result<const pmp::ProteinMpnnV48020Weights*> proteinmpnn_weights(
    const EngineConfig& config,
    std::string_view requested_package) {
  if (config.package.descriptor == nullptr) {
    return {failed_precondition(
                "inverse_fold requires a resolved ProteinMPNN package"),
            nullptr};
  }
  const hiko_u::PackageDescriptor& descriptor = *config.package.descriptor;
  if (descriptor.identity.package_kind != hiko_u::PackageKind::GraphIr ||
      descriptor.execution.mode != hiko_u::PackageExecutionMode::GraphIr ||
      !ascii_case_equal(descriptor.identity.package_id, kProteinMpnnPackageId) ||
      !ascii_case_equal(descriptor.execution.architecture_id,
                        kProteinMpnnArchitectureId)) {
    return {invalid(
                "inverse_fold requires proteinmpnn-v48-eps020; it does not use the aligner hikoboshi-mpnn-d64 package"),
            nullptr};
  }
  if (!package_name_matches(descriptor, requested_package)) {
    return {invalid(
                "inverse_fold request package does not match the resolved engine package"),
            nullptr};
  }
  const hiko_u::WeightsHandle weights =
      descriptor.compatibility_views.weights;
  if (weights.opaque == nullptr) {
    return {failed_precondition(
                "ProteinMPNN package has no prepared inverse-fold weights"),
            nullptr};
  }
  return {hiko_u::ok_status(),
          static_cast<const pmp::ProteinMpnnV48020Weights*>(weights.opaque)};
}

}  // namespace

universal::Result<InverseFoldResult> Engine::inverse_fold(
    const InverseFoldRequest& request) const {
  InverseFoldResult result{};
  const auto weights = proteinmpnn_weights(config(), request.package);
  if (!weights.status.ok()) {
    return {weights.status, result};
  }

  hiko_u::StructureView structure = request.structure.residue_count != 0
                                     ? request.structure
                                     : structure_from_coords(request);

  hiko::InverseFoldRequest algorithm_request{};
  algorithm_request.structure = structure;
  algorithm_request.weights = weights.value;
  algorithm_request.options.sampling_temp = request.sampling_temp;
  algorithm_request.options.num_seqs = request.num_seqs;
  algorithm_request.options.seed = request.seed;
  algorithm_request.options.decode_order = to_algorithm_order(request.decode_order);
  algorithm_request.options.backbone_noise = request.backbone_noise;
  algorithm_request.options.keep_log_probs = !request.logprobs_out.empty();

  auto algorithm_result =
      hiko::inverse_fold_proteinmpnn_v48_020(algorithm_request);
  if (!algorithm_result.status.ok()) {
    return {algorithm_result.status, result};
  }

  result.sequences.reserve(algorithm_result.value.sequences.size());
  for (const hiko::InverseFoldSequence& sequence :
       algorithm_result.value.sequences) {
    InverseFoldSequenceResult entry{};
    entry.sequence = sequence.sequence;
    entry.score = sequence.score;
    entry.recovery_vs_native = sequence.recovery_vs_native;
    entry.seed = sequence.seed;
    entry.decode_order = to_api_order(sequence.decode_order);
    result.sequences.push_back(std::move(entry));
  }
  result.logprobs.path = request.logprobs_out;
  result.logprobs.num_seqs = algorithm_result.value.logprobs.num_seqs;
  result.logprobs.residue_count =
      algorithm_result.value.logprobs.residue_count;
  result.logprobs.vocab_size = algorithm_result.value.logprobs.vocab_size;
  result.logprobs.values = std::move(algorithm_result.value.logprobs.values);
  return {hiko_u::ok_status(), std::move(result)};
}

}  // namespace hikoboshi::api

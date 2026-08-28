#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/all_vs_all_layout.hpp>
#include <hikoboshi/io/embedding_loader.hpp>
#include <hikoboshi/io/fasta_writer.hpp>
#include <hikoboshi/io/pdb_writer.hpp>
#include <hikoboshi/io/structure_directory.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace hikoboshi::cli {

bool is_help_flag(std::string_view arg) noexcept;
bool is_option(std::string_view arg, std::string_view name) noexcept;
bool parse_option_assignment(std::string_view arg,
                             std::string_view name,
                             std::string& value);
bool is_rejected_historical_flag(std::string_view arg) noexcept;
bool parse_float(std::string_view text, float& value) noexcept;
hikoboshi::universal::Result<hikoboshi::universal::Backend> parse_backend(
    std::string_view text) noexcept;
int report_status(hikoboshi::universal::Status status);
hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_default_cli_package() noexcept;
hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_cli_package(std::string_view package_id) noexcept;
hikoboshi::universal::Result<hikoboshi::api::Engine> make_engine_with_package(
    hikoboshi::universal::PackageHandle package,
    hikoboshi::universal::Backend backend,
    std::uint32_t thread_count);
bool parse_thread_count(std::string_view text,
                        std::uint32_t& thread_count) noexcept {
  if (text.empty()) {
    return false;
  }
  std::uint64_t parsed = 0;
  for (const char character : text) {
    if (character < '0' || character > '9') {
      return false;
    }
    const std::uint64_t digit =
        static_cast<std::uint64_t>(character - '0');
    if (parsed >
        (std::numeric_limits<std::uint32_t>::max() - digit) / 10U) {
      return false;
    }
    parsed = parsed * 10U + digit;
  }
  thread_count = static_cast<std::uint32_t>(parsed);
  return true;
}
void render_all_vs_all_summary(
    std::ostream& out,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<std::string>& pair_ids,
    const std::vector<std::string>& fasta_paths,
    const std::vector<std::string>& pdb_paths,
    bool include_dual_score_schema);
void render_gap_override_warning(std::ostream& out);
void render_gap_override_warning(std::ostream& out, std::string_view message);

namespace {

constexpr hikoboshi::universal::Status ok() noexcept {
  return {hikoboshi::universal::StatusCode::Ok, ""};
}

constexpr hikoboshi::universal::Status invalid_arguments(
    const char* detail) noexcept {
  return {hikoboshi::universal::StatusCode::InvalidArgument, detail};
}

constexpr hikoboshi::universal::Status unavailable(const char* detail) noexcept {
  return {hikoboshi::universal::StatusCode::Unavailable, detail};
}

constexpr hikoboshi::universal::Status internal_error(
    const char* detail) noexcept {
  return {hikoboshi::universal::StatusCode::InternalError, detail};
}

constexpr std::size_t kPairsPerAutoWorkerFloor = 2;

constexpr hikoboshi::io::StructureDirectoryDiagnostics
    kAllVsAllStructureDirectoryDiagnostics{
        "all-vs-all structure directory is not readable",
        "all-vs-all structure directory could not be read",
        "all-vs-all structure directory contained no PDB or mmCIF files",
    };

bool status_ok(hikoboshi::universal::Status status) noexcept {
  return status.code == hikoboshi::universal::StatusCode::Ok;
}

hikoboshi::api::AlignmentOptions resolve_artifact_alignment_options(
    hikoboshi::api::AlignmentOptions alignment,
    hikoboshi::api::AlignmentMode mode,
    const hikoboshi::universal::PackageDescriptor* descriptor) noexcept {
  const bool artifact_uses_soft_path =
      mode == hikoboshi::api::AlignmentMode::Soft;
  const hikoboshi::universal::AffineGapDefaults* defaults = nullptr;
  if (descriptor != nullptr) {
    defaults = artifact_uses_soft_path ? &descriptor->soft_gaps
                                       : &descriptor->gaps;
  }
  if (hikoboshi::api::is_package_default_gap(alignment.gap_open)) {
    alignment.gap_open = defaults != nullptr ? defaults->gap_open
                                             : hikoboshi::api::kDefaultGapOpen;
  }
  if (hikoboshi::api::is_package_default_gap(alignment.gap_extension)) {
    alignment.gap_extension = defaults != nullptr
                                  ? defaults->gap_extension
                                  : hikoboshi::api::kDefaultGapExtension;
  }
  return alignment;
}

bool starts_with_dash(std::string_view value) noexcept {
  return !value.empty() && value.front() == '-';
}

bool is_mode(std::string_view value) noexcept {
  return value == "structure" || value == "pdb" || value == "cif" ||
         value == "coords" || value == "embeddings" || value == "sequence";
}

bool is_structure_input_mode(std::string_view value) noexcept {
  return value == "structure" || value == "pdb" || value == "cif" ||
         value == "coords";
}

// Tokenization helper for ESM2-8M (mirrors the inline copy in pairwise.cpp).
std::int32_t esm2_aa_token(char c) noexcept {
  static constexpr struct {
    char letter;
    std::int32_t token_id;
  } kTokens[] = {
      // The embedded weight rows are indexed by the training dataset's
      // aatype, which the encoder passes through unchanged
      // (esm2_encoder.forward: `tokens = aatype.clamp(0, 19)`). Recovering
      // aatype from SCOPe40-test-v2's proteins.h5 against the residue letters
      // gives a bijection at purity 1.0000 over ~263k residues: the order is
      // ALPHABETICAL by one-letter code. NOT the AlphaFold-style
      // ARNDCQEGHILKMFPSTWYV this table used to carry, and not FAIR ESM-2's
      // frequency order. PAD=25, CLS/BOS=26, EOS=27, MASK=28 follow.
      //
      // aatype is strictly 0-19 in the training data, so rows 20-24 were
      // never trained; non-standard letters clamp to 19 as training did
      // rather than addressing untrained rows.
      {'A', 0},  {'C', 1},  {'D', 2},  {'E', 3},  {'F', 4},  {'G', 5},
      {'H', 6},  {'I', 7},  {'K', 8},  {'L', 9},  {'M', 10}, {'N', 11},
      {'P', 12}, {'Q', 13}, {'R', 14}, {'S', 15}, {'T', 16}, {'V', 17},
      {'W', 18}, {'Y', 19}, {'B', 19}, {'U', 19}, {'Z', 19}, {'O', 19},
      {'X', 19},
  };
  for (const auto& entry : kTokens) {
    if (entry.letter == c) {
      return entry.token_id;
    }
  }
  return -1;
}

bool is_sequence_input_package(
    const hikoboshi::universal::PackageHandle& package) noexcept {
  if (package.descriptor == nullptr) {
    return false;
  }
  return package.descriptor->execution.architecture_id == "hikoboshi_esm2_v1";
}

bool is_option_or_assignment(std::string_view arg,
                             std::string_view name) {
  std::string ignored;
  return is_option(arg, name) || parse_option_assignment(arg, name, ignored);
}

struct CliGapDefaults {
  float hard_gap_open;
  float hard_gap_extension;
  float soft_gap_open;
  float soft_gap_extension;
  std::string_view hard_message;
  std::string_view soft_message;
  std::string_view both_message;
};

constexpr CliGapDefaults kMpnn64CliGapDefaults{
    hikoboshi::api::kDefaultGapOpen,
    hikoboshi::api::kDefaultGapExtension,
    -3.21337F,
    -0.111704F,
    "user-provided affine gap values override hikoboshi-mpnn-d64 hard-SW calibrated defaults gap_open=-1.40000 and gap_extension=-0.150000",
    "user-provided affine gap values override hikoboshi-mpnn-d64 soft-SW calibrated defaults gap_open=-3.21337 and gap_extension=-0.111704",
    "user-provided affine gap values override hikoboshi-mpnn-d64 hard-SW calibrated defaults (gap_open=-1.40000, gap_extension=-0.150000) and soft-SW calibrated defaults (gap_open=-3.21337, gap_extension=-0.111704)"};

constexpr CliGapDefaults kEsm2CliGapDefaults{
    -1.01982F,
    +0.225736F,
    -6.72805F,
    -0.0159468F,
    "user-provided affine gap values override hikoboshi-esm2-8m hard-SW calibrated defaults gap_open=-1.01982 and gap_extension=+0.225736",
    "user-provided affine gap values override hikoboshi-esm2-8m soft-SW calibrated defaults gap_open=-6.72805 and gap_extension=-0.0159468",
    "user-provided affine gap values override hikoboshi-esm2-8m hard-SW calibrated defaults (gap_open=-1.01982, gap_extension=+0.225736) and soft-SW calibrated defaults (gap_open=-6.72805, gap_extension=-0.0159468)"};

bool explicit_gap_value_overrides(float value, float default_value) noexcept {
  return !hikoboshi::api::is_package_default_gap(value) && value != default_value;
}

bool alignment_mode_runs_hard(hikoboshi::api::AlignmentMode mode) noexcept {
  return mode == hikoboshi::api::AlignmentMode::Hard ||
         mode == hikoboshi::api::AlignmentMode::Both;
}

bool alignment_mode_runs_soft(hikoboshi::api::AlignmentMode mode) noexcept {
  return mode == hikoboshi::api::AlignmentMode::Soft ||
         mode == hikoboshi::api::AlignmentMode::Both;
}

std::string_view gap_override_warning_message(
    const hikoboshi::api::AllVsAllOptions& options,
    const CliGapDefaults& defaults) noexcept {
  const bool hard_overridden =
      alignment_mode_runs_hard(options.mode) &&
      (explicit_gap_value_overrides(options.alignment.gap_open,
                                    defaults.hard_gap_open) ||
       explicit_gap_value_overrides(options.alignment.gap_extension,
                                    defaults.hard_gap_extension));
  const bool soft_overridden =
      alignment_mode_runs_soft(options.mode) &&
      (explicit_gap_value_overrides(options.alignment.gap_open,
                                    defaults.soft_gap_open) ||
       explicit_gap_value_overrides(options.alignment.gap_extension,
                                    defaults.soft_gap_extension));
  if (hard_overridden && soft_overridden) {
    return defaults.both_message;
  }
  if (hard_overridden) {
    return defaults.hard_message;
  }
  if (soft_overridden) {
    return defaults.soft_message;
  }
  return {};
}

void print_all_vs_all_usage(std::ostream& out) {
  out << "usage: hikoboshi all-vs-all "
         "<structure|pdb|cif|coords|embeddings> <inputs...> [options]\n"
      << "\n"
      << "Structure inputs may be files or directories containing .pdb, .ent, "
         ".cif, or .mmcif files.\n"
      << "\n"
      << "options:\n"
      << "  --summary PATH      write TSV summary\n"
      << "  --output-dir DIR    write per-pair alignments/ and pdb/ artifacts\n"
      << "  --include-self      enumerate i <= j instead of i < j\n"
      << "  --gap-open VALUE    affine gap-open score\n"
      << "  --gap-extension VALUE\n"
      << "  --mode hard|soft|both\n"
      << "                     alignment branch (default: hard; soft/both opt in to soft Smith-Waterman, ~6-10x slower)\n"
      << "  --temperature VAL   soft Smith-Waterman temperature (default: 1.0; ignored with --mode hard)\n"
      << "  --package NAME      compiled package ID or alias\n"
      << "  --backend auto|scalar\n"
      << "  --threads N         all-vs-all worker threads: 0 auto, 1 serial\n"
      << "\n"
      << "Scaling: parallelism is across pairs. Speedup is best when pair "
         "count >> thread count. Output is bit-identical regardless of "
         "thread count.\n";
}

struct AllVsAllOptions {
  std::string mode;
  std::vector<std::string> inputs;
  std::string summary_path;
  std::string output_dir;
  hikoboshi::api::AllVsAllOptions request_options{};
  // Mirrors `PairwiseOptions::alignment_gap_*_set`: tracks whether the user
  // supplied --gap-open / --gap-extension explicitly so the sequence-route
  // dispatcher can pass the package-default sentinel for the unset fields.
  bool alignment_gap_open_set = false;
  bool alignment_gap_extension_set = false;
  hikoboshi::universal::PackageHandle package{nullptr, nullptr};
  bool package_selected = false;
  hikoboshi::universal::Backend backend = hikoboshi::universal::Backend::Auto;
  std::uint32_t thread_count = 0;
};

void render_gap_override_warning_if_needed(const AllVsAllOptions& options,
                                           const CliGapDefaults& defaults) {
  if (!options.alignment_gap_open_set &&
      !options.alignment_gap_extension_set) {
    return;
  }
  const std::string_view message =
      gap_override_warning_message(options.request_options, defaults);
  if (!message.empty()) {
    render_gap_override_warning(std::cerr, message);
  }
}

hikoboshi::universal::Status option_value(int& index,
                                        int argc,
                                        char** argv,
                                        std::string& value) {
  if (index + 1 >= argc) {
    return invalid_arguments("option requires a value");
  }
  ++index;
  value = argv[index];
  return ok();
}

bool apply_package_option(std::string_view arg,
                          int& index,
                          int argc,
                          char** argv,
                          AllVsAllOptions& options,
                          hikoboshi::universal::Status& status) {
  std::string value;
  if (!parse_option_assignment(arg, "--package", value) &&
      !is_option(arg, "--package")) {
    return false;
  }
  if (value.empty()) {
    status = option_value(index, argc, argv, value);
    if (!status_ok(status)) {
      return true;
    }
  }
  const hikoboshi::universal::Result<hikoboshi::universal::PackageHandle> package =
      resolve_cli_package(value);
  status = package.status;
  if (!status_ok(status)) {
    return true;
  }
  options.package = package.value;
  options.package_selected = true;
  return true;
}

hikoboshi::universal::Status parse_all_vs_all_options(
    int argc,
    char** argv,
    AllVsAllOptions& options) {
  for (int index = 0; index < argc; ++index) {
    const std::string_view arg{argv[index]};
    // Handle --mode and --temperature before the historical-flag rejection so
    // the soft-SW opt-in/opt-out flags do not collide with `--temperature` in
    // the legacy rejection set; the rejection set continues to block the
    // older `--soft` / `--soft-sw` aliases.
    {
      std::string value;
      if (parse_option_assignment(arg, "--mode", value) ||
          is_option(arg, "--mode")) {
        if (value.empty()) {
          const hikoboshi::universal::Status status =
              option_value(index, argc, argv, value);
          if (!status_ok(status)) return status;
        }
        if (value == "soft") {
          options.request_options.mode = hikoboshi::api::AlignmentMode::Soft;
        } else if (value == "hard") {
          options.request_options.mode = hikoboshi::api::AlignmentMode::Hard;
        } else if (value == "both") {
          options.request_options.mode = hikoboshi::api::AlignmentMode::Both;
        } else {
          return invalid_arguments("--mode must be 'hard', 'soft', or 'both'");
        }
        continue;
      }
      if (parse_option_assignment(arg, "--temperature", value) ||
          is_option(arg, "--temperature")) {
        if (value.empty()) {
          const hikoboshi::universal::Status status =
              option_value(index, argc, argv, value);
          if (!status_ok(status)) return status;
        }
        if (!parse_float(value, options.request_options.temperature)) {
          return invalid_arguments("--temperature must be a float");
        }
        continue;
      }
    }
    if (is_rejected_historical_flag(arg)) {
      return invalid_arguments(
          "that historical or reserved flag is not available in Hikoboshi 0.1.0");
    }
    hikoboshi::universal::Status status{};
    if (apply_package_option(arg, index, argc, argv, options, status)) {
      if (!status_ok(status)) {
        return status;
      }
      continue;
    }
    // These package and scoring options are reserved across encode, pairwise,
    // and all-vs-all. CLI rejection keeps adapters aligned with EngineConfig
    // validation rather than creating adapter-only workflows.
    if (is_option_or_assignment(arg, "--external-package") ||
        is_option_or_assignment(arg, "--package-path") ||
        is_option_or_assignment(arg, "--weights")) {
      return invalid_arguments(
          "external package paths are not supported in Hikoboshi 0.1.0; the CLI uses the compiled hikoboshi-mpnn-d64 registered package");
    }
    if (is_option_or_assignment(arg, "--graph-ir")) {
      return invalid_arguments(
          "graph_ir packages are reserved and not executable in Hikoboshi 0.1.0");
    }
    if (is_option_or_assignment(arg, "--all-atom") ||
        is_option_or_assignment(arg, "--all-atom-package")) {
      return invalid_arguments(
          "all-atom-only package routes are reserved; Hikoboshi 0.1.0 supports backbone structure and coords routes only");
    }
    if (is_option_or_assignment(arg, "--score-method") ||
        is_option_or_assignment(arg, "--cosine") ||
        is_option_or_assignment(arg, "--score-only")) {
      return invalid_arguments(
          "score-method selection is not a Hikoboshi 0.1 public workflow; cosine scoring and score-only APIs are reserved");
    }
    if (is_option_or_assignment(arg, "--algorithm") ||
        is_option_or_assignment(arg, "--alignment") ||
        is_option_or_assignment(arg, "--aligner") ||
        is_option_or_assignment(arg, "--global") ||
        is_option_or_assignment(arg, "--semiglobal")) {
      return invalid_arguments(
          "alignment algorithm selection is not a Hikoboshi 0.1 public workflow; hard_local_affine_sw_v1 is the only supported alignment");
    }
    if (is_option(arg, "--include-self")) {
      options.request_options.include_self = true;
      continue;
    }

    std::string value;
    if (parse_option_assignment(arg, "--summary", value) ||
        is_option(arg, "--summary") || parse_option_assignment(arg, "--metrics", value) ||
        is_option(arg, "--metrics")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      options.summary_path = value;
      continue;
    }
    if (parse_option_assignment(arg, "--output-dir", value) ||
        is_option(arg, "--output-dir")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      options.output_dir = value;
      continue;
    }
    if (parse_option_assignment(arg, "--gap-open", value) ||
        is_option(arg, "--gap-open")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_float(value, options.request_options.alignment.gap_open)) {
        return invalid_arguments("gap-open must be a float");
      }
      options.alignment_gap_open_set = true;
      continue;
    }
    if (parse_option_assignment(arg, "--gap-extension", value) ||
        parse_option_assignment(arg, "--gap-ext", value) ||
        is_option(arg, "--gap-extension") || is_option(arg, "--gap-ext")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_float(value,
                       options.request_options.alignment.gap_extension)) {
        return invalid_arguments("gap-extension must be a float");
      }
      options.alignment_gap_extension_set = true;
      continue;
    }
    if (parse_option_assignment(arg, "--backend", value) ||
        is_option(arg, "--backend")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      const hikoboshi::universal::Result<hikoboshi::universal::Backend> backend =
          parse_backend(value);
      if (!status_ok(backend.status)) {
        return backend.status;
      }
      options.backend = backend.value;
      continue;
    }
    if (parse_option_assignment(arg, "--threads", value) ||
        is_option(arg, "--threads")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_thread_count(value, options.thread_count)) {
        return invalid_arguments(
            "threads must be a non-negative integer in the supported range");
      }
      continue;
    }

    if (starts_with_dash(arg)) {
      return invalid_arguments("unknown all-vs-all option");
    }
    if (options.mode.empty()) {
      if (!is_mode(arg)) {
        return invalid_arguments("all-vs-all requires an explicit input mode");
      }
      options.mode = std::string{arg};
    } else {
      options.inputs.emplace_back(arg);
    }
  }

  if (options.mode.empty()) {
    return invalid_arguments("all-vs-all requires an input mode");
  }
  // The `sequence` input mode takes a single FASTA file containing all
  // records. Embeddings remain positional-file only. Structure-family modes
  // may take one directory positional that expands to many structure files, so
  // their final minimum-count validation happens after expansion.
  if (options.mode == "sequence") {
    if (options.inputs.size() != 1U) {
      return invalid_arguments(
          "sequence all-vs-all expects exactly one FASTA input file");
    }
  } else if (options.mode == "embeddings") {
    const std::size_t minimum_inputs =
        options.request_options.include_self ? 1U : 2U;
    if (options.inputs.size() < minimum_inputs) {
      return invalid_arguments("all-vs-all requires more inputs");
    }
  } else if (options.inputs.empty()) {
    return invalid_arguments("all-vs-all requires more inputs");
  }
  return ok();
}

hikoboshi::io::ArtifactInputId embedding_artifact_id(std::size_t index,
                                                   std::string_view path) {
  return {index, path, {}, "A", "", 1};
}

hikoboshi::io::ArtifactInputId structure_artifact_id(
    std::size_t index,
    std::string_view path,
    const hikoboshi::io::LoadedStructure& structure) {
  std::string_view chain = "A";
  std::string_view model_id{};
  std::int32_t model_index = structure.selected_model_index();
  if (!structure.selected_chain_id().empty()) {
    chain = structure.selected_chain_id();
  }
  if (!structure.selected_model_id().empty()) {
    model_id = structure.selected_model_id();
  }
  return {index,
          path,
          {},
          chain,
          model_id,
          model_index};
}

std::vector<std::string> input_ids(
    const std::vector<hikoboshi::io::ArtifactInputId>& ids) {
  std::vector<std::string> rendered;
  rendered.reserve(ids.size());
  for (const auto& id : ids) {
    rendered.push_back(hikoboshi::io::stable_input_id(id));
  }
  return rendered;
}

std::size_t cli_pair_count(std::size_t input_count,
                           bool include_self) noexcept {
  if (input_count == 0U || (!include_self && input_count == 1U)) {
    return 0U;
  }
  const std::size_t other = include_self ? input_count + 1U : input_count - 1U;
  if (input_count % 2U == 0U) {
    return (input_count / 2U) * other;
  }
  return input_count * (other / 2U);
}

hikoboshi::universal::Status expand_structure_inputs(AllVsAllOptions& options) {
  if (!is_structure_input_mode(options.mode)) {
    return ok();
  }

  std::vector<std::string> expanded;
  expanded.reserve(options.inputs.size());
  for (const std::string& input : options.inputs) {
    std::error_code ec;
    if (std::filesystem::is_directory(input, ec) && !ec) {
      std::vector<std::filesystem::path> files;
      const hikoboshi::universal::Status status =
          hikoboshi::io::list_structure_files(
              input, files, kAllVsAllStructureDirectoryDiagnostics);
      if (!status_ok(status)) {
        return status;
      }
      expanded.reserve(expanded.size() + files.size());
      for (const std::filesystem::path& file : files) {
        expanded.push_back(file.string());
      }
    } else {
      expanded.push_back(input);
    }
  }

  const std::size_t minimum_inputs =
      options.request_options.include_self ? 1U : 2U;
  if (expanded.size() < minimum_inputs) {
    return invalid_arguments("all-vs-all requires more inputs");
  }
  options.inputs = std::move(expanded);
  return ok();
}

std::size_t resolve_cli_thread_count(std::uint32_t requested,
                                     std::size_t pair_count,
                                     unsigned hardware) noexcept {
  if (requested != 0U) {
    return static_cast<std::size_t>(requested);
  }
  if (pair_count == 0U) {
    return 1U;
  }
  const std::size_t hardware_count =
      hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
  const std::size_t pair_cap =
      (pair_count + kPairsPerAutoWorkerFloor - 1U) /
      kPairsPerAutoWorkerFloor;
  return hardware_count < pair_cap ? hardware_count : pair_cap;
}

void render_thread_count_diagnostic(std::ostream& out,
                                    const AllVsAllOptions& options) {
  const unsigned hardware = std::thread::hardware_concurrency();
  const std::size_t pair_count = cli_pair_count(
      options.inputs.size(), options.request_options.include_self);
  const std::size_t resolved =
      resolve_cli_thread_count(options.thread_count, pair_count, hardware);

  out << "[hikoboshi all-vs-all] using " << resolved << " worker thread"
      << (resolved == 1U ? "" : "s") << " (";
  if (options.thread_count == 0U) {
    out << "auto-resolved from hardware_concurrency=" << hardware;
  } else {
    out << "user-set";
  }
  out << ", pair_count=" << pair_count << ")\n";
}

void fill_summary_paths(const hikoboshi::api::AllVsAllResult& result,
                        const std::vector<std::string>& ids,
                        std::vector<std::string>& pair_ids,
                        std::vector<std::string>& fasta_paths,
                        std::vector<std::string>& pdb_paths) {
  pair_ids.clear();
  fasta_paths.clear();
  pdb_paths.clear();
  pair_ids.reserve(result.records.size());
  for (const auto& record : result.records) {
    if (record.query_index < ids.size() && record.target_index < ids.size()) {
      pair_ids.push_back(ids[record.query_index] + "__" +
                         ids[record.target_index]);
    } else {
      pair_ids.push_back({});
    }
  }
}

hikoboshi::universal::Status write_summary_file(
    std::string_view path,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<std::string>& pair_ids,
    const std::vector<std::string>& fasta_paths,
    const std::vector<std::string>& pdb_paths,
    bool include_dual_score_schema) {
  if (path.empty()) {
    return ok();
  }
  std::ofstream out{std::string{path}, std::ios::binary};
  if (!out) {
    return unavailable("all-vs-all summary path is not writable");
  }
  render_all_vs_all_summary(out, result, pair_ids, fasta_paths, pdb_paths,
                            include_dual_score_schema);
  if (!out) {
    return unavailable("all-vs-all summary write failed");
  }
  return ok();
}

hikoboshi::universal::Status create_output_dirs(std::string_view output_dir) {
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path{output_dir} / "alignments", ec);
  if (ec) {
    return unavailable("all-vs-all alignments directory is not writable");
  }
  std::filesystem::create_directories(std::filesystem::path{output_dir} / "pdb",
                                      ec);
  if (ec) {
    return unavailable("all-vs-all pdb directory is not writable");
  }
  return ok();
}

hikoboshi::universal::Status render_structure_artifacts(
    const AllVsAllOptions& options,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<hikoboshi::io::LoadedStructure>& structures,
    const std::vector<hikoboshi::io::ArtifactInputId>& artifact_ids,
    const hikoboshi::universal::PackageDescriptor* descriptor,
    std::vector<std::string>& pair_ids,
    std::vector<std::string>& fasta_paths,
    std::vector<std::string>& pdb_paths) {
  if (options.output_dir.empty()) {
    return ok();
  }

  hikoboshi::universal::Status status = create_output_dirs(options.output_dir);
  if (!status_ok(status)) {
    return status;
  }

  const hikoboshi::universal::Result<std::vector<hikoboshi::io::PairArtifactPaths>>
      layout = hikoboshi::io::build_all_vs_all_artifact_layout(
          options.output_dir,
          {artifact_ids.data(), artifact_ids.size()},
          options.request_options.include_self);
  if (!status_ok(layout.status)) {
    return layout.status;
  }
  if (layout.value.size() != result.records.size()) {
    return internal_error("all-vs-all artifact layout size mismatch");
  }

  pair_ids.clear();
  fasta_paths.clear();
  pdb_paths.clear();
  pair_ids.reserve(layout.value.size());
  fasta_paths.reserve(layout.value.size());
  pdb_paths.reserve(layout.value.size());

  for (std::size_t index = 0; index < result.records.size(); ++index) {
    const hikoboshi::api::PairwiseResultRecord& record = result.records[index];
    const hikoboshi::io::PairArtifactPaths& paths = layout.value[index];
    if (record.query_index >= structures.size() ||
        record.target_index >= structures.size()) {
      return internal_error("all-vs-all record index is out of range");
    }

    status = hikoboshi::io::write_alignment_fasta(
        paths.fasta_path, record.result.path,
        hikoboshi::io::fasta_metadata_from_structure(
            structures[record.query_index].view()),
        hikoboshi::io::fasta_metadata_from_structure(
            structures[record.target_index].view()));
    if (!status_ok(status)) {
      return status;
    }

    hikoboshi::io::PdbWriterOptions writer_options{};
    writer_options.alignment = resolve_artifact_alignment_options(
        options.request_options.alignment, options.request_options.mode,
        descriptor);
    writer_options.backend = options.backend;
    status = hikoboshi::io::write_superposed_pdb(
        paths.pdb_path, record.result,
        structures[record.query_index].view(),
        structures[record.target_index].view(), writer_options);
    if (!status_ok(status)) {
      return status;
    }

    pair_ids.push_back(paths.pair_id);
    fasta_paths.push_back(paths.fasta_path);
    pdb_paths.push_back(paths.pdb_path);
  }
  return ok();
}

int finish_all_vs_all(std::string_view summary_path,
                      const hikoboshi::api::AllVsAllResult& result,
                      const std::vector<std::string>& pair_ids,
                      const std::vector<std::string>& fasta_paths,
                      const std::vector<std::string>& pdb_paths,
                      bool include_dual_score_schema) {
  render_all_vs_all_summary(std::cout, result, pair_ids, fasta_paths,
                            pdb_paths, include_dual_score_schema);
  const hikoboshi::universal::Status status = write_summary_file(
      summary_path, result, pair_ids, fasta_paths, pdb_paths,
      include_dual_score_schema);
  if (!status_ok(status)) {
    return report_status(status);
  }
  return 0;
}

// Adapter that turns the input-id table and `(query_index, target_index)`
// into the same `pair_id` string the legacy renderer used:
// `ids[query_index] + "__" + ids[target_index]`. The Tsv sink invokes this
// per record so the streaming path never has to materialize an
// `AllVsAllResult` to populate the column.
struct StreamingSummaryContext {
  const std::vector<std::string>* ids = nullptr;
};

std::string streaming_pair_id(std::size_t query_index,
                              std::size_t target_index,
                              void* user_data) {
  if (user_data == nullptr) {
    return {};
  }
  auto* context = static_cast<StreamingSummaryContext*>(user_data);
  if (context->ids == nullptr) {
    return {};
  }
  const std::vector<std::string>& ids = *context->ids;
  if (query_index >= ids.size() || target_index >= ids.size()) {
    return {};
  }
  return ids[query_index] + "__" + ids[target_index];
}

// Build the sink output list: stdout always, summary file when the caller
// passed `--summary PATH`. The summary file is opened in binary mode so the
// streamed contents are byte-identical to stdout, matching the legacy
// renderer's behavior.
struct StreamingSummaryOutputs {
  std::ofstream summary_file;
  std::vector<std::ostream*> outputs;
};

hikoboshi::universal::Status open_streaming_summary_outputs(
    std::string_view summary_path,
    StreamingSummaryOutputs& outputs) {
  outputs.outputs.clear();
  outputs.outputs.push_back(&std::cout);
  if (summary_path.empty()) {
    return ok();
  }
  outputs.summary_file.open(std::string{summary_path}, std::ios::binary);
  if (!outputs.summary_file) {
    return unavailable("all-vs-all summary path is not writable");
  }
  outputs.outputs.push_back(&outputs.summary_file);
  return ok();
}

template <typename Request>
hikoboshi::universal::Status run_streaming_summary(
    const hikoboshi::api::Engine& engine,
    const Request& request,
    std::string_view summary_path,
    const std::vector<std::string>& ids) {
  StreamingSummaryOutputs outputs;
  hikoboshi::universal::Status status =
      open_streaming_summary_outputs(summary_path, outputs);
  if (!status_ok(status)) {
    return status;
  }

  StreamingSummaryContext context{};
  context.ids = &ids;
  hikoboshi::api::TsvStreamingAllVsAllSink::Callbacks callbacks{};
  callbacks.pair_id = &streaming_pair_id;
  callbacks.user_data = &context;
  callbacks.include_dual_score_schema =
      alignment_mode_runs_soft(request.options.mode);

  hikoboshi::api::TsvStreamingAllVsAllSink sink(outputs.outputs, callbacks);
  status = hikoboshi::api::stream_all_vs_all(engine, request, sink);
  if (!status_ok(status)) {
    return status;
  }

  std::cout.flush();
  if (!std::cout) {
    return unavailable("all-vs-all summary stdout write failed");
  }
  if (outputs.summary_file.is_open()) {
    outputs.summary_file.flush();
    if (!outputs.summary_file) {
      return unavailable("all-vs-all summary write failed");
    }
  }
  return ok();
}

int run_sequence_all_vs_all(const AllVsAllOptions& options) {
  if (!options.output_dir.empty()) {
    return report_status(
        {hikoboshi::universal::StatusCode::Unavailable,
         "sequence-input all-vs-all artifacts require per-pair FASTA metadata sidecars"});
  }
  if (options.inputs.size() != 1U) {
    return report_status(
        {hikoboshi::universal::StatusCode::InvalidArgument,
         "sequence all-vs-all expects exactly one FASTA file input"});
  }

  std::ifstream in(options.inputs[0]);
  if (!in) {
    return report_status(
        {hikoboshi::universal::StatusCode::Unavailable,
         "sequence all-vs-all input FASTA is not readable"});
  }

  std::vector<std::string> names;
  std::vector<std::vector<std::int32_t>> token_storage;
  std::vector<std::string> sequences;
  std::string line;
  std::string current_name;
  std::string current_seq;
  const auto flush_record = [&]() {
    if (current_name.empty() && current_seq.empty()) return;
    names.push_back(current_name.empty() ? std::string{"seq" +
                                                       std::to_string(names.size())}
                                          : current_name);
    sequences.push_back(current_seq);
    current_name.clear();
    current_seq.clear();
  };
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) continue;
    if (line.front() == '>') {
      flush_record();
      current_name = line.substr(1);
      // Trim trailing whitespace from header.
      while (!current_name.empty() &&
             (current_name.back() == ' ' || current_name.back() == '\t')) {
        current_name.pop_back();
      }
      continue;
    }
    for (char c : line) {
      if (c == ' ' || c == '\t') continue;
      current_seq.push_back(c);
    }
  }
  flush_record();

  if (sequences.empty()) {
    return report_status(
        {hikoboshi::universal::StatusCode::InvalidArgument,
         "sequence all-vs-all FASTA contained no records"});
  }

  token_storage.reserve(sequences.size());
  for (const std::string& seq : sequences) {
    std::vector<std::int32_t> tokens;
    tokens.reserve(seq.size());
    for (char c : seq) {
      if (c == '-' || c == '*') continue;
      const char upper = (c >= 'a' && c <= 'z')
                             ? static_cast<char>(c - 'a' + 'A')
                             : c;
      std::int32_t token = esm2_aa_token(upper);
      if (token < 0) {
        token = 24;
      }
      tokens.push_back(token);
    }
    if (tokens.empty()) {
      return report_status(
          {hikoboshi::universal::StatusCode::InvalidArgument,
           "sequence all-vs-all FASTA contains a record with no valid AA residues"});
    }
    token_storage.push_back(std::move(tokens));
  }

  const hikoboshi::universal::Result<hikoboshi::api::Engine> engine_result =
      make_engine_with_package(options.package, options.backend,
                               options.thread_count);
  if (!status_ok(engine_result.status)) {
    return report_status(engine_result.status);
  }
  const hikoboshi::api::Engine& engine = engine_result.value;

  std::vector<hikoboshi::api::SequenceEntry> entries;
  entries.reserve(token_storage.size());
  for (std::size_t index = 0; index < token_storage.size(); ++index) {
    hikoboshi::api::SequenceEntry entry{};
    entry.name = names[index];
    entry.token_ids = {token_storage[index].data(), token_storage[index].size()};
    entries.push_back(entry);
  }

  hikoboshi::api::AllVsAllSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};
  request.options = options.request_options;
  // Replace unset CLI gap fields with the package-default sentinel so the
  // engine resolves them against the sequence package's calibrated values
  // (see run_sequence_pairwise for the matching pairwise-route comment).
  if (!options.alignment_gap_open_set) {
    request.options.alignment.gap_open =
        hikoboshi::api::kPackageDefaultGapSentinel;
  }
  if (!options.alignment_gap_extension_set) {
    request.options.alignment.gap_extension =
        hikoboshi::api::kPackageDefaultGapSentinel;
  }

  const std::vector<std::string> ids = names;
  const hikoboshi::universal::Status status =
      run_streaming_summary(engine, request, options.summary_path, ids);
  if (!status_ok(status)) {
    return report_status(status);
  }
  render_gap_override_warning_if_needed(options, kEsm2CliGapDefaults);
  return 0;
}

int run_embedding_all_vs_all(const AllVsAllOptions& options) {
  if (!options.output_dir.empty()) {
    return report_status(
        {hikoboshi::universal::StatusCode::Unavailable,
         "embedding-only all-vs-all artifacts require sequence and structure metadata sidecars"});
  }

  std::vector<hikoboshi::io::LoadedEmbedding> loaded;
  std::vector<hikoboshi::universal::EmbeddingView> views;
  std::vector<hikoboshi::io::ArtifactInputId> artifact_ids;
  loaded.reserve(options.inputs.size());
  views.reserve(options.inputs.size());
  artifact_ids.reserve(options.inputs.size());

  for (std::size_t index = 0; index < options.inputs.size(); ++index) {
    hikoboshi::universal::Result<hikoboshi::io::LoadedEmbedding> embedding =
        hikoboshi::io::load_embedding_from_file(options.inputs[index]);
    if (!status_ok(embedding.status)) {
      return report_status(embedding.status);
    }
    loaded.push_back(std::move(embedding.value));
    views.push_back(loaded.back().view());
    artifact_ids.push_back(embedding_artifact_id(index, options.inputs[index]));
  }

  hikoboshi::api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.data(), views.size()};
  request.options = options.request_options;
  hikoboshi::api::EngineConfig config{};
  config.execution.backend = options.backend;
  config.execution.thread_count = options.thread_count;
  const hikoboshi::api::Engine engine{config};

  // p46: embedding all-vs-all has no per-pair PDB/FASTA artifact path, so
  // every CLI call lands on the streaming TSV summary path. The legacy
  // collected path is only reached when an artifact-producing flow needs
  // the AllVsAllResult; embedding mode never sets --output-dir today.
  const std::vector<std::string> ids = input_ids(artifact_ids);
  const hikoboshi::universal::Status status =
      run_streaming_summary(engine, request, options.summary_path, ids);
  if (!status_ok(status)) {
    return report_status(status);
  }
  render_gap_override_warning_if_needed(options, kMpnn64CliGapDefaults);
  return 0;
}

int run_structure_all_vs_all_streaming(
    const AllVsAllOptions& options,
    const hikoboshi::api::Engine& engine,
    const std::vector<hikoboshi::universal::StructureView>& views,
    const std::vector<hikoboshi::io::ArtifactInputId>& artifact_ids) {
  const std::vector<std::string> ids = input_ids(artifact_ids);
  hikoboshi::universal::Status status{};
  if (options.mode == "coords") {
    std::vector<hikoboshi::api::CoordsInputView> coords;
    coords.reserve(views.size());
    for (const auto& view : views) {
      coords.push_back({view.residue_count, view.coordinates, view.atom_sources,
                        view.residue_codes, view.residues});
    }
    hikoboshi::api::AllVsAllCoordsRequest request{};
    request.coords = {coords.data(), coords.size()};
    request.options = options.request_options;
    status = run_streaming_summary(engine, request, options.summary_path, ids);
  } else {
    hikoboshi::api::AllVsAllStructureRequest request{};
    request.structures = {views.data(), views.size()};
    request.options = options.request_options;
    status = run_streaming_summary(engine, request, options.summary_path, ids);
  }
  if (!status_ok(status)) {
    return report_status(status);
  }
  render_gap_override_warning_if_needed(options, kMpnn64CliGapDefaults);
  return 0;
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

int run_structure_all_vs_all_collected(
    const AllVsAllOptions& options,
    const hikoboshi::api::Engine& engine,
    const hikoboshi::universal::PackageDescriptor* descriptor,
    const std::vector<hikoboshi::io::LoadedStructure>& loaded,
    const std::vector<hikoboshi::universal::StructureView>& views,
    const std::vector<hikoboshi::io::ArtifactInputId>& artifact_ids) {
  // p46: kept on the legacy collected path because per-pair FASTA/PDB
  // artifacts under --output-dir need the full AllVsAllResult to render
  // every pair's superposed structure. Streaming the artifact branch is
  // tracked as a separate follow-up; for now this branch reuses the
  // pre-p46 buffered API and the deprecation marker is suppressed at this
  // call site.
  hikoboshi::universal::Result<hikoboshi::api::AllVsAllResult> result{};
  if (options.mode == "coords") {
    std::vector<hikoboshi::api::CoordsInputView> coords;
    coords.reserve(views.size());
    for (const auto& view : views) {
      coords.push_back({view.residue_count, view.coordinates, view.atom_sources,
                        view.residue_codes, view.residues});
    }
    hikoboshi::api::AllVsAllCoordsRequest request{};
    request.coords = {coords.data(), coords.size()};
    request.options = options.request_options;
    result = hikoboshi::api::collect_all_vs_all(engine, request);
  } else {
    hikoboshi::api::AllVsAllStructureRequest request{};
    request.structures = {views.data(), views.size()};
    request.options = options.request_options;
    result = hikoboshi::api::collect_all_vs_all(engine, request);
  }
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }
  render_gap_override_warning_if_needed(options, kMpnn64CliGapDefaults);

  std::vector<std::string> pair_ids;
  std::vector<std::string> fasta_paths;
  std::vector<std::string> pdb_paths;
  fill_summary_paths(result.value, input_ids(artifact_ids), pair_ids,
                     fasta_paths, pdb_paths);

  const hikoboshi::universal::Status artifact_status = render_structure_artifacts(
      options, result.value, loaded, artifact_ids, descriptor, pair_ids,
      fasta_paths, pdb_paths);
  if (!status_ok(artifact_status)) {
    return report_status(artifact_status);
  }
  return finish_all_vs_all(options.summary_path, result.value, pair_ids,
                           fasta_paths, pdb_paths,
                           alignment_mode_runs_soft(
                               options.request_options.mode));
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

int run_structure_all_vs_all(const AllVsAllOptions& options) {
  std::vector<hikoboshi::io::LoadedStructure> loaded;
  std::vector<hikoboshi::universal::StructureView> views;
  std::vector<hikoboshi::io::ArtifactInputId> artifact_ids;
  loaded.reserve(options.inputs.size());
  views.reserve(options.inputs.size());
  artifact_ids.reserve(options.inputs.size());

  for (std::size_t index = 0; index < options.inputs.size(); ++index) {
    hikoboshi::universal::Result<hikoboshi::io::LoadedStructure> structure =
        hikoboshi::io::load_structure_from_file(options.inputs[index]);
    if (!status_ok(structure.status)) {
      return report_status(structure.status);
    }
    loaded.push_back(std::move(structure.value));
    views.push_back(loaded.back().view());
    artifact_ids.push_back(
        structure_artifact_id(index, options.inputs[index], loaded.back()));
  }

  hikoboshi::universal::Result<hikoboshi::universal::PackageHandle> package{};
  if (options.package_selected) {
    package = {{hikoboshi::universal::StatusCode::Ok, ""}, options.package};
  } else {
    package = resolve_default_cli_package();
  }
  if (!status_ok(package.status)) {
    return report_status(package.status);
  }

  const hikoboshi::universal::Result<hikoboshi::api::Engine> engine =
      make_engine_with_package(package.value, options.backend,
                               options.thread_count);
  if (!status_ok(engine.status)) {
    return report_status(engine.status);
  }

  if (options.output_dir.empty()) {
    return run_structure_all_vs_all_streaming(options, engine.value, views,
                                              artifact_ids);
  }
  return run_structure_all_vs_all_collected(
      options, engine.value, package.value.descriptor, loaded, views,
      artifact_ids);
}

}  // namespace

int run_all_vs_all(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_all_vs_all_usage(std::cout);
    return 0;
  }

  AllVsAllOptions options{};
  const hikoboshi::universal::Status status =
      parse_all_vs_all_options(argc, argv, options);
  if (!status_ok(status)) {
    print_all_vs_all_usage(std::cerr);
    return report_status(status);
  }

  if (options.mode == "embeddings") {
    render_thread_count_diagnostic(std::cerr, options);
    return run_embedding_all_vs_all(options);
  }
  const bool sequence_mode = options.mode == "sequence";
  const bool sequence_package =
      options.package_selected && is_sequence_input_package(options.package);
  if (sequence_mode || sequence_package) {
    if (!options.package_selected ||
        !is_sequence_input_package(options.package)) {
      return report_status(
          {hikoboshi::universal::StatusCode::InvalidArgument,
           "sequence all-vs-all requires --package esm2-8m or another sequence-input package"});
    }
    render_thread_count_diagnostic(std::cerr, options);
    return run_sequence_all_vs_all(options);
  }
  const hikoboshi::universal::Status input_status =
      expand_structure_inputs(options);
  if (!status_ok(input_status)) {
    if (input_status.code ==
            hikoboshi::universal::StatusCode::InvalidArgument &&
        std::string_view{input_status.detail} ==
            "all-vs-all requires more inputs") {
      print_all_vs_all_usage(std::cerr);
    }
    return report_status(input_status);
  }
  render_thread_count_diagnostic(std::cerr, options);
  return run_structure_all_vs_all(options);
}

}  // namespace hikoboshi::cli

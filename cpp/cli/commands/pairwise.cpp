#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/embedding_loader.hpp>
#include <hikoboshi/io/fasta_writer.hpp>
#include <hikoboshi/io/pdb_writer.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
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
    hikoboshi::universal::Backend backend);
void render_pairwise_summary(std::ostream& out,
                             std::string_view input_mode,
                             std::string_view query,
                             std::string_view target,
                             const hikoboshi::api::PairwiseResult& result,
                             bool include_dual_score_schema);
void render_pairwise_warnings(std::ostream& out,
                              const hikoboshi::api::PairwiseResult& result);

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

bool status_ok(hikoboshi::universal::Status status) noexcept {
  return status.code == hikoboshi::universal::StatusCode::Ok;
}

bool alignment_mode_runs_soft(hikoboshi::api::AlignmentMode mode) noexcept {
  return mode == hikoboshi::api::AlignmentMode::Soft ||
         mode == hikoboshi::api::AlignmentMode::Both;
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

// Tokenization helper for the hikoboshi-esm2-8m compacted 29-row vocab. The
// mapping mirrors the embedded token table maintained in
// `cpp/weights/embedded_esm2_8m.cpp`; CLI cannot reach into
// `hikoboshi::weights::detail` directly so the table is inlined here. A
// drift between the two tables is caught at packet review.
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
  return -1;  // Unmatched residue codes map to X downstream.
}

bool looks_like_aa_string(std::string_view value) noexcept {
  if (value.empty()) {
    return false;
  }
  for (char c : value) {
    const char upper = (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A')
                                              : c;
    const bool is_aa = (upper >= 'A' && upper <= 'Z') || c == '-' || c == '*';
    if (!is_aa) {
      return false;
    }
  }
  return true;
}

hikoboshi::universal::Status read_single_fasta_sequence(
    const std::string& path, std::string& sequence_out) {
  std::ifstream in(path);
  if (!in) {
    return {hikoboshi::universal::StatusCode::Unavailable,
            "sequence FASTA input is not readable"};
  }
  sequence_out.clear();
  std::string line;
  bool header_seen = false;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line.front() == '>') {
      if (header_seen) {
        break;  // Stop at the next record header; the CLI accepts a single
                // query/target sequence per FASTA input.
      }
      header_seen = true;
      continue;
    }
    if (!header_seen) {
      // Body before any header — treat as raw FASTA without a header line.
      header_seen = true;
    }
    for (char c : line) {
      if (c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
      sequence_out.push_back(c);
    }
  }
  if (sequence_out.empty()) {
    return {hikoboshi::universal::StatusCode::InvalidArgument,
            "sequence FASTA input contains no residue letters"};
  }
  return ok();
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

void print_pairwise_usage(std::ostream& out) {
  out << "usage: hikoboshi pairwise [structure|pdb|cif|coords|embeddings|sequence] "
         "<query> <target> [options]\n"
      << "\n"
      << "options:\n"
      << "  --summary PATH     write TSV metric summary\n"
      << "  --fasta PATH       write alignment FASTA when sequence metadata exists\n"
      << "  --pdb PATH         write superposed PDB for structure inputs\n"
      << "  --gap-open VALUE   affine gap-open score\n"
      << "  --gap-extension VALUE\n"
      << "  --mode hard|soft|both\n"
      << "                    alignment branch (default: hard; soft/both are ~6-10x slower)\n"
      << "  --temperature VAL  soft Smith-Waterman temperature (default: 1.0; ignored with --mode hard)\n"
      << "  --package NAME     compiled package ID or alias\n"
      << "  --backend auto|scalar\n";
}

struct PairwiseOptions {
  std::string mode = "structure";
  std::string query;
  std::string target;
  std::string summary_path;
  std::string fasta_path;
  std::string pdb_path;
  hikoboshi::api::AlignmentOptions alignment{};
  // Track whether the user supplied --gap-open / --gap-extension explicitly
  // so the sequence-route dispatcher can hand the package-default sentinel
  // (NaN) to the engine for the unset fields. Without these flags the
  // sequence path silently injects MPNN-64 defaults (-1.4 / -0.15) into
  // ESM2-8M calls; fe2 pinned the bl5 collapse to that path.
  bool alignment_gap_open_set = false;
  bool alignment_gap_extension_set = false;
  hikoboshi::api::AlignmentMode alignment_mode =
      hikoboshi::api::AlignmentMode::Hard;
  float temperature = hikoboshi::api::kDefaultSoftTemperature;
  hikoboshi::universal::PackageHandle package{nullptr, nullptr};
  bool package_selected = false;
  hikoboshi::universal::Backend backend = hikoboshi::universal::Backend::Auto;
};

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

hikoboshi::universal::Status apply_string_option(std::string_view arg,
                                               int& index,
                                               int argc,
                                               char** argv,
                                               std::string_view name,
                                               std::string& destination) {
  std::string value;
  if (parse_option_assignment(arg, name, value)) {
    destination = value;
    return ok();
  }
  if (is_option(arg, name)) {
    const hikoboshi::universal::Status status =
        option_value(index, argc, argv, value);
    if (!status_ok(status)) {
      return status;
    }
    destination = value;
    return ok();
  }
  return {hikoboshi::universal::StatusCode::Ok, "not matched"};
}

bool apply_package_option(std::string_view arg,
                          int& index,
                          int argc,
                          char** argv,
                          PairwiseOptions& options,
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

hikoboshi::universal::Status parse_pairwise_options(int argc,
                                                  char** argv,
                                                  PairwiseOptions& options) {
  std::vector<std::string> positionals;
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
          options.alignment_mode = hikoboshi::api::AlignmentMode::Soft;
        } else if (value == "hard") {
          options.alignment_mode = hikoboshi::api::AlignmentMode::Hard;
        } else if (value == "both") {
          options.alignment_mode = hikoboshi::api::AlignmentMode::Both;
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
        if (!parse_float(value, options.temperature)) {
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

    status = apply_string_option(
        arg, index, argc, argv, "--summary", options.summary_path);
    if (status.detail != nullptr && std::string_view{status.detail} != "not matched") {
      if (!status_ok(status)) return status;
      continue;
    }
    if (status_ok(status) && status.detail != nullptr &&
        std::string_view{status.detail} != "not matched") {
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--metrics",
                                 options.summary_path);
    if (status.detail != nullptr && std::string_view{status.detail} != "not matched") {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--fasta",
                                 options.fasta_path);
    if (status.detail != nullptr && std::string_view{status.detail} != "not matched") {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--output",
                                 options.fasta_path);
    if (status.detail != nullptr && std::string_view{status.detail} != "not matched") {
      if (!status_ok(status)) return status;
      continue;
    }
    status =
        apply_string_option(arg, index, argc, argv, "--pdb", options.pdb_path);
    if (status.detail != nullptr && std::string_view{status.detail} != "not matched") {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--superpose",
                                 options.pdb_path);
    if (status.detail != nullptr && std::string_view{status.detail} != "not matched") {
      if (!status_ok(status)) return status;
      continue;
    }

    std::string value;
    if (parse_option_assignment(arg, "--gap-open", value) ||
        is_option(arg, "--gap-open")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_float(value, options.alignment.gap_open)) {
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
      if (!parse_float(value, options.alignment.gap_extension)) {
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

    if (starts_with_dash(arg)) {
      return invalid_arguments("unknown pairwise option");
    }
    positionals.emplace_back(arg);
  }

  if (!positionals.empty() && is_mode(positionals.front())) {
    options.mode = positionals.front();
    positionals.erase(positionals.begin());
  }
  if (positionals.size() != 2) {
    return invalid_arguments("pairwise expects exactly two inputs");
  }
  options.query = positionals[0];
  options.target = positionals[1];
  return ok();
}

hikoboshi::io::FastaInputMetadata fasta_metadata_from_embedding(
    const hikoboshi::io::LoadedEmbedding& embedding) noexcept {
  hikoboshi::io::FastaInputMetadata metadata{};
  metadata.input_id = embedding.input_id();
  const hikoboshi::universal::EmbeddingView view = embedding.view();
  metadata.residue_codes = view.residue_codes;
  metadata.residue_count = embedding.residue_count();
  metadata.has_sequence_metadata = embedding.has_residue_metadata();
  return metadata;
}

hikoboshi::universal::Status write_summary_file(
    const PairwiseOptions& options,
    const hikoboshi::api::PairwiseResult& result) {
  if (options.summary_path.empty()) {
    return ok();
  }
  std::ofstream out{options.summary_path, std::ios::binary};
  if (!out) {
    return unavailable("pairwise summary path is not writable");
  }
  const bool include_soft_schema =
      alignment_mode_runs_soft(options.alignment_mode);
  render_pairwise_summary(out, options.mode, options.query, options.target,
                          result, include_soft_schema);
  if (!out) {
    return unavailable("pairwise summary write failed");
  }
  return ok();
}

int finish_pairwise(const PairwiseOptions& options,
                    const hikoboshi::api::PairwiseResult& result) {
  render_pairwise_warnings(std::cerr, result);
  const bool include_soft_schema =
      alignment_mode_runs_soft(options.alignment_mode);
  render_pairwise_summary(std::cout, options.mode, options.query,
                          options.target, result, include_soft_schema);
  const hikoboshi::universal::Status status = write_summary_file(options, result);
  if (!status_ok(status)) {
    return report_status(status);
  }
  return 0;
}

hikoboshi::io::FastaInputMetadata fasta_metadata_for_sequence_letters(
    std::string_view input_id,
    const std::vector<char>& residue_codes) noexcept {
  hikoboshi::io::FastaInputMetadata metadata{};
  metadata.input_id = input_id;
  metadata.residue_codes = {residue_codes.data(), residue_codes.size()};
  metadata.residue_count = residue_codes.size();
  metadata.has_sequence_metadata = !residue_codes.empty();
  return metadata;
}

hikoboshi::universal::Status resolve_sequence_for_arg(
    const std::string& arg, std::vector<std::int32_t>& tokens,
    std::vector<char>& residue_codes) {
  std::string sequence;
  if (looks_like_aa_string(arg)) {
    sequence = arg;
  } else {
    const hikoboshi::universal::Status status =
        read_single_fasta_sequence(arg, sequence);
    if (!status_ok(status)) {
      return status;
    }
  }
  tokens.clear();
  residue_codes.clear();
  tokens.reserve(sequence.size());
  residue_codes.reserve(sequence.size());
  for (char c : sequence) {
    if (c == '-' || c == '*') {
      continue;
    }
    const char upper = (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A')
                                              : c;
    std::int32_t token = esm2_aa_token(upper);
    if (token < 0) {
      token = 24;  // Unknown -> X.
    }
    tokens.push_back(token);
    residue_codes.push_back(upper);
  }
  if (tokens.empty()) {
    return {hikoboshi::universal::StatusCode::InvalidArgument,
            "sequence input must contain at least one valid AA residue"};
  }
  return ok();
}

int run_sequence_pairwise(PairwiseOptions options) {
  options.mode = "sequence";
  std::vector<std::int32_t> query_tokens;
  std::vector<std::int32_t> target_tokens;
  std::vector<char> query_codes;
  std::vector<char> target_codes;
  hikoboshi::universal::Status status =
      resolve_sequence_for_arg(options.query, query_tokens, query_codes);
  if (!status_ok(status)) {
    return report_status(status);
  }
  status = resolve_sequence_for_arg(options.target, target_tokens, target_codes);
  if (!status_ok(status)) {
    return report_status(status);
  }

  const hikoboshi::universal::Result<hikoboshi::api::Engine> engine =
      make_engine_with_package(options.package, options.backend);
  if (!status_ok(engine.status)) {
    return report_status(engine.status);
  }

  hikoboshi::api::PairwiseSequenceRequest request{};
  request.query_token_ids = {query_tokens.data(), query_tokens.size()};
  request.target_token_ids = {target_tokens.data(), target_tokens.size()};
  request.alignment = options.alignment;
  // Replace unset CLI gap fields with the package-default sentinel so the
  // engine resolves them against the resolved sequence package's calibrated
  // gap values. Without this, the sequence route silently inherits the
  // PairwiseOptions struct defaults (MPNN-64 -1.4 / -0.15) on ESM2-8M.
  if (!options.alignment_gap_open_set) {
    request.alignment.gap_open = hikoboshi::api::kPackageDefaultGapSentinel;
  }
  if (!options.alignment_gap_extension_set) {
    request.alignment.gap_extension = hikoboshi::api::kPackageDefaultGapSentinel;
  }
  request.mode = options.alignment_mode;
  request.temperature = options.temperature;
  const hikoboshi::universal::Result<hikoboshi::api::PairwiseResult> result =
      engine.value.pairwise(request);
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }

  if (!options.fasta_path.empty()) {
    status = hikoboshi::io::write_alignment_fasta(
        options.fasta_path, result.value.path,
        fasta_metadata_for_sequence_letters(options.query, query_codes),
        fasta_metadata_for_sequence_letters(options.target, target_codes));
    if (!status_ok(status)) {
      return report_status(status);
    }
  }
  if (!options.pdb_path.empty()) {
    return report_status(
        {hikoboshi::universal::StatusCode::Unavailable,
         "superposed PDB output is not produced for sequence-input alignments"});
  }
  return finish_pairwise(options, result.value);
}

int run_embedding_pairwise(const PairwiseOptions& options) {
  const hikoboshi::universal::Result<hikoboshi::io::LoadedEmbedding> query =
      hikoboshi::io::load_embedding_from_file(options.query);
  if (!status_ok(query.status)) {
    return report_status(query.status);
  }
  const hikoboshi::universal::Result<hikoboshi::io::LoadedEmbedding> target =
      hikoboshi::io::load_embedding_from_file(options.target);
  if (!status_ok(target.status)) {
    return report_status(target.status);
  }

  hikoboshi::api::Engine engine;
  hikoboshi::api::PairwiseEmbeddingRequest request{};
  request.query = query.value.view();
  request.target = target.value.view();
  request.alignment = options.alignment;
  request.mode = options.alignment_mode;
  request.temperature = options.temperature;
  const hikoboshi::universal::Result<hikoboshi::api::PairwiseResult> result =
      engine.pairwise(request);
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }

  if (!options.fasta_path.empty()) {
    const hikoboshi::universal::Status status =
        hikoboshi::io::write_alignment_fasta(
            options.fasta_path, result.value.path,
            fasta_metadata_from_embedding(query.value),
            fasta_metadata_from_embedding(target.value));
    if (!status_ok(status)) {
      return report_status(status);
    }
  }
  if (!options.pdb_path.empty()) {
    return report_status(
        {hikoboshi::universal::StatusCode::Unavailable,
         "superposed PDB output requires structure inputs"});
  }
  return finish_pairwise(options, result.value);
}

int run_structure_pairwise(const PairwiseOptions& options) {
  const hikoboshi::universal::Result<hikoboshi::io::LoadedStructure> query =
      hikoboshi::io::load_structure_from_file(options.query);
  if (!status_ok(query.status)) {
    return report_status(query.status);
  }
  const hikoboshi::universal::Result<hikoboshi::io::LoadedStructure> target =
      hikoboshi::io::load_structure_from_file(options.target);
  if (!status_ok(target.status)) {
    return report_status(target.status);
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
      make_engine_with_package(package.value, options.backend);
  if (!status_ok(engine.status)) {
    return report_status(engine.status);
  }

  hikoboshi::universal::Result<hikoboshi::api::PairwiseResult> result{};
  if (options.mode == "coords") {
    hikoboshi::api::PairwiseCoordsRequest request{};
    request.query = {query.value.view().residue_count,
                     query.value.view().coordinates,
                     query.value.view().atom_sources,
                     query.value.view().residue_codes,
                     query.value.view().residues};
    request.target = {target.value.view().residue_count,
                      target.value.view().coordinates,
                      target.value.view().atom_sources,
                      target.value.view().residue_codes,
                      target.value.view().residues};
    request.alignment = options.alignment;
    request.mode = options.alignment_mode;
    request.temperature = options.temperature;
    result = engine.value.pairwise(request);
  } else {
    hikoboshi::api::PairwiseStructureRequest request{};
    request.query = query.value.view();
    request.target = target.value.view();
    request.alignment = options.alignment;
    request.mode = options.alignment_mode;
    request.temperature = options.temperature;
    result = engine.value.pairwise(request);
  }
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }

  if (!options.fasta_path.empty()) {
    const hikoboshi::universal::Status status =
        hikoboshi::io::write_alignment_fasta(
            options.fasta_path, result.value.path,
            hikoboshi::io::fasta_metadata_from_structure(query.value.view()),
            hikoboshi::io::fasta_metadata_from_structure(target.value.view()));
    if (!status_ok(status)) {
      return report_status(status);
    }
  }
  if (!options.pdb_path.empty()) {
    hikoboshi::io::PdbWriterOptions writer_options{};
    writer_options.alignment = resolve_artifact_alignment_options(
        options.alignment, options.alignment_mode, package.value.descriptor);
    writer_options.backend = options.backend;
    const hikoboshi::universal::Status status =
        hikoboshi::io::write_superposed_pdb(options.pdb_path, result.value,
                                          query.value.view(),
                                          target.value.view(),
                                          writer_options);
    if (!status_ok(status)) {
      return report_status(status);
    }
  }
  return finish_pairwise(options, result.value);
}

}  // namespace

int run_pairwise(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_pairwise_usage(std::cout);
    return 0;
  }

  PairwiseOptions options{};
  const hikoboshi::universal::Status status =
      parse_pairwise_options(argc, argv, options);
  if (!status_ok(status)) {
    print_pairwise_usage(std::cerr);
    return report_status(status);
  }

  if (options.mode == "embeddings") {
    return run_embedding_pairwise(options);
  }
  // Sequence-input dispatch: triggered either by the explicit `sequence`
  // mode keyword or by a sequence-input package selection (e.g.
  // `--package esm2-8m`).
  const bool sequence_mode = options.mode == "sequence";
  const bool sequence_package =
      options.package_selected && is_sequence_input_package(options.package);
  if (sequence_mode || sequence_package) {
    if (!options.package_selected ||
        !is_sequence_input_package(options.package)) {
      return report_status(
          {hikoboshi::universal::StatusCode::InvalidArgument,
           "sequence pairwise requires --package esm2-8m or another sequence-input package"});
    }
    return run_sequence_pairwise(options);
  }
  return run_structure_pairwise(options);
}

}  // namespace hikoboshi::cli

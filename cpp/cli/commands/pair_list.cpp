#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/all_vs_all_layout.hpp>
#include <hikoboshi/io/fasta_writer.hpp>
#include <hikoboshi/io/pdb_writer.hpp>
#include <hikoboshi/io/structure_directory.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
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
bool parse_thread_count(std::string_view text,
                        std::uint32_t& thread_count) noexcept;
int report_status(hikoboshi::universal::Status status);
hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_default_cli_package() noexcept;
hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_cli_package(std::string_view package_id) noexcept;
hikoboshi::universal::Result<hikoboshi::api::Engine> make_engine_with_package(
    hikoboshi::universal::PackageHandle package,
    hikoboshi::universal::Backend backend,
    std::uint32_t thread_count);
void render_all_vs_all_summary(
    std::ostream& out,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<std::string>& pair_ids,
    const std::vector<std::string>& fasta_paths,
    const std::vector<std::string>& pdb_paths,
    bool include_dual_score_schema);

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

bool starts_with_dash(std::string_view value) noexcept {
  return !value.empty() && value.front() == '-';
}

constexpr hikoboshi::io::StructureDirectoryDiagnostics
    kPairListStructureDirectoryDiagnostics{
        "pair-list PDB directory is not readable",
        "pair-list PDB directory could not be read",
        "pair-list PDB directory contained no PDB or mmCIF files",
    };

const char* pair_list_error_detail(const std::string& message) {
  thread_local std::string buffer;
  buffer = message;
  return buffer.c_str();
}

hikoboshi::universal::Status dynamic_invalid_argument(
    const std::string& message) {
  return invalid_arguments(pair_list_error_detail(message));
}

std::string_view trim_surrounding_whitespace(std::string_view text) noexcept {
  while (!text.empty() &&
         (text.front() == ' ' || text.front() == '\t' ||
          text.front() == '\r')) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         (text.back() == ' ' || text.back() == '\t' ||
          text.back() == '\r')) {
    text.remove_suffix(1);
  }
  return text;
}

bool is_option_or_assignment(std::string_view arg, std::string_view name) {
  std::string ignored;
  return is_option(arg, name) || parse_option_assignment(arg, name, ignored);
}

void print_pair_list_usage(std::ostream& out) {
  out << "usage: hikoboshi pair-list --pairs FILE.tsv "
         "(--fasta FILE.fa | PDB_DIR) [options]\n"
      << "\n"
      << "options:\n"
      << "  --pairs FILE.tsv       two-column query_id<TAB>target_id TSV\n"
      << "  --fasta FILE.fa        named sequence FASTA source\n"
      << "  --summary PATH         write TSV summary\n"
      << "  --package NAME         compiled package ID or alias\n"
      << "  --mode hard|soft|both  alignment branch (default: hard)\n"
      << "  --temperature VALUE    soft Smith-Waterman temperature\n"
      << "  --threads N         all-vs-all worker threads: 0 auto, 1 serial\n"
      << "  --gap-open VALUE       affine gap-open score (overrides package default)\n"
      << "  --gap-extension VALUE  affine gap-extension score\n"
      << "  --output-dir DIR       write per-pair alignments/ and pdb/ artifacts\n"
      << "  --parity-mode strict|fast\n";
}

struct PairListOptions {
  std::string pairs_path;
  std::string fasta_path;
  std::string source_dir;
  std::string summary_path;
  std::string output_dir;
  double gap_open = 0.0;
  double gap_extension = 0.0;
  bool gaps_overridden = false;
  hikoboshi::universal::PackageHandle package{nullptr, nullptr};
  bool package_selected = false;
  hikoboshi::api::ParityTag parity = hikoboshi::api::ParityTag::Fast;
  hikoboshi::api::AlignmentMode alignment_mode =
      hikoboshi::api::AlignmentMode::Hard;
  float temperature = hikoboshi::api::kDefaultSoftTemperature;
  std::uint32_t thread_count = 0;
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
                                               std::string& destination,
                                               bool& matched) {
  std::string value;
  if (parse_option_assignment(arg, name, value)) {
    destination = value;
    matched = true;
    return ok();
  }
  if (is_option(arg, name)) {
    const hikoboshi::universal::Status status =
        option_value(index, argc, argv, value);
    if (!status_ok(status)) {
      matched = true;
      return status;
    }
    destination = value;
    matched = true;
    return ok();
  }
  matched = false;
  return ok();
}

bool apply_package_option(std::string_view arg,
                          int& index,
                          int argc,
                          char** argv,
                          PairListOptions& options,
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

bool is_sequence_input_package(
    const hikoboshi::universal::PackageHandle& package) noexcept {
  if (package.descriptor == nullptr) {
    return false;
  }
  return package.descriptor->execution.architecture_id == "hikoboshi_esm2_v1";
}

hikoboshi::universal::Status parse_parity_mode(std::string_view value,
                                             PairListOptions& options) {
  if (value == "strict") {
    options.parity = hikoboshi::api::ParityTag::Strict;
    return ok();
  }
  if (value == "fast") {
    options.parity = hikoboshi::api::ParityTag::Fast;
    return ok();
  }
  return invalid_arguments("--parity-mode must be 'strict' or 'fast'");
}

hikoboshi::universal::Status parse_alignment_mode(std::string_view value,
                                                PairListOptions& options) {
  if (value == "hard") {
    options.alignment_mode = hikoboshi::api::AlignmentMode::Hard;
    return ok();
  }
  if (value == "soft") {
    options.alignment_mode = hikoboshi::api::AlignmentMode::Soft;
    return ok();
  }
  if (value == "both") {
    options.alignment_mode = hikoboshi::api::AlignmentMode::Both;
    return ok();
  }
  return invalid_arguments("--mode must be 'hard', 'soft', or 'both'");
}

hikoboshi::universal::Status parse_pair_list_options(int argc,
                                                   char** argv,
                                                   PairListOptions& options) {
  std::vector<std::string> positionals;
  for (int index = 0; index < argc; ++index) {
    const std::string_view arg{argv[index]};
    {
      std::string value;
      if (parse_option_assignment(arg, "--mode", value) ||
          is_option(arg, "--mode")) {
        if (value.empty()) {
          const hikoboshi::universal::Status status =
              option_value(index, argc, argv, value);
          if (!status_ok(status)) return status;
        }
        const hikoboshi::universal::Status status =
            parse_alignment_mode(value, options);
        if (!status_ok(status)) return status;
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
      if (parse_option_assignment(arg, "--threads", value) ||
          is_option(arg, "--threads")) {
        if (value.empty()) {
          const hikoboshi::universal::Status status =
              option_value(index, argc, argv, value);
          if (!status_ok(status)) return status;
        }
        if (!parse_thread_count(value, options.thread_count)) {
          return invalid_arguments(
              "threads must be a non-negative integer in the supported range");
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
    if (is_option_or_assignment(arg, "--external-package") ||
        is_option_or_assignment(arg, "--package-path") ||
        is_option_or_assignment(arg, "--weights")) {
      return invalid_arguments(
          "external package paths are not supported in Hikoboshi 0.1.0; the CLI uses compiled registered packages");
    }
    if (is_option_or_assignment(arg, "--graph-ir")) {
      return invalid_arguments(
          "graph_ir packages are reserved and not executable in Hikoboshi 0.1.0");
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

    bool matched = false;
    status = apply_string_option(arg, index, argc, argv, "--pairs",
                                 options.pairs_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--fasta",
                                 options.fasta_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    {
      std::string gv;
      status = apply_string_option(arg, index, argc, argv, "--gap-open", gv,
                                   matched);
      if (!status_ok(status)) {
        return status;
      }
      if (matched) {
        options.gap_open = std::strtod(gv.c_str(), nullptr);
        options.gaps_overridden = true;
        continue;
      }
      status = apply_string_option(arg, index, argc, argv, "--gap-extension", gv,
                                   matched);
      if (!status_ok(status)) {
        return status;
      }
      if (matched) {
        options.gap_extension = std::strtod(gv.c_str(), nullptr);
        options.gaps_overridden = true;
        continue;
      }
    }
    status = apply_string_option(arg, index, argc, argv, "--output-dir",
                                 options.output_dir, matched);
    if (!status_ok(status)) {
      return status;
    }
    if (matched) {
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--summary",
                                 options.summary_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--metrics",
                                 options.summary_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    std::string value;
    if (parse_option_assignment(arg, "--parity-mode", value) ||
        is_option(arg, "--parity-mode")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      status = parse_parity_mode(value, options);
      if (!status_ok(status)) return status;
      continue;
    }

    if (starts_with_dash(arg)) {
      return invalid_arguments("unknown pair-list option");
    }
    positionals.emplace_back(arg);
  }

  if (options.pairs_path.empty()) {
    return invalid_arguments("pair-list requires --pairs FILE.tsv");
  }
  if (!options.fasta_path.empty() && !positionals.empty()) {
    return invalid_arguments(
        "pair-list accepts either --fasta FILE.fa or one positional PDB directory, not both");
  }
  if (options.fasta_path.empty()) {
    if (positionals.size() != 1U) {
      return invalid_arguments(
          "pair-list requires --fasta FILE.fa or one positional PDB directory");
    }
    options.source_dir = positionals.front();
  }
  return ok();
}

hikoboshi::universal::Status parse_pairs_tsv(
    const std::string& path,
    std::vector<std::pair<std::string, std::string>>& pairs) {
  std::ifstream in(path);
  if (!in) {
    return unavailable("pair-list pairs file is not readable");
  }
  pairs.clear();
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    const std::string_view trimmed_line =
        trim_surrounding_whitespace(std::string_view{line});
    if (trimmed_line.empty() || trimmed_line.front() == '#') {
      continue;
    }
    const std::size_t tab = trimmed_line.find('\t');
    if (tab == std::string_view::npos ||
        trimmed_line.find('\t', tab + 1U) != std::string_view::npos) {
      return dynamic_invalid_argument(
          "pair-list pairs TSV line " + std::to_string(line_number) +
          " must contain exactly two tab-separated fields");
    }
    const std::string_view query =
        trim_surrounding_whitespace(trimmed_line.substr(0, tab));
    const std::string_view target =
        trim_surrounding_whitespace(trimmed_line.substr(tab + 1U));
    pairs.emplace_back(std::string{query}, std::string{target});
  }
  if (!in.eof()) {
    return unavailable("pair-list pairs file could not be read");
  }
  return ok();
}

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

struct FastaRecords {
  std::vector<std::string> names;
  std::vector<std::vector<std::int32_t>> tokens;
  // Residue letters are retained so the sequence route can emit alignment
  // FASTAs; tokens alone cannot reconstruct them.
  std::vector<std::string> sequences;
};

hikoboshi::universal::Status read_fasta_records(const std::string& path,
                                              FastaRecords& records) {
  std::ifstream in(path);
  if (!in) {
    return unavailable("sequence pair-list input FASTA is not readable");
  }

  records.names.clear();
  records.tokens.clear();
  std::vector<std::string> sequences;
  std::string line;
  std::string current_name;
  std::string current_seq;
  const auto flush_record = [&]() {
    if (current_name.empty() && current_seq.empty()) return;
    records.names.push_back(
        current_name.empty()
            ? std::string{"seq" + std::to_string(records.names.size())}
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
    return invalid_arguments("sequence pair-list FASTA contained no records");
  }

  records.sequences = sequences;
  records.tokens.reserve(sequences.size());
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
      return invalid_arguments(
          "sequence pair-list FASTA contains a record with no valid AA residues");
    }
    records.tokens.push_back(std::move(tokens));
  }
  return ok();
}

std::vector<std::string> pair_ids_from_input_pairs(
    const std::vector<std::pair<std::string, std::string>>& pairs) {
  std::vector<std::string> ids;
  ids.reserve(pairs.size());
  for (const auto& pair : pairs) {
    ids.push_back(pair.first + "__" + pair.second);
  }
  return ids;
}

hikoboshi::universal::Status write_summary_file(
    std::string_view path,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<std::string>& pair_ids,
    bool include_dual_score_schema,
    const std::vector<std::string>& fasta_paths = {},
    const std::vector<std::string>& pdb_paths = {}) {
  if (path.empty()) {
    return ok();
  }
  std::ofstream out{std::string{path}, std::ios::binary};
  if (!out) {
    return unavailable("pair-list summary path is not writable");
  }
  render_all_vs_all_summary(
      out, result, pair_ids, fasta_paths, pdb_paths,
      include_dual_score_schema);
  if (!out) {
    return unavailable("pair-list summary write failed");
  }
  return ok();
}

// Per-pair artifact emission for the structure route. Mirrors all-vs-all's
// writer but drives the layout from the supplied pair list rather than an
// i<j enumeration, using the existing per-pair path primitive.
// Sequence-route artifact emission. Same purpose as the structure-route
// writer, but there are no coordinates, so only the alignment FASTA is
// produced -- that is what carries the correspondence.
hikoboshi::universal::Status write_pair_list_sequence_artifacts(
    const PairListOptions& options,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<std::pair<std::string, std::string>>& input_pairs,
    const FastaRecords& fasta,
    std::vector<std::string>& fasta_paths) {
  fasta_paths.clear();
  if (options.output_dir.empty()) {
    return ok();
  }
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path{options.output_dir} / "alignments", ec);
  if (ec) {
    return hikoboshi::universal::unavailable_status(
        "pair-list could not create the alignments output directory");
  }
  fasta_paths.reserve(result.records.size());
  for (std::size_t index = 0; index < result.records.size(); ++index) {
    const hikoboshi::api::PairwiseResultRecord& record = result.records[index];
    if (record.query_index >= fasta.sequences.size() ||
        record.target_index >= fasta.sequences.size() ||
        index >= input_pairs.size()) {
      return hikoboshi::universal::internal_error_status(
          "pair-list sequence artifact index is out of range");
    }
    hikoboshi::io::ArtifactInputId q{};
    q.input_index = record.query_index;
    q.source_path = input_pairs[index].first;
    q.file_stem = input_pairs[index].first;
    hikoboshi::io::ArtifactInputId t{};
    t.input_index = record.target_index;
    t.source_path = input_pairs[index].second;
    t.file_stem = input_pairs[index].second;
    const hikoboshi::universal::Result<hikoboshi::io::PairArtifactPaths> paths =
        hikoboshi::io::pair_artifact_paths(options.output_dir, q, t);
    if (!status_ok(paths.status)) {
      return paths.status;
    }
    const std::string& qseq = fasta.sequences[record.query_index];
    const std::string& tseq = fasta.sequences[record.target_index];
    hikoboshi::io::FastaInputMetadata qmeta{};
    qmeta.input_id = input_pairs[index].first;
    qmeta.residue_codes = {qseq.data(), qseq.size()};
    qmeta.residue_count = qseq.size();
    qmeta.has_sequence_metadata = true;
    hikoboshi::io::FastaInputMetadata tmeta{};
    tmeta.input_id = input_pairs[index].second;
    tmeta.residue_codes = {tseq.data(), tseq.size()};
    tmeta.residue_count = tseq.size();
    tmeta.has_sequence_metadata = true;
    const hikoboshi::universal::Status status = hikoboshi::io::write_alignment_fasta(
        paths.value.fasta_path, record.result.path, qmeta, tmeta);
    if (!status_ok(status)) {
      return status;
    }
    fasta_paths.push_back(paths.value.fasta_path);
  }
  return ok();
}

hikoboshi::universal::Status write_pair_list_artifacts(
    const PairListOptions& options,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<std::pair<std::string, std::string>>& input_pairs,
    const std::vector<hikoboshi::universal::StructureView>& views,
    std::vector<std::string>& fasta_paths,
    std::vector<std::string>& pdb_paths) {
  fasta_paths.clear();
  pdb_paths.clear();
  if (options.output_dir.empty()) {
    return ok();
  }
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path{options.output_dir} / "alignments", ec);
  if (ec) {
    return hikoboshi::universal::unavailable_status(
        "pair-list could not create the alignments output directory");
  }
  std::filesystem::create_directories(
      std::filesystem::path{options.output_dir} / "pdb", ec);
  if (ec) {
    return hikoboshi::universal::unavailable_status(
        "pair-list could not create the pdb output directory");
  }
  fasta_paths.reserve(result.records.size());
  pdb_paths.reserve(result.records.size());
  for (std::size_t index = 0; index < result.records.size(); ++index) {
    const hikoboshi::api::PairwiseResultRecord& record = result.records[index];
    if (record.query_index >= views.size() ||
        record.target_index >= views.size() || index >= input_pairs.size()) {
      return hikoboshi::universal::internal_error_status(
          "pair-list artifact index is out of range");
    }
    const std::string q_stem =
        hikoboshi::io::file_stem_from_path(input_pairs[index].first);
    const std::string t_stem =
        hikoboshi::io::file_stem_from_path(input_pairs[index].second);
    hikoboshi::io::ArtifactInputId q{};
    q.input_index = record.query_index;
    q.source_path = input_pairs[index].first;
    q.file_stem = q_stem;
    hikoboshi::io::ArtifactInputId t{};
    t.input_index = record.target_index;
    t.source_path = input_pairs[index].second;
    t.file_stem = t_stem;
    const hikoboshi::universal::Result<hikoboshi::io::PairArtifactPaths> paths =
        hikoboshi::io::pair_artifact_paths(options.output_dir, q, t);
    if (!status_ok(paths.status)) {
      return paths.status;
    }
    hikoboshi::universal::Status status = hikoboshi::io::write_alignment_fasta(
        paths.value.fasta_path, record.result.path,
        hikoboshi::io::fasta_metadata_from_structure(views[record.query_index]),
        hikoboshi::io::fasta_metadata_from_structure(views[record.target_index]));
    if (!status_ok(status)) {
      return status;
    }
    hikoboshi::io::PdbWriterOptions writer_options{};
    status = hikoboshi::io::write_superposed_pdb(
        paths.value.pdb_path, record.result, views[record.query_index],
        views[record.target_index], writer_options);
    if (!status_ok(status)) {
      return status;
    }
    fasta_paths.push_back(paths.value.fasta_path);
    pdb_paths.push_back(paths.value.pdb_path);
  }
  return ok();
}

int finish_pair_list(const PairListOptions& options,
                     const hikoboshi::api::AllVsAllResult& result,
                     const std::vector<std::pair<std::string, std::string>>&
                         input_pairs,
                     const std::vector<std::string>& fasta_paths_in = {},
                     const std::vector<std::string>& pdb_paths_in = {}) {
  const std::vector<std::string> pair_ids =
      pair_ids_from_input_pairs(input_pairs);
  const std::vector<std::string> empty_paths;
  const std::vector<std::string>& art_fasta =
      fasta_paths_in.empty() ? empty_paths : fasta_paths_in;
  const std::vector<std::string>& art_pdb =
      pdb_paths_in.empty() ? empty_paths : pdb_paths_in;
  const bool include_soft_schema =
      alignment_mode_runs_soft(options.alignment_mode);
  render_all_vs_all_summary(std::cout, result, pair_ids, art_fasta,
                            art_pdb, include_soft_schema);
  const hikoboshi::universal::Status status =
      write_summary_file(options.summary_path, result, pair_ids,
                         include_soft_schema, art_fasta, art_pdb);
  if (!status_ok(status)) {
    return report_status(status);
  }
  return 0;
}

hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_pair_list_sequence_package(const PairListOptions& options) {
  if (options.package_selected) {
    return {ok(), options.package};
  }
  return resolve_cli_package("esm2-8m");
}

int run_sequence_pair_list(
    const PairListOptions& options,
    const std::vector<std::pair<std::string, std::string>>& pairs) {
  const hikoboshi::universal::Result<hikoboshi::universal::PackageHandle> package =
      resolve_pair_list_sequence_package(options);
  if (!status_ok(package.status)) {
    return report_status(package.status);
  }
  if (!is_sequence_input_package(package.value)) {
    return report_status(
        {hikoboshi::universal::StatusCode::InvalidArgument,
         "sequence pair-list requires --package esm2-8m or another sequence-input package"});
  }

  FastaRecords fasta;
  hikoboshi::universal::Status status =
      read_fasta_records(options.fasta_path, fasta);
  if (!status_ok(status)) {
    return report_status(status);
  }

  const hikoboshi::universal::Result<hikoboshi::api::Engine> engine =
      make_engine_with_package(package.value, hikoboshi::universal::Backend::Auto,
                               options.thread_count);
  if (!status_ok(engine.status)) {
    return report_status(engine.status);
  }

  std::vector<hikoboshi::api::SequenceEntry> entries;
  entries.reserve(fasta.tokens.size());
  for (std::size_t index = 0; index < fasta.tokens.size(); ++index) {
    hikoboshi::api::SequenceEntry entry{};
    entry.name = fasta.names[index];
    entry.token_ids = {fasta.tokens[index].data(), fasta.tokens[index].size()};
    entries.push_back(entry);
  }

  hikoboshi::api::PairListSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};
  request.pairs = pairs;
  request.parity = options.parity;
  if (options.gaps_overridden) {
    request.options.alignment.gap_open = options.gap_open;
    request.options.alignment.gap_extension = options.gap_extension;
  } else {
    request.options.alignment.gap_open =
        hikoboshi::api::kPackageDefaultGapSentinel;
    request.options.alignment.gap_extension =
        hikoboshi::api::kPackageDefaultGapSentinel;
  }
  request.options.mode = options.alignment_mode;
  request.options.temperature = options.temperature;

  const hikoboshi::universal::Result<hikoboshi::api::AllVsAllResult> result =
      engine.value.collect_pair_list(request);
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }
  std::vector<std::string> seq_artifact_fasta;
  status = write_pair_list_sequence_artifacts(options, result.value, pairs,
                                              fasta, seq_artifact_fasta);
  if (!status_ok(status)) {
    return report_status(status);
  }
  return finish_pair_list(options, result.value, pairs, seq_artifact_fasta);
}

int run_structure_pair_list(
    const PairListOptions& options,
    const std::vector<std::pair<std::string, std::string>>& pairs) {
  std::vector<std::filesystem::path> files;
  hikoboshi::universal::Status status =
      hikoboshi::io::list_structure_files(
          options.source_dir, files, kPairListStructureDirectoryDiagnostics);
  if (!status_ok(status)) {
    return report_status(status);
  }

  std::vector<hikoboshi::io::LoadedStructure> loaded;
  std::vector<hikoboshi::universal::StructureView> views;
  loaded.reserve(files.size());
  views.reserve(files.size());
  for (const std::filesystem::path& path : files) {
    hikoboshi::universal::Result<hikoboshi::io::LoadedStructure> structure =
        hikoboshi::io::load_structure_from_file(path.string());
    if (!status_ok(structure.status)) {
      return report_status(structure.status);
    }
    loaded.push_back(std::move(structure.value));
    views.push_back(loaded.back().view());
  }

  hikoboshi::universal::Result<hikoboshi::universal::PackageHandle> package{};
  if (options.package_selected) {
    package = {ok(), options.package};
  } else {
    package = resolve_default_cli_package();
  }
  if (!status_ok(package.status)) {
    return report_status(package.status);
  }

  const hikoboshi::universal::Result<hikoboshi::api::Engine> engine =
      make_engine_with_package(package.value, hikoboshi::universal::Backend::Auto,
                               options.thread_count);
  if (!status_ok(engine.status)) {
    return report_status(engine.status);
  }

  hikoboshi::api::PairListStructureRequest request{};
  request.structures = {views.data(), views.size()};
  request.pairs = pairs;
  request.options.mode = options.alignment_mode;
  request.options.temperature = options.temperature;
  const hikoboshi::universal::Result<hikoboshi::api::AllVsAllResult> result =
      engine.value.collect_pair_list(request);
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }
  std::vector<std::string> artifact_fasta;
  std::vector<std::string> artifact_pdb;
  status = write_pair_list_artifacts(options, result.value, pairs, views,
                                     artifact_fasta, artifact_pdb);
  if (!status_ok(status)) {
    return report_status(status);
  }
  return finish_pair_list(options, result.value, pairs, artifact_fasta,
                          artifact_pdb);
}

}  // namespace

int run_pair_list(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_pair_list_usage(std::cout);
    return 0;
  }

  PairListOptions options{};
  hikoboshi::universal::Status status =
      parse_pair_list_options(argc, argv, options);
  if (!status_ok(status)) {
    print_pair_list_usage(std::cerr);
    return report_status(status);
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  status = parse_pairs_tsv(options.pairs_path, pairs);
  if (!status_ok(status)) {
    return report_status(status);
  }

  if (!options.fasta_path.empty()) {
    return run_sequence_pair_list(options, pairs);
  }
  return run_structure_pair_list(options, pairs);
}

}  // namespace hikoboshi::cli

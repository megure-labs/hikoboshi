#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <fstream>
#include <iostream>
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
void render_encode_summary(std::ostream& out,
                           std::string_view input_mode,
                           std::string_view input,
                           const hikoboshi::api::EncodeResult& result);

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

bool starts_with_dash(std::string_view value) noexcept {
  return !value.empty() && value.front() == '-';
}

bool is_mode(std::string_view value) noexcept {
  return value == "structure" || value == "pdb" || value == "cif" ||
         value == "coords" || value == "sequence";
}

std::int32_t esm2_aa_token(char c) noexcept {
  static constexpr struct {
    char letter;
    std::int32_t token_id;
  } kTokens[] = {
      // Embedded weight rows are indexed by the training dataset aatype, which
      // the encoder passes through unchanged (esm2_encoder.forward:
      // `tokens = aatype.clamp(0, 19)`). That order is ALPHABETICAL by
      // one-letter code -- verified at purity 1.0000 over ~263k residues of
      // SCOPe40-test-v2. aatype is strictly 0-19 there, so rows 20-24 were
      // never trained and non-standard letters clamp to 19 as training did.
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

bool is_sequence_input_package(
    const hikoboshi::universal::PackageHandle& package) noexcept {
  if (package.descriptor == nullptr) {
    return false;
  }
  return package.descriptor->execution.architecture_id == "hikoboshi_esm2_v1";
}

hikoboshi::universal::Status read_single_fasta_sequence(
    const std::string& path, std::string& sequence_out) {
  std::ifstream in(path);
  if (!in) {
    return {hikoboshi::universal::StatusCode::Unavailable,
            "encode sequence FASTA input is not readable"};
  }
  sequence_out.clear();
  std::string line;
  bool header_seen = false;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line.front() == '>') {
      if (header_seen) break;
      header_seen = true;
      continue;
    }
    if (!header_seen) header_seen = true;
    for (char c : line) {
      if (c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
      sequence_out.push_back(c);
    }
  }
  if (sequence_out.empty()) {
    return {hikoboshi::universal::StatusCode::InvalidArgument,
            "encode sequence FASTA input contains no residue letters"};
  }
  return ok();
}

bool is_option_or_assignment(std::string_view arg,
                             std::string_view name) {
  std::string ignored;
  return is_option(arg, name) || parse_option_assignment(arg, name, ignored);
}

void print_encode_usage(std::ostream& out) {
  out << "usage: hikoboshi encode [structure|pdb|cif|coords] <input> [options]\n"
      << "\n"
      << "options:\n"
      << "  --summary PATH      write TSV encode summary\n"
      << "  --package NAME      compiled package ID or alias\n"
      << "  --backend auto|scalar\n";
}

struct EncodeOptions {
  std::string mode = "structure";
  std::string input;
  std::string summary_path;
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

bool apply_package_option(std::string_view arg,
                          int& index,
                          int argc,
                          char** argv,
                          EncodeOptions& options,
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

hikoboshi::universal::Status parse_encode_options(int argc,
                                                char** argv,
                                                EncodeOptions& options) {
  std::vector<std::string> positionals;
  for (int index = 0; index < argc; ++index) {
    const std::string_view arg{argv[index]};
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
      return invalid_arguments("unknown encode option");
    }
    positionals.emplace_back(arg);
  }

  if (!positionals.empty() && is_mode(positionals.front())) {
    options.mode = positionals.front();
    positionals.erase(positionals.begin());
  }
  if (positionals.size() != 1) {
    return invalid_arguments("encode expects exactly one input");
  }
  options.input = positionals.front();
  return ok();
}

hikoboshi::universal::Status write_summary_file(
    const EncodeOptions& options,
    const hikoboshi::api::EncodeResult& result) {
  if (options.summary_path.empty()) {
    return ok();
  }
  std::ofstream out{options.summary_path, std::ios::binary};
  if (!out) {
    return unavailable("encode summary path is not writable");
  }
  render_encode_summary(out, options.mode, options.input, result);
  if (!out) {
    return unavailable("encode summary write failed");
  }
  return ok();
}

}  // namespace

int run_encode_sequence(EncodeOptions options) {
  options.mode = "sequence";
  hikoboshi::universal::Status status{hikoboshi::universal::StatusCode::Ok, ""};
  std::string sequence;
  if (looks_like_aa_string(options.input)) {
    sequence = options.input;
  } else {
    status = read_single_fasta_sequence(options.input, sequence);
    if (!status_ok(status)) {
      return report_status(status);
    }
  }

  std::vector<std::int32_t> tokens;
  tokens.reserve(sequence.size());
  for (char c : sequence) {
    if (c == '-' || c == '*') continue;
    const char upper = (c >= 'a' && c <= 'z')
                           ? static_cast<char>(c - 'a' + 'A')
                           : c;
    std::int32_t token = esm2_aa_token(upper);
    if (token < 0) token = 24;
    tokens.push_back(token);
  }
  if (tokens.empty()) {
    return report_status(
        {hikoboshi::universal::StatusCode::InvalidArgument,
         "encode sequence input must contain at least one valid AA residue"});
  }

  const hikoboshi::universal::Result<hikoboshi::api::Engine> engine =
      make_engine_with_package(options.package, options.backend);
  if (!status_ok(engine.status)) {
    return report_status(engine.status);
  }

  hikoboshi::api::EncodeSequenceRequest request{};
  request.token_ids = {tokens.data(), tokens.size()};
  const hikoboshi::universal::Result<hikoboshi::api::EncodeResult> result =
      engine.value.encode(request);
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }

  render_encode_summary(std::cout, options.mode, options.input, result.value);
  status = write_summary_file(options, result.value);
  if (!status_ok(status)) {
    return report_status(status);
  }
  return 0;
}

int run_encode(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_encode_usage(std::cout);
    return 0;
  }

  EncodeOptions options{};
  hikoboshi::universal::Status status =
      parse_encode_options(argc, argv, options);
  if (!status_ok(status)) {
    print_encode_usage(std::cerr);
    return report_status(status);
  }

  const bool sequence_mode = options.mode == "sequence";
  const bool sequence_package =
      options.package_selected && is_sequence_input_package(options.package);
  if (sequence_mode || sequence_package) {
    if (!options.package_selected ||
        !is_sequence_input_package(options.package)) {
      return report_status(
          {hikoboshi::universal::StatusCode::InvalidArgument,
           "encode sequence requires --package esm2-8m or another sequence-input package"});
    }
    return run_encode_sequence(options);
  }

  const hikoboshi::universal::Result<hikoboshi::io::LoadedStructure> loaded =
      hikoboshi::io::load_structure_from_file(options.input);
  if (!status_ok(loaded.status)) {
    return report_status(loaded.status);
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

  hikoboshi::universal::Result<hikoboshi::api::EncodeResult> result{};
  if (options.mode == "coords") {
    const hikoboshi::universal::StructureView view = loaded.value.view();
    hikoboshi::api::EncodeCoordsRequest request{};
    request.coords = {view.residue_count, view.coordinates, view.atom_sources,
                      view.residue_codes, view.residues};
    result = engine.value.encode(request);
  } else {
    hikoboshi::api::EncodeStructureRequest request{};
    request.structure = loaded.value.view();
    result = engine.value.encode(request);
  }
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }

  render_encode_summary(std::cout, options.mode, options.input, result.value);
  status = write_summary_file(options, result.value);
  if (!status_ok(status)) {
    return report_status(status);
  }
  return 0;
}

}  // namespace hikoboshi::cli

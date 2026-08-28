#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/design_fasta_writer.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace hikoboshi::cli {

bool is_help_flag(std::string_view arg) noexcept;
bool is_option(std::string_view arg, std::string_view name) noexcept;
bool parse_option_assignment(std::string_view arg,
                             std::string_view name,
                             std::string& value);
bool is_rejected_historical_flag(std::string_view arg) noexcept;
bool parse_float(std::string_view text, float& value) noexcept;
int report_status(hikoboshi::universal::Status status);
hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_default_design_package() noexcept;
hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_cli_package(std::string_view package_id) noexcept;
hikoboshi::universal::Result<hikoboshi::api::Engine> make_engine_with_package(
    hikoboshi::universal::PackageHandle package,
    hikoboshi::universal::Backend backend);

namespace {

constexpr hikoboshi::universal::Status ok() noexcept {
  return {hikoboshi::universal::StatusCode::Ok, ""};
}

constexpr hikoboshi::universal::Status invalid_arguments(
    const char* detail) noexcept {
  return {hikoboshi::universal::StatusCode::InvalidArgument, detail};
}

bool status_ok(hikoboshi::universal::Status status) noexcept {
  return status.code == hikoboshi::universal::StatusCode::Ok;
}

bool is_option_or_assignment(std::string_view arg,
                             std::string_view name) {
  std::string ignored;
  return is_option(arg, name) || parse_option_assignment(arg, name, ignored);
}

void print_design_usage(std::ostream& out) {
  out << "usage: hikoboshi design --pdb INPUT.pdb --out OUT.fasta [options]\n"
      << "\n"
      << "options:\n"
      << "  --num-seqs N             number of sequences to design (default 1)\n"
      << "  --sampling-temp T        sampling temperature (default 0.1)\n"
      << "  --seed N                 deterministic sampling seed (default 0)\n"
      << "  --decode-order random|n_to_c\n"
      << "  --package NAME           compiled ProteinMPNN package ID or alias\n"
      << "  --backbone-noise SIGMA   coordinate noise sigma (default 0.0)\n"
      << "  --out-logprobs PATH      write opt-in [N,L,21] log-prob NPZ\n";
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

bool parse_size(std::string_view text, std::size_t& value) noexcept {
  std::string copy{text};
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(copy.c_str(), &end, 10);
  if (errno != 0 || end == copy.c_str() || *end != '\0') {
    return false;
  }
  if (parsed > static_cast<unsigned long long>(
                   std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

bool parse_seed(std::string_view text, std::uint64_t& value) noexcept {
  std::string copy{text};
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(copy.c_str(), &end, 10);
  if (errno != 0 || end == copy.c_str() || *end != '\0') {
    return false;
  }
  value = static_cast<std::uint64_t>(parsed);
  return true;
}

hikoboshi::api::InverseFoldDecodeOrder parse_decode_order(
    std::string_view text,
    hikoboshi::universal::Status& status) {
  if (text == "random") {
    status = ok();
    return hikoboshi::api::InverseFoldDecodeOrder::Random;
  }
  if (text == "n_to_c" || text == "n-to-c") {
    status = ok();
    return hikoboshi::api::InverseFoldDecodeOrder::NToC;
  }
  status = invalid_arguments("decode-order must be random or n_to_c");
  return hikoboshi::api::InverseFoldDecodeOrder::Random;
}

struct DesignOptions {
  std::string pdb_path;
  std::string out_path;
  std::string out_logprobs_path;
  std::string package_id = "proteinmpnn-v48-eps020";
  hikoboshi::universal::PackageHandle package{nullptr, nullptr};
  bool package_selected = false;
  float sampling_temp = 0.1F;
  std::size_t num_seqs = 1;
  std::uint64_t seed = 0;
  hikoboshi::api::InverseFoldDecodeOrder decode_order =
      hikoboshi::api::InverseFoldDecodeOrder::Random;
  float backbone_noise = 0.0F;
};

bool apply_package_option(std::string_view arg,
                          int& index,
                          int argc,
                          char** argv,
                          DesignOptions& options,
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
  const auto package = resolve_cli_package(value);
  status = package.status;
  if (!status_ok(status)) {
    return true;
  }
  options.package = package.value;
  options.package_id = value;
  options.package_selected = true;
  return true;
}

hikoboshi::universal::Status parse_design_options(int argc,
                                                char** argv,
                                                DesignOptions& options) {
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

    if (is_option_or_assignment(arg, "--external-package") ||
        is_option_or_assignment(arg, "--package-path") ||
        is_option_or_assignment(arg, "--weights")) {
      return invalid_arguments(
          "external package paths are not supported in Hikoboshi 0.1.0; design uses the compiled ProteinMPNN registered package");
    }

    std::string value;
    if (parse_option_assignment(arg, "--pdb", value) ||
        is_option(arg, "--pdb")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      options.pdb_path = value;
      continue;
    }
    if (parse_option_assignment(arg, "--out", value) ||
        is_option(arg, "--out")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      options.out_path = value;
      continue;
    }
    if (parse_option_assignment(arg, "--out-logprobs", value) ||
        is_option(arg, "--out-logprobs")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      options.out_logprobs_path = value;
      continue;
    }
    if (parse_option_assignment(arg, "--sampling-temp", value) ||
        is_option(arg, "--sampling-temp")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_float(value, options.sampling_temp)) {
        return invalid_arguments("sampling-temp must be a finite number");
      }
      continue;
    }
    if (parse_option_assignment(arg, "--backbone-noise", value) ||
        is_option(arg, "--backbone-noise")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_float(value, options.backbone_noise)) {
        return invalid_arguments("backbone-noise must be a finite number");
      }
      continue;
    }
    if (parse_option_assignment(arg, "--num-seqs", value) ||
        is_option(arg, "--num-seqs")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_size(value, options.num_seqs) || options.num_seqs == 0) {
        return invalid_arguments("num-seqs must be a positive integer");
      }
      continue;
    }
    if (parse_option_assignment(arg, "--seed", value) ||
        is_option(arg, "--seed")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      if (!parse_seed(value, options.seed)) {
        return invalid_arguments("seed must be a non-negative integer");
      }
      continue;
    }
    if (parse_option_assignment(arg, "--decode-order", value) ||
        is_option(arg, "--decode-order")) {
      if (value.empty()) {
        status = option_value(index, argc, argv, value);
        if (!status_ok(status)) return status;
      }
      options.decode_order = parse_decode_order(value, status);
      if (!status_ok(status)) {
        return status;
      }
      continue;
    }

    if (!arg.empty() && arg.front() == '-') {
      return invalid_arguments("unknown design option");
    }
    return invalid_arguments("design requires explicit --pdb and --out options");
  }

  if (options.pdb_path.empty()) {
    return invalid_arguments("design requires --pdb");
  }
  if (options.out_path.empty()) {
    return invalid_arguments("design requires --out");
  }
  return ok();
}

}  // namespace

int run_design(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_design_usage(std::cout);
    return 0;
  }

  DesignOptions options{};
  hikoboshi::universal::Status status =
      parse_design_options(argc, argv, options);
  if (!status_ok(status)) {
    return report_status(status);
  }

  if (!options.package_selected) {
    const auto package = resolve_default_design_package();
    if (!status_ok(package.status)) {
      return report_status(package.status);
    }
    options.package = package.value;
  }

  const auto loaded = hikoboshi::io::load_structure_from_file(options.pdb_path);
  if (!status_ok(loaded.status)) {
    return report_status(loaded.status);
  }

  const auto engine =
      make_engine_with_package(options.package, hikoboshi::universal::Backend::Auto);
  if (!status_ok(engine.status)) {
    return report_status(engine.status);
  }

  hikoboshi::api::InverseFoldRequest request{};
  request.structure = loaded.value.view();
  request.pdb_path = options.pdb_path;
  request.package = options.package_id;
  request.sampling_temp = options.sampling_temp;
  request.num_seqs = options.num_seqs;
  request.seed = options.seed;
  request.decode_order = options.decode_order;
  request.backbone_noise = options.backbone_noise;
  request.logprobs_out = options.out_logprobs_path;

  const auto result = engine.value.inverse_fold(request);
  if (!status_ok(result.status)) {
    return report_status(result.status);
  }

  hikoboshi::io::DesignFastaWriterOptions writer_options{};
  writer_options.sampling_temp = options.sampling_temp;
  status =
      hikoboshi::io::write_design_fasta(options.out_path, result.value,
                                      writer_options);
  if (!status_ok(status)) {
    return report_status(status);
  }

  if (!options.out_logprobs_path.empty()) {
    status = hikoboshi::io::write_inverse_fold_logprobs_npz(
        options.out_logprobs_path, result.value);
    if (!status_ok(status)) {
      return report_status(status);
    }
  }

  return 0;
}

}  // namespace hikoboshi::cli

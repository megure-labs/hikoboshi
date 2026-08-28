#include <hikoboshi/api/score_alignment.hpp>
#include <hikoboshi/errors/format.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <ostream>
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
int report_status(hikoboshi::universal::Status status);

namespace {

using hikoboshi::api::ScoreAlignmentRequest;
using hikoboshi::api::ScoreAlignmentResult;
using hikoboshi::universal::AlignmentPath;
using hikoboshi::universal::AlignmentStep;
using hikoboshi::universal::kAlignmentGapSentinel;
using hikoboshi::universal::MetricValue;
using hikoboshi::universal::Status;
using hikoboshi::universal::StatusCode;

constexpr int kMetricPrecision = 6;

constexpr Status ok() noexcept {
  return {StatusCode::Ok, ""};
}

constexpr Status invalid_arguments(const char* detail) noexcept {
  return {StatusCode::InvalidArgument, detail};
}

constexpr Status unavailable(const char* detail) noexcept {
  return {StatusCode::Unavailable, detail};
}

bool status_ok(Status status) noexcept {
  return status.code == StatusCode::Ok;
}

bool starts_with_dash(std::string_view value) noexcept {
  return !value.empty() && value.front() == '-';
}

void print_score_alignment_usage(std::ostream& out) {
  out << "usage: hikoboshi score-alignment --query Q --target T "
         "--correspondences PAIRS.tsv [options]\n"
      << "\n"
      << "options:\n"
      << "  --query PATH              query structure (PDB or mmCIF)\n"
      << "  --target PATH             target structure\n"
      << "  --correspondences PATH    TSV with two columns: "
         "q_residue_index<TAB>t_residue_index;\n"
      << "                            one row per aligned pair, 0-based "
         "indexing into the\n"
      << "                            structure's residue table\n"
      << "  --output-format FORMAT    'tsv' (default) or 'json'\n"
      << "  --summary PATH            write output to file (default: stdout)\n";
}

struct ScoreAlignmentOptions {
  std::string query_path;
  std::string target_path;
  std::string correspondences_path;
  std::string summary_path;
  std::string output_format = "tsv";
};

Status option_value(int& index, int argc, char** argv, std::string& value) {
  if (index + 1 >= argc) {
    return invalid_arguments("option requires a value");
  }
  ++index;
  value = argv[index];
  return ok();
}

Status apply_string_option(std::string_view arg,
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
    matched = true;
    return option_value(index, argc, argv, destination);
  }
  matched = false;
  return ok();
}

Status parse_options(int argc, char** argv, ScoreAlignmentOptions& options) {
  for (int index = 0; index < argc; ++index) {
    const std::string_view arg{argv[index]};
    bool matched = false;

    Status status =
        apply_string_option(arg, index, argc, argv, "--query",
                            options.query_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--target",
                                 options.target_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--correspondences",
                                 options.correspondences_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--summary",
                                 options.summary_path, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }
    status = apply_string_option(arg, index, argc, argv, "--output-format",
                                 options.output_format, matched);
    if (matched) {
      if (!status_ok(status)) return status;
      continue;
    }

    if (starts_with_dash(arg)) {
      return invalid_arguments("unknown score-alignment option");
    }
    return invalid_arguments(
        "score-alignment does not accept positional arguments; use --query, "
        "--target, and --correspondences");
  }

  if (options.query_path.empty()) {
    return invalid_arguments("score-alignment requires --query PATH");
  }
  if (options.target_path.empty()) {
    return invalid_arguments("score-alignment requires --target PATH");
  }
  if (options.correspondences_path.empty()) {
    return invalid_arguments(
        "score-alignment requires --correspondences PATH");
  }
  if (options.output_format != "tsv" && options.output_format != "json") {
    return invalid_arguments(
        "--output-format must be 'tsv' or 'json'");
  }
  return ok();
}

bool parse_index_token(std::string_view token, std::int32_t& value) noexcept {
  if (token.empty()) {
    return false;
  }
  std::string copy{token};
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(copy.c_str(), &end, 10);
  if (errno != 0 || end == copy.c_str() || *end != '\0') {
    return false;
  }
  if (parsed < std::numeric_limits<std::int32_t>::min() ||
      parsed > std::numeric_limits<std::int32_t>::max()) {
    return false;
  }
  value = static_cast<std::int32_t>(parsed);
  return true;
}

std::string_view trim_view(std::string_view value) noexcept {
  std::size_t begin = 0;
  while (begin < value.size() &&
         (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\r')) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin &&
         (value[end - 1] == ' ' || value[end - 1] == '\t' ||
          value[end - 1] == '\r')) {
    --end;
  }
  return value.substr(begin, end - begin);
}

Status load_correspondences(const std::string& path, AlignmentPath& out) {
  std::ifstream input(path);
  if (!input) {
    return unavailable("score-alignment correspondences file is not readable");
  }
  out.steps.clear();
  out.aligned_pairs = 0;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::string_view trimmed = trim_view(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    const std::size_t tab = trimmed.find('\t');
    if (tab == std::string_view::npos) {
      return invalid_arguments(
          "score-alignment correspondences row must be two tab-separated "
          "integer columns");
    }
    const std::string_view first = trim_view(trimmed.substr(0, tab));
    const std::string_view second = trim_view(trimmed.substr(tab + 1));
    if (second.find('\t') != std::string_view::npos) {
      return invalid_arguments(
          "score-alignment correspondences row must contain exactly two "
          "tab-separated columns");
    }
    AlignmentStep step{};
    if (!parse_index_token(first, step.query_index)) {
      return invalid_arguments(
          "score-alignment correspondences first column must be an integer "
          "or the gap sentinel -1");
    }
    if (!parse_index_token(second, step.target_index)) {
      return invalid_arguments(
          "score-alignment correspondences second column must be an integer "
          "or the gap sentinel -1");
    }
    step.residue_score = 0.0F;
    out.steps.push_back(step);
    if (step.query_index != kAlignmentGapSentinel &&
        step.target_index != kAlignmentGapSentinel) {
      ++out.aligned_pairs;
    }
  }
  if (input.bad()) {
    return unavailable("score-alignment correspondences read failed");
  }
  (void)line_number;
  return ok();
}

std::string format_double(double value) {
  std::ostringstream out;
  out << std::setprecision(kMetricPrecision) << value;
  return out.str();
}

std::string format_metric(MetricValue metric) {
  return hikoboshi::errors::format_metric(metric, kMetricPrecision);
}

std::string format_metric_json(MetricValue metric) {
  if (!metric.valid) {
    return "null";
  }
  return format_double(metric.value);
}

void render_tsv(std::ostream& out, const ScoreAlignmentResult& result) {
  out << "rmsd\ttm_score_query\ttm_score_target\tlddt\tidentity"
      << "\tcoverage_query\tcoverage_target\tcoverage_mean\taligned_pairs\n";
  out << format_metric(result.rmsd) << '\t'
      << format_metric(result.tm_score_query) << '\t'
      << format_metric(result.tm_score_target) << '\t'
      << format_metric(result.lddt) << '\t'
      << format_metric(result.identity) << '\t'
      << format_metric(result.coverage_query) << '\t'
      << format_metric(result.coverage_target) << '\t'
      << format_metric(result.coverage_mean) << '\t'
      << result.aligned_pairs << '\n';
}

void render_json(std::ostream& out, const ScoreAlignmentResult& result) {
  out << "{\n"
      << "  \"rmsd\": " << format_metric_json(result.rmsd) << ",\n"
      << "  \"tm_score_query\": "
      << format_metric_json(result.tm_score_query) << ",\n"
      << "  \"tm_score_target\": "
      << format_metric_json(result.tm_score_target) << ",\n"
      << "  \"lddt\": " << format_metric_json(result.lddt) << ",\n"
      << "  \"identity\": " << format_metric_json(result.identity) << ",\n"
      << "  \"coverage_query\": "
      << format_metric_json(result.coverage_query) << ",\n"
      << "  \"coverage_target\": "
      << format_metric_json(result.coverage_target) << ",\n"
      << "  \"coverage_mean\": "
      << format_metric_json(result.coverage_mean) << ",\n"
      << "  \"aligned_pairs\": " << result.aligned_pairs << "\n"
      << "}\n";
}

void render_result(std::ostream& out,
                   const ScoreAlignmentOptions& options,
                   const ScoreAlignmentResult& result) {
  if (options.output_format == "json") {
    render_json(out, result);
  } else {
    render_tsv(out, result);
  }
}

Status write_summary_file(const ScoreAlignmentOptions& options,
                          const ScoreAlignmentResult& result) {
  if (options.summary_path.empty()) {
    return ok();
  }
  std::ofstream out(options.summary_path, std::ios::binary);
  if (!out) {
    return unavailable("score-alignment summary path is not writable");
  }
  render_result(out, options, result);
  if (!out) {
    return unavailable("score-alignment summary write failed");
  }
  return ok();
}

}  // namespace

int run_score_alignment(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_score_alignment_usage(std::cout);
    return 0;
  }

  ScoreAlignmentOptions options{};
  Status status = parse_options(argc, argv, options);
  if (!status_ok(status)) {
    print_score_alignment_usage(std::cerr);
    return report_status(status);
  }

  hikoboshi::universal::Result<hikoboshi::io::LoadedStructure> query =
      hikoboshi::io::load_structure_from_file(options.query_path);
  if (!status_ok(query.status)) {
    return report_status(query.status);
  }
  hikoboshi::universal::Result<hikoboshi::io::LoadedStructure> target =
      hikoboshi::io::load_structure_from_file(options.target_path);
  if (!status_ok(target.status)) {
    return report_status(target.status);
  }

  ScoreAlignmentRequest request{};
  request.query_structure = query.value.view();
  request.target_structure = target.value.view();
  status = load_correspondences(options.correspondences_path,
                                request.correspondences);
  if (!status_ok(status)) {
    return report_status(status);
  }

  ScoreAlignmentResult result{};
  status = hikoboshi::api::score_alignment(request, result);
  if (!status_ok(status)) {
    return report_status(status);
  }

  if (options.summary_path.empty()) {
    render_result(std::cout, options, result);
  } else {
    render_result(std::cout, options, result);
    status = write_summary_file(options, result);
    if (!status_ok(status)) {
      return report_status(status);
    }
  }
  return 0;
}

}  // namespace hikoboshi::cli

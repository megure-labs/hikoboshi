#include <hikoboshi/io/all_vs_all_layout.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hikoboshi::io {
namespace {

std::string padded_input_index(std::size_t zero_based_index) {
  std::ostringstream out;
  out << std::setw(4) << std::setfill('0') << (zero_based_index + 1);
  return out.str();
}

std::string sanitized_or(std::string_view value, std::string_view fallback) {
  std::string sanitized = sanitize_identifier(value.empty() ? fallback : value);
  if (sanitized.empty()) {
    sanitized = std::string{fallback};
  }
  return sanitized;
}

std::string model_suffix(const ArtifactInputId& input) {
  if (!input.model_id.empty()) {
    return sanitized_or(input.model_id, "1");
  }
  const std::int32_t model_index =
      input.model_index <= 0 ? 1 : input.model_index;
  return std::to_string(model_index);
}

std::string join_path(std::string_view base, std::string_view child) {
  std::string joined{base};
  if (!joined.empty() && joined.back() != '/' && joined.back() != '\\') {
    joined.push_back('/');
  }
  joined.append(child);
  return joined;
}

PairArtifactPaths make_pair_paths(std::string_view output_dir,
                                  const ArtifactInputId& query,
                                  const ArtifactInputId& target) {
  PairArtifactPaths paths{};
  paths.query_index = query.input_index;
  paths.target_index = target.input_index;
  paths.query_id = stable_input_id(query);
  paths.target_id = stable_input_id(target);
  paths.pair_id = paths.query_id + "__" + paths.target_id;
  paths.fasta_path =
      join_path(join_path(output_dir, "alignments"), paths.pair_id + ".fasta");
  paths.pdb_path =
      join_path(join_path(output_dir, "pdb"), paths.pair_id + ".pdb");
  return paths;
}

}  // namespace

std::string file_stem_from_path(std::string_view path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::string_view filename =
      slash == std::string_view::npos ? path : path.substr(slash + 1);
  const std::size_t dot = filename.find_last_of('.');
  if (dot == std::string_view::npos || dot == 0) {
    return std::string{filename};
  }
  return std::string{filename.substr(0, dot)};
}

std::string stable_input_id(const ArtifactInputId& input) {
  const std::string stem = input.file_stem.empty()
                               ? file_stem_from_path(input.source_path)
                               : std::string{input.file_stem};
  return padded_input_index(input.input_index) + "_" +
         sanitized_or(stem, "input") + "_" +
         sanitized_or(input.chain_id, "A") + "_model" + model_suffix(input);
}

universal::Result<PairArtifactPaths> pair_artifact_paths(
    std::string_view output_dir,
    const ArtifactInputId& query,
    const ArtifactInputId& target) {
  if (output_dir.empty()) {
    return {universal::invalid_argument_status("output directory must be non-empty"), {}};
  }
  return {universal::ok_status(), make_pair_paths(output_dir, query, target)};
}

universal::Result<std::vector<PairArtifactPaths>>
build_all_vs_all_artifact_layout(
    std::string_view output_dir,
    universal::Span<const ArtifactInputId> inputs,
    bool include_self) {
  if (output_dir.empty()) {
    return {universal::invalid_argument_status("output directory must be non-empty"), {}};
  }
  if (inputs.data == nullptr && inputs.size != 0) {
    return {universal::invalid_argument_status("input descriptor span is invalid"), {}};
  }

  std::vector<PairArtifactPaths> paths;
  for (std::size_t i = 0; i < inputs.size; ++i) {
    const std::size_t j_begin = include_self ? i : i + 1;
    for (std::size_t j = j_begin; j < inputs.size; ++j) {
      paths.push_back(make_pair_paths(output_dir, inputs.data[i],
                                      inputs.data[j]));
    }
  }
  return {universal::ok_status(), std::move(paths)};
}

}  // namespace hikoboshi::io

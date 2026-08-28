#ifndef HIKOBOSHI_IO_ALL_VS_ALL_LAYOUT_HPP
#define HIKOBOSHI_IO_ALL_VS_ALL_LAYOUT_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::io {

struct ArtifactInputId {
  std::size_t input_index = 0;
  std::string_view source_path{};
  std::string_view file_stem{};
  std::string_view chain_id{"A"};
  std::string_view model_id{};
  std::int32_t model_index = 1;
};

struct PairArtifactPaths {
  std::size_t query_index = 0;
  std::size_t target_index = 0;
  std::string query_id;
  std::string target_id;
  std::string pair_id;
  std::string fasta_path;
  std::string pdb_path;
};

std::string sanitize_identifier(std::string_view value);

std::string file_stem_from_path(std::string_view path);

std::string stable_input_id(const ArtifactInputId& input);

[[nodiscard]] universal::Result<PairArtifactPaths> pair_artifact_paths(
    std::string_view output_dir,
    const ArtifactInputId& query,
    const ArtifactInputId& target);

[[nodiscard]] universal::Result<std::vector<PairArtifactPaths>>
build_all_vs_all_artifact_layout(
    std::string_view output_dir,
    universal::Span<const ArtifactInputId> inputs,
    bool include_self = false);

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_ALL_VS_ALL_LAYOUT_HPP

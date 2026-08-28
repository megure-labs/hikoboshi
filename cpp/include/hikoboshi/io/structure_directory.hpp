#ifndef HIKOBOSHI_IO_STRUCTURE_DIRECTORY_HPP
#define HIKOBOSHI_IO_STRUCTURE_DIRECTORY_HPP

#include <filesystem>
#include <vector>

#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::io {

struct StructureDirectoryDiagnostics {
  const char* not_readable;
  const char* read_failed;
  const char* empty;
};

[[nodiscard]] bool is_structure_path(const std::filesystem::path& path);

[[nodiscard]] universal::Status list_structure_files(
    const std::filesystem::path& directory,
    std::vector<std::filesystem::path>& files);

[[nodiscard]] universal::Status list_structure_files(
    const std::filesystem::path& directory,
    std::vector<std::filesystem::path>& files,
    const StructureDirectoryDiagnostics& diagnostics);

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_STRUCTURE_DIRECTORY_HPP

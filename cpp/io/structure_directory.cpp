#include <hikoboshi/io/structure_directory.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>

namespace hikoboshi::io {
namespace {

constexpr StructureDirectoryDiagnostics kDefaultDiagnostics{
    "structure directory is not readable",
    "structure directory could not be read",
    "structure directory contained no PDB or mmCIF files",
};

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

bool ends_with_ascii_case(std::string_view value,
                          std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         ascii_case_equal(value.substr(value.size() - suffix.size()), suffix);
}

}  // namespace

bool is_structure_path(const std::filesystem::path& path) {
  const std::string filename = path.filename().string();
  return ends_with_ascii_case(filename, ".pdb") ||
         ends_with_ascii_case(filename, ".ent") ||
         ends_with_ascii_case(filename, ".cif") ||
         ends_with_ascii_case(filename, ".mmcif");
}

universal::Status list_structure_files(
    const std::filesystem::path& directory,
    std::vector<std::filesystem::path>& files) {
  return list_structure_files(directory, files, kDefaultDiagnostics);
}

universal::Status list_structure_files(
    const std::filesystem::path& directory,
    std::vector<std::filesystem::path>& files,
    const StructureDirectoryDiagnostics& diagnostics) {
  std::error_code ec;
  if (!std::filesystem::is_directory(directory, ec) || ec) {
    return universal::unavailable_status(diagnostics.not_readable);
  }
  files.clear();
  for (std::filesystem::directory_iterator it{directory, ec};
       !ec && it != std::filesystem::directory_iterator{}; it.increment(ec)) {
    std::error_code file_ec;
    if (!it->is_regular_file(file_ec) || file_ec) {
      continue;
    }
    if (is_structure_path(it->path())) {
      files.push_back(it->path());
    }
  }
  if (ec) {
    return universal::unavailable_status(diagnostics.read_failed);
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    return universal::invalid_argument_status(diagnostics.empty);
  }
  return universal::ok_status();
}

}  // namespace hikoboshi::io

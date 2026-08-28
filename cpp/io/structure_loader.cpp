// Structure loader entry point: file/extension dispatch and orchestration of
// raw parse -> normalize -> atom inference. Returns a LoadedStructure that
// owns the backing storage for a chartered StructureView.

#include <hikoboshi/io/structure_loader.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

LoadedStructure::LoadedStructure() : impl_(std::make_unique<Impl>()) {}
LoadedStructure::~LoadedStructure() = default;
LoadedStructure::LoadedStructure(LoadedStructure&&) noexcept = default;
LoadedStructure& LoadedStructure::operator=(LoadedStructure&&) noexcept =
    default;

universal::StructureView LoadedStructure::view() const noexcept {
  universal::StructureView view{};
  view.residue_count = impl_->normalized.size();
  view.coordinates = universal::Span<const float>{impl_->coordinates.data(),
                                                  impl_->coordinates.size()};
  view.atom_sources = universal::Span<const universal::AtomSource>{
      impl_->atom_sources.data(), impl_->atom_sources.size()};
  view.residue_codes = universal::Span<const char>{impl_->residue_codes.data(),
                                                   impl_->residue_codes.size()};
  view.residues = universal::Span<const universal::ResidueMetadataView>{
      impl_->residues.data(), impl_->residues.size()};
  view.chain_breaks = universal::Span<const universal::ChainBreakView>{
      impl_->chain_breaks.data(), impl_->chain_breaks.size()};
  view.input_id = std::string_view{impl_->input_id_storage};
  view.source_filename = std::string_view{impl_->source_filename_storage};
  return view;
}

std::size_t LoadedStructure::residue_count() const noexcept {
  return impl_->normalized.size();
}

std::string_view LoadedStructure::input_id() const noexcept {
  return std::string_view{impl_->input_id_storage};
}
std::string_view LoadedStructure::source_filename() const noexcept {
  return std::string_view{impl_->source_filename_storage};
}
std::string_view LoadedStructure::selected_model_id() const noexcept {
  return std::string_view{impl_->selected_model_id_storage};
}
std::int32_t LoadedStructure::selected_model_index() const noexcept {
  return impl_->selected_model_index_value;
}
std::string_view LoadedStructure::selected_chain_id() const noexcept {
  return std::string_view{impl_->selected_chain_id_storage};
}
std::string_view LoadedStructure::altloc_note() const noexcept {
  return std::string_view{impl_->altloc_note_storage};
}

universal::Span<float> LoadedStructure::mutable_coordinates() noexcept {
  return universal::Span<float>{impl_->coordinates.data(),
                                impl_->coordinates.size()};
}
universal::Span<universal::AtomSource>
LoadedStructure::mutable_atom_sources() noexcept {
  return universal::Span<universal::AtomSource>{impl_->atom_sources.data(),
                                                impl_->atom_sources.size()};
}

namespace {

bool ends_with_ignore_case(std::string_view text,
                           std::string_view suffix) noexcept {
  if (text.size() < suffix.size()) {
    return false;
  }
  const auto lower = [](char value) {
    return value >= 'A' && value <= 'Z'
               ? static_cast<char>(value + ('a' - 'A'))
               : value;
  };
  return std::equal(text.end() - static_cast<std::ptrdiff_t>(suffix.size()),
                    text.end(), suffix.begin(), suffix.end(),
                    [&](char lhs, char rhs) {
                      return lower(lhs) == lower(rhs);
                    });
}

bool content_looks_mmcif(std::string_view content) noexcept {
  // mmCIF documents always begin with `data_` or contain `_atom_site.`
  if (content.size() >= 5 && content.substr(0, 5) == "data_") {
    return true;
  }
  return content.find("_atom_site.") != std::string_view::npos;
}

bool content_looks_pdb(std::string_view content) noexcept {
  // PDB ATOM/HETATM/MODEL records are column-anchored; quick scan for the
  // record names at line starts.
  std::size_t pos = 0;
  while (pos < content.size()) {
    std::size_t end = content.find('\n', pos);
    std::string_view line =
        content.substr(pos, end == std::string_view::npos ? content.size() - pos
                                                         : end - pos);
    if (line.size() >= 6) {
      const std::string_view tag = line.substr(0, 6);
      if (tag == "ATOM  " || tag == "HETATM" || tag == "MODEL ") {
        return true;
      }
    }
    if (end == std::string_view::npos) {
      break;
    }
    pos = end + 1;
  }
  return false;
}

std::string read_file_contents(std::string_view path,
                               universal::Status& status) {
  std::ifstream stream{std::string(path), std::ios::binary};
  if (!stream) {
    status = universal::unavailable_status("structure file not readable");
    return {};
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  status = universal::ok_status();
  return buffer.str();
}

std::string filename_from_path(std::string_view path) {
  const std::size_t slash = path.find_last_of("/\\");
  if (slash == std::string_view::npos) {
    return std::string{path};
  }
  return std::string{path.substr(slash + 1)};
}

}  // namespace

StructureFormat detect_structure_format(std::string_view path,
                                        std::string_view content) noexcept {
  if (ends_with_ignore_case(path, ".cif") ||
      ends_with_ignore_case(path, ".mmcif")) {
    return StructureFormat::MmCif;
  }
  if (ends_with_ignore_case(path, ".pdb") ||
      ends_with_ignore_case(path, ".ent")) {
    return StructureFormat::Pdb;
  }
  if (content_looks_mmcif(content)) {
    return StructureFormat::MmCif;
  }
  if (content_looks_pdb(content)) {
    return StructureFormat::Pdb;
  }
  return StructureFormat::Unknown;
}

universal::Result<LoadedStructure> load_structure_from_file(
    std::string_view path,
    const StructureLoadOptions& options) {
  universal::Result<LoadedStructure> result{};
  result.value = LoadedStructure{};

  universal::Status read_status{};
  std::string content = read_file_contents(path, read_status);
  if (!read_status.ok()) {
    result.status = read_status;
    return result;
  }

  const StructureFormat format = detect_structure_format(path, content);
  if (format == StructureFormat::Unknown) {
    result.status = universal::invalid_argument_status(
        "unrecognized structure file format; "
        "extension and content cues did not match PDB or mmCIF");
    return result;
  }

  const std::string filename = filename_from_path(path);
  if (format == StructureFormat::Pdb) {
    return load_pdb_from_string(content, filename, options);
  }
  return load_mmcif_from_string(content, filename, options);
}

universal::Result<LoadedStructure> load_pdb_from_string(
    std::string_view content,
    std::string_view origin_label,
    const StructureLoadOptions& options) {
  universal::Result<LoadedStructure> result{};
  result.value = LoadedStructure{};
  result.value.impl().source_filename_storage = std::string{origin_label};
  result.value.impl().input_id_storage = std::string{origin_label};

  std::vector<detail::RawAtomRecord> raw;
  universal::Status parse_status = parse_pdb_records(content, raw);
  if (!parse_status.ok()) {
    result.status = parse_status;
    return result;
  }
  if (raw.empty()) {
    result.status = universal::invalid_argument_status(
        "no ATOM or HETATM records present in PDB input");
    return result;
  }

  universal::Status normalize_status = normalize_structure(
      universal::Span<const detail::RawAtomRecord>{raw.data(), raw.size()},
      options, result.value);
  if (!normalize_status.ok()) {
    result.status = normalize_status;
    return result;
  }

  if (options.infer_missing_atoms) {
    universal::Status inference_status = infer_missing_atoms(result.value);
    if (!inference_status.ok()) {
      result.status = inference_status;
      return result;
    }
  }

  result.status = universal::ok_status();
  return result;
}

universal::Result<LoadedStructure> load_mmcif_from_string(
    std::string_view content,
    std::string_view origin_label,
    const StructureLoadOptions& options) {
  universal::Result<LoadedStructure> result{};
  result.value = LoadedStructure{};
  result.value.impl().source_filename_storage = std::string{origin_label};
  result.value.impl().input_id_storage = std::string{origin_label};

  std::vector<detail::RawAtomRecord> raw;
  universal::Status parse_status = parse_mmcif_records(content, raw);
  if (!parse_status.ok()) {
    result.status = parse_status;
    return result;
  }
  if (raw.empty()) {
    result.status = universal::invalid_argument_status(
        "no _atom_site rows present in mmCIF input");
    return result;
  }

  universal::Status normalize_status = normalize_structure(
      universal::Span<const detail::RawAtomRecord>{raw.data(), raw.size()},
      options, result.value);
  if (!normalize_status.ok()) {
    result.status = normalize_status;
    return result;
  }

  if (options.infer_missing_atoms) {
    universal::Status inference_status = infer_missing_atoms(result.value);
    if (!inference_status.ok()) {
      result.status = inference_status;
      return result;
    }
  }

  result.status = universal::ok_status();
  return result;
}

}  // namespace hikoboshi::io

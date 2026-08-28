#ifndef HIKOBOSHI_IO_STRUCTURE_LOADER_HPP
#define HIKOBOSHI_IO_STRUCTURE_LOADER_HPP

#include <cstddef>
#include <cstdint>
#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

enum class StructureFormat : std::uint8_t {
  Unknown = 0,
  Pdb = 1,
  MmCif = 2,
};

enum class AltlocPolicy : std::uint8_t {
  PreferBlankThenA = 0,
};

struct StructureLoadOptions {
  std::optional<int> model_index{};
  std::string model_id{};
  std::string chain_id{};
  std::optional<std::size_t> chain_index{};
  bool infer_missing_atoms = true;
  bool include_modified_residues = true;
  AltlocPolicy altloc_policy = AltlocPolicy::PreferBlankThenA;
};

namespace detail {

// Per-atom record produced by raw parsers, before normalization.
struct RawAtomRecord {
  std::int64_t source_record_index = -1;
  std::int32_t serial = 0;
  std::string atom_name;
  std::string element;
  char altloc = ' ';
  std::string residue_name;
  std::string chain_id;
  std::int32_t residue_number = 0;
  char insertion_code = ' ';
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float occupancy = 1.0F;
  bool is_hetatm = false;
  std::int32_t model_index = 1;
  std::string model_id;
};

// Per-residue normalized atom slot.
struct NormalizedResidue {
  char residue_code = 'X';
  std::string original_residue_name;
  std::string chain_id;
  std::string model_id;
  std::int32_t model_index = 0;
  std::int32_t residue_number = 0;
  char insertion_code = ' ';
  std::string source_id;
  std::int64_t source_residue_index = -1;
  std::int64_t source_record_index = -1;
  std::array<std::array<float, universal::kCoordinateAxisCount>,
             universal::kCanonicalAtomCount>
      coords{};
  std::array<universal::AtomSource, universal::kCanonicalAtomCount>
      atom_sources{};
  std::array<bool, universal::kCanonicalAtomCount> altloc_blank_seen{};
  std::array<char, universal::kCanonicalAtomCount> altloc_chosen{
      ' ', ' ', ' ', ' ', ' '};
};

}  // namespace detail

class LoadedStructure {
 public:
  struct Impl {
    std::vector<float> coordinates;
    std::vector<universal::AtomSource> atom_sources;
    std::vector<char> residue_codes;
    std::vector<universal::ResidueMetadataView> residues;
    std::vector<universal::ChainBreakView> chain_breaks;
    std::vector<detail::NormalizedResidue> normalized;
    std::deque<std::string> string_pool;
    std::string source_filename_storage;
    std::string input_id_storage;
    std::string selected_model_id_storage;
    std::string selected_chain_id_storage;
    std::string altloc_note_storage;
    std::int32_t selected_model_index_value = 0;

    std::string_view intern(std::string value) {
      string_pool.push_back(std::move(value));
      return std::string_view{string_pool.back()};
    }
  };

  LoadedStructure();
  ~LoadedStructure();
  LoadedStructure(const LoadedStructure&) = delete;
  LoadedStructure& operator=(const LoadedStructure&) = delete;
  LoadedStructure(LoadedStructure&&) noexcept;
  LoadedStructure& operator=(LoadedStructure&&) noexcept;

  universal::StructureView view() const noexcept;

  std::size_t residue_count() const noexcept;

  std::string_view input_id() const noexcept;
  std::string_view source_filename() const noexcept;
  std::string_view selected_model_id() const noexcept;
  std::int32_t selected_model_index() const noexcept;
  std::string_view selected_chain_id() const noexcept;
  std::string_view altloc_note() const noexcept;

  universal::Span<float> mutable_coordinates() noexcept;
  universal::Span<universal::AtomSource> mutable_atom_sources() noexcept;

  Impl& impl() noexcept { return *impl_; }
  const Impl& impl() const noexcept { return *impl_; }

 private:
  std::unique_ptr<Impl> impl_;
};

StructureFormat detect_structure_format(std::string_view path,
                                        std::string_view content) noexcept;

[[nodiscard]] universal::Result<LoadedStructure> load_structure_from_file(
    std::string_view path,
    const StructureLoadOptions& options = {});

[[nodiscard]] universal::Result<LoadedStructure> load_pdb_from_string(
    std::string_view content,
    std::string_view origin_label,
    const StructureLoadOptions& options = {});

[[nodiscard]] universal::Result<LoadedStructure> load_mmcif_from_string(
    std::string_view content,
    std::string_view origin_label,
    const StructureLoadOptions& options = {});

// Parser entry points used internally and by tests.
[[nodiscard]] universal::Status parse_pdb_records(
    std::string_view content,
    std::vector<detail::RawAtomRecord>& out);
[[nodiscard]] universal::Status parse_mmcif_records(
    std::string_view content,
    std::vector<detail::RawAtomRecord>& out);

// Normalize raw atom records into per-residue slots applying the chartered
// model/chain selection, altloc policy, modified-residue whitelist, HETATM
// filtering, and residue ordering. Sets selected_model/chain on the result.
[[nodiscard]] universal::Status normalize_structure(
    universal::Span<const detail::RawAtomRecord> raw_records,
    const StructureLoadOptions& options,
    LoadedStructure& out);

// Apply atom inference using the rigid template fit. Marks inferred atoms
// AtomSource::Inferred and glycine virtual CB AtomSource::Virtual. Refuses to
// fill atoms when the chartered guard fails.
[[nodiscard]] universal::Status infer_missing_atoms(LoadedStructure& structure);

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_STRUCTURE_LOADER_HPP

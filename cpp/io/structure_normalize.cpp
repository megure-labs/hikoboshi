// Chartered structure normalization. Selects model/chain, applies the altloc
// policy, filters HETATM down to whitelisted modified residues, maps residue
// names to canonical one-letter codes, projects atom records into the
// N/CA/C/O/CB canonical slots, and produces ChainBreakView entries on
// residue-number gaps.

#include <hikoboshi/io/structure_loader.hpp>

#include <hikoboshi/io/residue_table.hpp>
#include <hikoboshi/io/structure_atoms.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

namespace {

int canonical_atom_index(std::string_view name) noexcept {
  if (name == "N") return 0;
  if (name == "CA") return 1;
  if (name == "C") return 2;
  if (name == "O") return 3;
  if (name == "CB") return 4;
  // mmCIF MSE uses SE for selenium-CB; the modified-residue mapping treats
  // MSE as MET, so SE acts as the side-chain CB analog and we ignore it here.
  return -1;
}

bool altloc_better(char proposed, char current,
                   bool current_blank_seen) noexcept {
  // Prefer blank, then 'A', then first-seen.
  const auto rank = [](char c) {
    if (c == ' ' || c == '\0') return 0;
    if (c == 'A' || c == 'a') return 1;
    return 2;
  };
  if (current_blank_seen) {
    return false;
  }
  if (proposed == ' ' || proposed == '\0') {
    return true;
  }
  return rank(proposed) < rank(current);
}

struct ResidueKey {
  std::string chain_id;
  std::int32_t residue_number;
  char insertion_code;
};

struct ResidueKeyHash {
  std::size_t operator()(const ResidueKey& k) const noexcept {
    std::size_t h = std::hash<std::string>{}(k.chain_id);
    h ^= std::hash<std::int32_t>{}(k.residue_number) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<char>{}(k.insertion_code) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    return h;
  }
};

struct ResidueKeyEq {
  bool operator()(const ResidueKey& a, const ResidueKey& b) const noexcept {
    return a.chain_id == b.chain_id && a.residue_number == b.residue_number &&
           a.insertion_code == b.insertion_code;
  }
};

bool accepted_polymer_record(const detail::RawAtomRecord& rec,
                             const StructureLoadOptions& options) noexcept {
  if (canonical_atom_index(rec.atom_name) < 0) {
    return false;
  }
  if (!rec.is_hetatm) {
    return true;
  }
  const ResidueIdentity id = classify_residue_name(rec.residue_name);
  return options.include_modified_residues &&
         is_modified_amino_acid(rec.residue_name) &&
         id.one_letter != kUnknownResidueCode;
}

}  // namespace

universal::Status normalize_structure(
    universal::Span<const detail::RawAtomRecord> raw_records,
    const StructureLoadOptions& options,
    LoadedStructure& out) {
  auto& impl = out.impl();
  impl.normalized.clear();
  impl.coordinates.clear();
  impl.atom_sources.clear();
  impl.residue_codes.clear();
  impl.residues.clear();
  impl.chain_breaks.clear();
  impl.string_pool.clear();
  impl.altloc_note_storage.clear();

  // Pass 1: choose the target model.
  std::int32_t selected_model_index = 0;
  std::string selected_model_id;
  bool model_set = false;
  if (options.model_index.has_value()) {
    selected_model_index = *options.model_index;
    model_set = true;
  } else if (!options.model_id.empty()) {
    for (std::size_t i = 0; i < raw_records.size; ++i) {
      const auto& rec = raw_records.data[i];
      if (rec.model_id == options.model_id) {
        selected_model_index = rec.model_index;
        model_set = true;
        break;
      }
    }
    if (!model_set) {
      return universal::invalid_argument_status(
          "requested model id not present in input");
    }
  } else {
    selected_model_index = 1;
    model_set = true;
  }
  if (!model_set) {
    return universal::invalid_argument_status(
        "structure has no model records");
  }

  // Discover the model id string for metadata.
  bool selected_model_seen = false;
  for (std::size_t i = 0; i < raw_records.size; ++i) {
    const auto& rec = raw_records.data[i];
    if (rec.model_index == selected_model_index) {
      selected_model_id = rec.model_id;
      selected_model_seen = true;
      break;
    }
  }
  if (!selected_model_seen) {
    return universal::invalid_argument_status(
        "selected model index not present in input");
  }
  impl.selected_model_index_value = selected_model_index;
  impl.selected_model_id_storage = selected_model_id;

  // Pass 2: choose the target polymer chain. Chain defaults must ignore
  // waters, ligands, and other stripped HETATM-only chains.
  std::vector<std::string> chains_in_order;
  std::unordered_set<std::string> chains_seen;
  for (std::size_t i = 0; i < raw_records.size; ++i) {
    const auto& rec = raw_records.data[i];
    if (rec.model_index != selected_model_index) {
      continue;
    }
    if (!accepted_polymer_record(rec, options)) {
      continue;
    }
    if (chains_seen.insert(rec.chain_id).second) {
      chains_in_order.push_back(rec.chain_id);
    }
  }

  std::string selected_chain;
  if (!options.chain_id.empty()) {
    auto it = std::find(chains_in_order.begin(), chains_in_order.end(),
                        options.chain_id);
    if (it == chains_in_order.end()) {
      return universal::invalid_argument_status(
          "requested chain id not present in selected model");
    }
    selected_chain = *it;
  } else if (options.chain_index.has_value()) {
    if (*options.chain_index >= chains_in_order.size()) {
      return universal::invalid_argument_status(
          "requested chain index out of range");
    }
    selected_chain = chains_in_order[*options.chain_index];
  } else {
    auto a_it = std::find(chains_in_order.begin(), chains_in_order.end(),
                          std::string{"A"});
    if (a_it != chains_in_order.end()) {
      selected_chain = *a_it;
    } else if (!chains_in_order.empty()) {
      selected_chain = chains_in_order.front();
    } else {
      return universal::invalid_argument_status(
          "selected model has no polymer chains");
    }
  }
  impl.selected_chain_id_storage = selected_chain;

  // Pass 3: collect rows for the selected model+chain in record order.
  std::vector<const detail::RawAtomRecord*> kept;
  kept.reserve(raw_records.size);
  for (std::size_t i = 0; i < raw_records.size; ++i) {
    const auto& rec = raw_records.data[i];
    if (rec.model_index != selected_model_index) continue;
    if (rec.chain_id != selected_chain) continue;
    if (!accepted_polymer_record(rec, options)) {
      // Strip waters, ligands, and non-canonical atoms before aggregation.
      continue;
    }
    kept.push_back(&rec);
  }

  // Pass 4: aggregate per-residue with altloc policy.
  std::unordered_map<ResidueKey, std::size_t, ResidueKeyHash, ResidueKeyEq>
      residue_index;
  std::vector<detail::NormalizedResidue> residues;
  residues.reserve(kept.size() / 5 + 1);
  std::vector<std::int64_t> first_record_seen;
  bool any_altloc_decision = false;

  for (const detail::RawAtomRecord* rec_ptr : kept) {
    const detail::RawAtomRecord& rec = *rec_ptr;
    const int atom_idx = canonical_atom_index(rec.atom_name);
    if (atom_idx < 0) {
      // Non-canonical atoms (e.g., side-chain CG, OXT) are stripped because
      // Hikoboshi 0.1.0 normalizes to N/CA/C/O/CB only.
      continue;
    }

    const ResidueKey key{rec.chain_id, rec.residue_number, rec.insertion_code};
    auto it = residue_index.find(key);
    if (it == residue_index.end()) {
      detail::NormalizedResidue slot{};
      const ResidueIdentity id = classify_residue_name(rec.residue_name);
      slot.residue_code = id.one_letter;
      slot.original_residue_name = rec.residue_name;
      slot.chain_id = rec.chain_id;
      slot.model_id = selected_model_id;
      slot.model_index = selected_model_index;
      slot.residue_number = rec.residue_number;
      slot.insertion_code = rec.insertion_code;
      slot.source_id = rec.chain_id + "_" + std::to_string(rec.residue_number);
      slot.source_residue_index = static_cast<std::int64_t>(residues.size());
      slot.source_record_index = rec.source_record_index;
      residues.push_back(std::move(slot));
      it = residue_index.emplace(key, residues.size() - 1).first;
      first_record_seen.push_back(rec.source_record_index);
    }
    detail::NormalizedResidue& slot = residues[it->second];

    const char altloc = rec.altloc;
    if (slot.atom_sources[atom_idx] == universal::AtomSource::Observed) {
      if (!altloc_better(altloc, slot.altloc_chosen[atom_idx],
                         slot.altloc_blank_seen[atom_idx])) {
        continue;
      }
      any_altloc_decision = true;
    }
    slot.coords[atom_idx][0] = rec.x;
    slot.coords[atom_idx][1] = rec.y;
    slot.coords[atom_idx][2] = rec.z;
    slot.atom_sources[atom_idx] = universal::AtomSource::Observed;
    if (altloc == ' ' || altloc == '\0') {
      slot.altloc_blank_seen[atom_idx] = true;
    }
    slot.altloc_chosen[atom_idx] = altloc;
  }

  // Sort residues by file order: first appearance in `kept`. Use
  // `first_record_seen` directly indexed by residue order (which matches the
  // order in `residues`).
  std::vector<std::size_t> order(residues.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) {
              return first_record_seen[a] < first_record_seen[b];
            });

  std::vector<detail::NormalizedResidue> sorted_residues;
  sorted_residues.reserve(residues.size());
  for (std::size_t idx : order) {
    sorted_residues.push_back(std::move(residues[idx]));
  }

  if (sorted_residues.empty()) {
    return universal::invalid_argument_status(
        "selected chain has no accepted amino-acid residues");
  }

  // Surface chain-break metadata for residue-number gaps.
  std::vector<universal::ChainBreakView> chain_breaks;
  for (std::size_t i = 1; i < sorted_residues.size(); ++i) {
    const auto& prev = sorted_residues[i - 1];
    const auto& curr = sorted_residues[i];
    if (curr.chain_id == prev.chain_id) {
      const std::int32_t expected_next = prev.residue_number + 1;
      const bool number_gap =
          curr.residue_number > expected_next ||
          (curr.residue_number == prev.residue_number &&
           curr.insertion_code != prev.insertion_code &&
           prev.insertion_code != ' ' && curr.insertion_code == ' ');
      if (number_gap) {
        chain_breaks.push_back(
            universal::ChainBreakView{static_cast<std::size_t>(i - 1)});
      }
    }
  }

  if (any_altloc_decision) {
    impl.altloc_note_storage = "altloc selection applied";
  }

  // Allocate flattened buffers.
  const std::size_t L = sorted_residues.size();
  impl.coordinates.assign(L * 5 * 3, 0.0F);
  impl.atom_sources.assign(L * 5, universal::AtomSource::Missing);
  impl.residue_codes.assign(L, 'X');
  impl.residues.clear();
  impl.residues.reserve(L);

  const universal::Status pack_status = canonical_atom_pack(
      CanonicalAtomPackRequest{
          universal::Span<const detail::NormalizedResidue>{
              sorted_residues.data(), sorted_residues.size()}},
      CanonicalAtomPackOutput{
          universal::Span<float>{impl.coordinates.data(),
                                 impl.coordinates.size()},
          universal::Span<universal::AtomSource>{impl.atom_sources.data(),
                                                 impl.atom_sources.size()}});
  if (!pack_status.ok()) {
    return pack_status;
  }

  for (std::size_t i = 0; i < L; ++i) {
    const auto& slot = sorted_residues[i];
    impl.residue_codes[i] = slot.residue_code;

    universal::ResidueMetadataView md{};
    md.residue_code = slot.residue_code;
    md.original_residue_name =
        impl.intern(std::string{slot.original_residue_name});
    md.chain_id = impl.intern(std::string{slot.chain_id});
    md.model_id = impl.intern(std::string{slot.model_id});
    md.model_index = slot.model_index;
    md.residue_number = slot.residue_number;
    md.insertion_code = slot.insertion_code;
    md.source_id = impl.intern(std::string{slot.source_id});
    md.source_residue_index = static_cast<std::int64_t>(i);
    md.source_filename = std::string_view{impl.source_filename_storage};
    md.source_record_index = slot.source_record_index;
    impl.residues.push_back(md);
  }

  impl.chain_breaks = std::move(chain_breaks);
  impl.normalized = std::move(sorted_residues);

  return universal::ok_status();
}

}  // namespace hikoboshi::io

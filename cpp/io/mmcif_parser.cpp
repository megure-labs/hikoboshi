// Minimal mmCIF/CIF _atom_site loop parser. Supports the common single-block
// loop_ over _atom_site.* columns produced by wwPDB mmCIF distributions.
// Refuses to silently zero-fill on unparseable coordinate fields and surfaces
// an InvalidArgument status when required columns are absent.

#include <hikoboshi/io/structure_loader.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::io {

namespace {

bool starts_with(std::string_view text, std::string_view prefix) noexcept {
  return text.size() >= prefix.size() &&
         text.substr(0, prefix.size()) == prefix;
}

std::string_view trim(std::string_view text) noexcept {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(begin, end - begin);
}

bool parse_int_token(std::string_view text, std::int32_t& out) noexcept {
  std::string_view s = trim(text);
  if (s.empty()) {
    return false;
  }
  if (s.front() == '+') {
    s.remove_prefix(1);
  }
  if (s.empty()) {
    return false;
  }
  std::int64_t value = 0;
  const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != s.data() + s.size()) {
    return false;
  }
  out = static_cast<std::int32_t>(value);
  return true;
}

bool parse_float_token(std::string_view text, float& out) noexcept {
  std::string_view s = trim(text);
  if (s.empty() || s == "?" || s == ".") {
    return false;
  }
  if (s.front() == '+') {
    s.remove_prefix(1);
  }
  if (s.empty()) {
    return false;
  }
  double value = 0.0;
  const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != s.data() + s.size()) {
    return false;
  }
  if (!std::isfinite(value)) {
    return false;
  }
  out = static_cast<float>(value);
  return true;
}

// Tokenize one mmCIF data line: handles double-quoted, single-quoted, and
// whitespace-separated tokens. Does not support semi-colon multi-line text
// fields (rare for _atom_site rows).
std::vector<std::string> tokenize_line(std::string_view line) {
  std::vector<std::string> tokens;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() &&
           std::isspace(static_cast<unsigned char>(line[i]))) {
      ++i;
    }
    if (i >= line.size()) {
      break;
    }
    const char ch = line[i];
    if (ch == '"' || ch == '\'') {
      const char quote = ch;
      ++i;
      const std::size_t start = i;
      while (i < line.size() && line[i] != quote) {
        ++i;
      }
      tokens.emplace_back(line.substr(start, i - start));
      if (i < line.size()) {
        ++i;
      }
    } else {
      const std::size_t start = i;
      while (i < line.size() &&
             !std::isspace(static_cast<unsigned char>(line[i]))) {
        ++i;
      }
      tokens.emplace_back(line.substr(start, i - start));
    }
  }
  return tokens;
}

}  // namespace

universal::Status parse_mmcif_records(std::string_view content,
                                      std::vector<detail::RawAtomRecord>& out) {
  out.clear();

  std::size_t pos = 0;
  bool found_atom_site_loop = false;
  std::int64_t record_index = 0;

  while (pos <= content.size()) {
    std::size_t line_end = content.find('\n', pos);
    std::string_view line = content.substr(
        pos, line_end == std::string_view::npos ? content.size() - pos
                                                : line_end - pos);
    line = trim(line);

    if (starts_with(line, "loop_")) {
      // Scan loop columns.
      std::size_t scan = line_end == std::string_view::npos ? content.size()
                                                            : line_end + 1;
      std::vector<std::string> columns;
      bool is_atom_site = false;
      while (scan <= content.size()) {
        std::size_t next_end = content.find('\n', scan);
        std::string_view next = content.substr(
            scan, next_end == std::string_view::npos ? content.size() - scan
                                                     : next_end - scan);
        std::string_view trimmed = trim(next);
        if (starts_with(trimmed, "_atom_site.")) {
          is_atom_site = true;
          columns.emplace_back(std::string{trimmed.substr(11)});
        } else if (starts_with(trimmed, "_")) {
          // Different category; abandon this loop.
          columns.clear();
          break;
        } else if (trimmed.empty()) {
          // Blank line within column declarations is benign.
        } else {
          // First data row of the loop.
          break;
        }
        scan = next_end == std::string_view::npos ? content.size()
                                                  : next_end + 1;
      }

      if (!is_atom_site || columns.empty()) {
        if (line_end == std::string_view::npos) {
          break;
        }
        pos = line_end + 1;
        continue;
      }
      found_atom_site_loop = true;

      // Build column-name lookup.
      std::unordered_map<std::string, std::size_t> col_index;
      col_index.reserve(columns.size());
      for (std::size_t i = 0; i < columns.size(); ++i) {
        col_index[columns[i]] = i;
      }

      auto require = [&](const char* name) -> std::size_t {
        auto it = col_index.find(name);
        return it == col_index.end() ? static_cast<std::size_t>(-1)
                                     : it->second;
      };

      const std::size_t k_group = require("group_PDB");
      const std::size_t k_atom_name = require("label_atom_id");
      const std::size_t k_atom_alt = require("auth_atom_id");
      const std::size_t k_alt = require("label_alt_id");
      const std::size_t k_resname = require("auth_comp_id");
      const std::size_t k_resname_alt = require("label_comp_id");
      const std::size_t k_chain = require("auth_asym_id");
      const std::size_t k_chain_alt = require("label_asym_id");
      const std::size_t k_resnum = require("auth_seq_id");
      const std::size_t k_resnum_alt = require("label_seq_id");
      const std::size_t k_icode = require("pdbx_PDB_ins_code");
      const std::size_t k_x = require("Cartn_x");
      const std::size_t k_y = require("Cartn_y");
      const std::size_t k_z = require("Cartn_z");
      const std::size_t k_occ = require("occupancy");
      const std::size_t k_element = require("type_symbol");
      const std::size_t k_model = require("pdbx_PDB_model_num");
      const std::size_t k_serial = require("id");

      if (k_atom_name == static_cast<std::size_t>(-1) &&
          k_atom_alt == static_cast<std::size_t>(-1)) {
        return universal::invalid_argument_status(
            "mmCIF _atom_site loop missing required atom-name column");
      }
      if (k_x == static_cast<std::size_t>(-1) ||
          k_y == static_cast<std::size_t>(-1) ||
          k_z == static_cast<std::size_t>(-1)) {
        return universal::invalid_argument_status(
            "mmCIF _atom_site loop missing required Cartesian coordinate "
            "columns");
      }

      pos = scan;
      while (pos <= content.size()) {
        std::size_t row_end = content.find('\n', pos);
        std::string_view row = content.substr(
            pos, row_end == std::string_view::npos ? content.size() - pos
                                                   : row_end - pos);
        std::string_view trimmed_row = trim(row);

        if (trimmed_row.empty()) {
          if (row_end == std::string_view::npos) {
            break;
          }
          pos = row_end + 1;
          continue;
        }

        if (starts_with(trimmed_row, "loop_") ||
            starts_with(trimmed_row, "data_") ||
            starts_with(trimmed_row, "_") ||
            starts_with(trimmed_row, "#")) {
          // End of this _atom_site data block.
          break;
        }

        std::vector<std::string> tokens = tokenize_line(trimmed_row);
        if (tokens.size() < columns.size()) {
          // Some mmCIF files wrap rows; collect tokens across continuation
          // lines until the row is complete.
          std::size_t cont = pos;
          while (tokens.size() < columns.size()) {
            cont = row_end == std::string_view::npos ? content.size()
                                                     : row_end + 1;
            if (cont >= content.size()) {
              break;
            }
            row_end = content.find('\n', cont);
            std::string_view extra = content.substr(
                cont, row_end == std::string_view::npos ? content.size() - cont
                                                        : row_end - cont);
            std::vector<std::string> more = tokenize_line(trim(extra));
            tokens.insert(tokens.end(), more.begin(), more.end());
            pos = row_end == std::string_view::npos ? content.size()
                                                    : row_end;
          }
        }

        if (tokens.size() < columns.size()) {
          return universal::invalid_argument_status(
              "mmCIF _atom_site row column count mismatch");
        }

        detail::RawAtomRecord rec{};
        rec.source_record_index = record_index;

        const std::string& group =
            k_group == static_cast<std::size_t>(-1) ? tokens[0] : tokens[k_group];
        rec.is_hetatm = (group == "HETATM");

        if (k_serial != static_cast<std::size_t>(-1)) {
          std::int32_t serial = 0;
          parse_int_token(tokens[k_serial], serial);
          rec.serial = serial;
        }

        const std::size_t name_col = k_atom_name != static_cast<std::size_t>(-1)
                                         ? k_atom_name
                                         : k_atom_alt;
        std::string atom_name = tokens[name_col];
        // mmCIF often quotes names like "CA"
        rec.atom_name = atom_name;

        if (k_alt != static_cast<std::size_t>(-1)) {
          const std::string& alt = tokens[k_alt];
          rec.altloc = (alt == "." || alt == "?") ? ' ' : alt.front();
        } else {
          rec.altloc = ' ';
        }

        const std::size_t res_col = k_resname != static_cast<std::size_t>(-1)
                                        ? k_resname
                                        : k_resname_alt;
        if (res_col != static_cast<std::size_t>(-1)) {
          rec.residue_name = tokens[res_col];
        }

        const std::size_t chain_col = k_chain != static_cast<std::size_t>(-1)
                                          ? k_chain
                                          : k_chain_alt;
        if (chain_col != static_cast<std::size_t>(-1)) {
          const std::string& chain = tokens[chain_col];
          rec.chain_id = (chain == "." || chain == "?") ? std::string{}
                                                        : chain;
        }

        const std::size_t resnum_col = k_resnum != static_cast<std::size_t>(-1)
                                           ? k_resnum
                                           : k_resnum_alt;
        if (resnum_col != static_cast<std::size_t>(-1)) {
          std::int32_t rn = 0;
          if (!parse_int_token(tokens[resnum_col], rn)) {
            return universal::invalid_argument_status(
                "mmCIF _atom_site row has unparseable residue number");
          }
          rec.residue_number = rn;
        }

        if (k_icode != static_cast<std::size_t>(-1)) {
          const std::string& icode = tokens[k_icode];
          rec.insertion_code =
              (icode == "?" || icode == "." || icode.empty())
                  ? ' '
                  : icode.front();
        }

        float x = 0.0F;
        float y = 0.0F;
        float z = 0.0F;
        if (!parse_float_token(tokens[k_x], x) ||
            !parse_float_token(tokens[k_y], y) ||
            !parse_float_token(tokens[k_z], z)) {
          return universal::invalid_argument_status(
              "mmCIF _atom_site row has unparseable coordinate; refusing to "
              "silently zero-fill");
        }
        rec.x = x;
        rec.y = y;
        rec.z = z;

        if (k_occ != static_cast<std::size_t>(-1)) {
          float occ = 1.0F;
          parse_float_token(tokens[k_occ], occ);
          rec.occupancy = occ;
        }

        if (k_element != static_cast<std::size_t>(-1)) {
          rec.element = tokens[k_element];
        }

        if (k_model != static_cast<std::size_t>(-1)) {
          std::int32_t model_index = 1;
          if (parse_int_token(tokens[k_model], model_index)) {
            rec.model_index = model_index;
            rec.model_id = tokens[k_model];
          } else {
            rec.model_index = 1;
            rec.model_id = "1";
          }
        } else {
          rec.model_index = 1;
          rec.model_id = "1";
        }

        out.push_back(std::move(rec));
        ++record_index;

        if (row_end == std::string_view::npos) {
          pos = content.size();
          break;
        }
        pos = row_end + 1;
      }

      continue;
    }

    if (line_end == std::string_view::npos) {
      break;
    }
    pos = line_end + 1;
  }

  if (!found_atom_site_loop) {
    return universal::invalid_argument_status(
        "mmCIF input does not contain an _atom_site loop");
  }

  return universal::ok_status();
}

}  // namespace hikoboshi::io

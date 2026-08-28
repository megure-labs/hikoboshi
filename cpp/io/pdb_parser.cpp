// Column-oriented PDB ATOM/HETATM parser. Produces RawAtomRecords for the
// downstream normalize step. Refuses to silently fix malformed coordinate
// fields and surfaces an InvalidArgument status for unparseable records.

#include <hikoboshi/io/structure_loader.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
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
  while (begin < end && (text[begin] == ' ' || text[begin] == '\t')) {
    ++begin;
  }
  while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t' ||
                         text[end - 1] == '\r')) {
    --end;
  }
  return text.substr(begin, end - begin);
}

bool parse_int_field(std::string_view text, std::int32_t& out) noexcept {
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

bool parse_float_field(std::string_view text, float& out) noexcept {
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

std::string_view substr_safe(std::string_view text, std::size_t pos,
                             std::size_t count) noexcept {
  if (pos >= text.size()) {
    return std::string_view{};
  }
  return text.substr(pos, std::min(count, text.size() - pos));
}

}  // namespace

universal::Status parse_pdb_records(std::string_view content,
                                    std::vector<detail::RawAtomRecord>& out) {
  out.clear();
  std::int32_t current_model_index = 1;
  std::string current_model_id = "1";
  bool in_model = false;
  std::int64_t record_index = 0;

  std::size_t pos = 0;
  while (pos <= content.size()) {
    std::size_t end = content.find('\n', pos);
    std::string_view line = content.substr(
        pos,
        end == std::string_view::npos ? content.size() - pos : end - pos);

    if (starts_with(line, "MODEL ")) {
      std::int32_t parsed_index = 0;
      const std::string_view serial = substr_safe(line, 10, 4);
      if (parse_int_field(serial, parsed_index)) {
        current_model_index = parsed_index;
        current_model_id = std::to_string(parsed_index);
      } else {
        return universal::invalid_argument_status(
            "PDB MODEL record missing or unparseable serial");
      }
      in_model = true;
    } else if (starts_with(line, "ENDMDL")) {
      in_model = false;
    } else if (starts_with(line, "ATOM  ") || starts_with(line, "HETATM")) {
      if (line.size() < 54) {
        return universal::invalid_argument_status(
            "PDB ATOM/HETATM record shorter than the 54-column coordinate "
            "block");
      }

      detail::RawAtomRecord rec{};
      rec.is_hetatm = starts_with(line, "HETATM");
      rec.source_record_index = record_index;
      rec.model_index = in_model ? current_model_index : 1;
      rec.model_id = in_model ? current_model_id : std::string{"1"};

      std::int32_t serial = 0;
      if (!parse_int_field(substr_safe(line, 6, 5), serial)) {
        return universal::invalid_argument_status(
            "PDB ATOM/HETATM record has invalid serial column");
      }
      rec.serial = serial;

      rec.atom_name = std::string{trim(substr_safe(line, 12, 4))};
      rec.altloc = line[16];
      rec.residue_name = std::string{trim(substr_safe(line, 17, 3))};

      const std::string_view chain_field = substr_safe(line, 21, 1);
      rec.chain_id = chain_field.empty() ? std::string{}
                                         : std::string{trim(chain_field)};

      std::int32_t resnum = 0;
      if (!parse_int_field(substr_safe(line, 22, 4), resnum)) {
        return universal::invalid_argument_status(
            "PDB ATOM/HETATM record has invalid residue number");
      }
      rec.residue_number = resnum;

      const std::string_view icode_field = substr_safe(line, 26, 1);
      rec.insertion_code =
          icode_field.empty() ? ' ' : icode_field.front();

      float x = 0.0F;
      float y = 0.0F;
      float z = 0.0F;
      if (!parse_float_field(substr_safe(line, 30, 8), x) ||
          !parse_float_field(substr_safe(line, 38, 8), y) ||
          !parse_float_field(substr_safe(line, 46, 8), z)) {
        return universal::invalid_argument_status(
            "PDB ATOM/HETATM record has invalid coordinate field; refusing to "
            "silently zero-fill");
      }
      rec.x = x;
      rec.y = y;
      rec.z = z;

      if (line.size() >= 60) {
        float occupancy = 1.0F;
        if (parse_float_field(substr_safe(line, 54, 6), occupancy)) {
          rec.occupancy = occupancy;
        }
      }

      if (line.size() >= 78) {
        rec.element = std::string{trim(substr_safe(line, 76, 2))};
      }

      out.push_back(std::move(rec));
      ++record_index;
    }

    if (end == std::string_view::npos) {
      break;
    }
    pos = end + 1;
  }

  return universal::ok_status();
}

}  // namespace hikoboshi::io

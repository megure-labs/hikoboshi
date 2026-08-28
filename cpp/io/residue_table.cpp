// Residue identity table and frozen N/CA/C/O/CB residue template.
//
// Source citation: the backbone bond-length literals and N/CA/C angle follow
// Engh & Huber (Acta Cryst. (1991) A47, 392-400) as represented in the IUCr
// mmCIF Engh-Huber/Priestle examples. The CA/C/O angle literal remains the
// current i03-selected 120.5 deg polyalanine ideal-geometry value from Pappu,
// Srinivasan & Rose (PNAS 2000, 97:12565-12570, Table 3). The CB position is
// derived from the chartered glycine virtual-CB constants applied to the
// current N/CA/C frame. The same N/CA/C/O/CB template is used for all 20
// canonical amino acids because Hikoboshi 0.1.0 normalizes inputs to exactly
// these five canonical atoms. The modified-residue whitelist is taken verbatim
// from STRUCTURE_INPUT_CHARTER.md and is the only mapping accepted at the
// parser surface. tests/cpp/residue_template_frozen_tests.cpp freezes the
// emitted constants and frame; do not change values without a correction packet.

#include <hikoboshi/io/residue_table.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

namespace {

struct CanonicalEntry {
  char one_letter;
  std::string_view three_letter;
};

constexpr std::array<CanonicalEntry, 20> kCanonicalEntries = {{
    {'A', "ALA"}, {'R', "ARG"}, {'N', "ASN"}, {'D', "ASP"}, {'C', "CYS"},
    {'Q', "GLN"}, {'E', "GLU"}, {'G', "GLY"}, {'H', "HIS"}, {'I', "ILE"},
    {'L', "LEU"}, {'K', "LYS"}, {'M', "MET"}, {'F', "PHE"}, {'P', "PRO"},
    {'S', "SER"}, {'T', "THR"}, {'W', "TRP"}, {'Y', "TYR"}, {'V', "VAL"},
}};

struct ModifiedEntry {
  std::string_view three_letter;
  char canonical_one_letter;
  std::string_view canonical_three_letter;
};

constexpr std::array<ModifiedEntry, 7> kModifiedWhitelist = {{
    {"MSE", 'M', "MET"},
    {"SEC", 'C', "CYS"},
    {"PYL", 'K', "LYS"},
    {"SEP", 'S', "SER"},
    {"TPO", 'T', "THR"},
    {"PTR", 'Y', "TYR"},
    {"HYP", 'P', "PRO"},
}};

constexpr bool equal_ignore_case_3(std::string_view a,
                                   std::string_view b) noexcept {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    char ca = a[i];
    char cb = b[i];
    if (ca >= 'a' && ca <= 'z') {
      ca = static_cast<char>(ca - ('a' - 'A'));
    }
    if (cb >= 'a' && cb <= 'z') {
      cb = static_cast<char>(cb - ('a' - 'A'));
    }
    if (ca != cb) {
      return false;
    }
  }
  return true;
}

std::string_view trim_residue_name(std::string_view name) noexcept {
  std::size_t begin = 0;
  std::size_t end = name.size();
  while (begin < end && (name[begin] == ' ' || name[begin] == '\t')) {
    ++begin;
  }
  while (end > begin && (name[end - 1] == ' ' || name[end - 1] == '\t')) {
    --end;
  }
  return name.substr(begin, end - begin);
}

const ResidueTemplate& build_template() noexcept {
  static const ResidueTemplate tmpl = []() {
    ResidueTemplate t{};

    constexpr float kNCA = 1.458F;
    constexpr float kCAC = 1.525F;
    constexpr float kCO = 1.231F;
    constexpr float kAngleNCAC = 1.9416F;   // 111.2 deg in radians
    constexpr float kAngleCACO = 2.1031F;   // 120.5 deg in radians

    const float ca[3] = {0.0F, 0.0F, 0.0F};

    const float n_pos[3] = {-kNCA, 0.0F, 0.0F};

    const float c_pos[3] = {
        kCAC * std::cos(kAngleNCAC - 1.5707963267948966F),
        kCAC * std::sin(kAngleNCAC - 1.5707963267948966F),
        0.0F,
    };

    const float ca_to_c[3] = {c_pos[0] - ca[0], c_pos[1] - ca[1],
                              c_pos[2] - ca[2]};
    const float ca_to_c_len = std::sqrt(ca_to_c[0] * ca_to_c[0] +
                                        ca_to_c[1] * ca_to_c[1] +
                                        ca_to_c[2] * ca_to_c[2]);
    const float u[3] = {ca_to_c[0] / ca_to_c_len, ca_to_c[1] / ca_to_c_len,
                        ca_to_c[2] / ca_to_c_len};
    const float v[3] = {-u[1], u[0], 0.0F};
    const float cos_co = std::cos(3.14159265358979F - kAngleCACO);
    const float sin_co = std::sin(3.14159265358979F - kAngleCACO);
    const float o_dir[3] = {cos_co * u[0] + sin_co * v[0],
                            cos_co * u[1] + sin_co * v[1],
                            cos_co * u[2] + sin_co * v[2]};
    const float o_pos[3] = {c_pos[0] + kCO * o_dir[0],
                            c_pos[1] + kCO * o_dir[1],
                            c_pos[2] + kCO * o_dir[2]};

    const float b_vec[3] = {ca[0] - n_pos[0], ca[1] - n_pos[1],
                            ca[2] - n_pos[2]};
    const float c_vec[3] = {c_pos[0] - ca[0], c_pos[1] - ca[1],
                            c_pos[2] - ca[2]};
    const float a_vec[3] = {b_vec[1] * c_vec[2] - b_vec[2] * c_vec[1],
                            b_vec[2] * c_vec[0] - b_vec[0] * c_vec[2],
                            b_vec[0] * c_vec[1] - b_vec[1] * c_vec[0]};

    constexpr float kAlpha = -0.58273431F;
    constexpr float kBeta = 0.56802827F;
    constexpr float kGamma = -0.54067466F;

    const float cb_pos[3] = {
        ca[0] + kAlpha * a_vec[0] + kBeta * b_vec[0] + kGamma * c_vec[0],
        ca[1] + kAlpha * a_vec[1] + kBeta * b_vec[1] + kGamma * c_vec[1],
        ca[2] + kAlpha * a_vec[2] + kBeta * b_vec[2] + kGamma * c_vec[2],
    };

    auto store = [&](std::size_t atom, const float p[3]) {
      const std::size_t base = atom * kCoordinateAxisCount;
      t.coords[base + 0] = p[0];
      t.coords[base + 1] = p[1];
      t.coords[base + 2] = p[2];
    };
    store(static_cast<std::size_t>(universal::CanonicalAtom::N), n_pos);
    store(static_cast<std::size_t>(universal::CanonicalAtom::CA), ca);
    store(static_cast<std::size_t>(universal::CanonicalAtom::C), c_pos);
    store(static_cast<std::size_t>(universal::CanonicalAtom::O), o_pos);
    store(static_cast<std::size_t>(universal::CanonicalAtom::CB), cb_pos);
    return t;
  }();
  return tmpl;
}

}  // namespace

ResidueIdentity classify_residue_name(std::string_view residue_name) noexcept {
  const std::string_view trimmed = trim_residue_name(residue_name);
  for (const auto& entry : kCanonicalEntries) {
    if (equal_ignore_case_3(trimmed, entry.three_letter)) {
      return ResidueIdentity{entry.one_letter, entry.three_letter,
                             entry.three_letter};
    }
  }
  for (const auto& entry : kModifiedWhitelist) {
    if (equal_ignore_case_3(trimmed, entry.three_letter)) {
      return ResidueIdentity{entry.canonical_one_letter, entry.three_letter,
                             entry.canonical_three_letter};
    }
  }
  return ResidueIdentity{kUnknownResidueCode, std::string_view{},
                         std::string_view{}};
}

bool is_modified_amino_acid(std::string_view residue_name) noexcept {
  const std::string_view trimmed = trim_residue_name(residue_name);
  for (const auto& entry : kModifiedWhitelist) {
    if (equal_ignore_case_3(trimmed, entry.three_letter)) {
      return true;
    }
  }
  return false;
}

const ResidueTemplate& canonical_residue_template() noexcept {
  return build_template();
}

}  // namespace hikoboshi::io

#ifndef HIKOBOSHI_IO_RESIDUE_TABLE_HPP
#define HIKOBOSHI_IO_RESIDUE_TABLE_HPP

#include <array>
#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

inline constexpr char kUnknownResidueCode = 'X';
inline constexpr std::size_t kCanonicalAtomCount =
    universal::kCanonicalAtomCount;
inline constexpr std::size_t kCoordinateAxisCount =
    universal::kCoordinateAxisCount;

struct ResidueIdentity {
  char one_letter;
  std::string_view three_letter;
  std::string_view canonical_three_letter;
};

ResidueIdentity classify_residue_name(std::string_view residue_name) noexcept;

bool is_modified_amino_acid(std::string_view residue_name) noexcept;

struct ResidueTemplate {
  std::array<float, kCanonicalAtomCount * kCoordinateAxisCount> coords;
};

const ResidueTemplate& canonical_residue_template() noexcept;

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_RESIDUE_TABLE_HPP

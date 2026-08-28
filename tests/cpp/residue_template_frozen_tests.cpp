#include <hikoboshi/io/residue_table.hpp>
#include <hikoboshi/universal/structure.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace hikoboshi::io {
bool compute_glycine_virtual_cb(const float n[3], const float ca[3],
                                const float c[3], float out[3]) noexcept;
}  // namespace hikoboshi::io

namespace pio = hikoboshi::io;
namespace pu = hikoboshi::universal;

namespace {

using Point = std::array<float, 3>;

void fail(const char* tag) {
  std::fprintf(stderr, "residue_template_frozen_tests: %s\n", tag);
  std::exit(1);
}

bool nearly_equal(float a, float b, float tolerance = 2.0e-5F) {
  return std::fabs(a - b) <= tolerance;
}

Point atom_point(const pio::ResidueTemplate& tmpl, pu::CanonicalAtom atom) {
  const std::size_t base = static_cast<std::size_t>(atom) * 3;
  return Point{tmpl.coords[base + 0], tmpl.coords[base + 1],
               tmpl.coords[base + 2]};
}

float distance(Point a, Point b) {
  const float dx = a[0] - b[0];
  const float dy = a[1] - b[1];
  const float dz = a[2] - b[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void expect_near(float actual, float expected, const char* tag,
                 float tolerance = 2.0e-5F) {
  if (!nearly_equal(actual, expected, tolerance)) {
    std::fprintf(stderr,
                 "residue_template_frozen_tests: %s actual=%0.8f expected=%0.8f\n",
                 tag, static_cast<double>(actual),
                 static_cast<double>(expected));
    std::exit(1);
  }
}

void expect_point(Point actual, Point expected, const char* tag) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (!nearly_equal(actual[axis], expected[axis])) {
      std::fprintf(stderr,
                   "residue_template_frozen_tests: %s axis=%zu actual=%0.8f "
                   "expected=%0.8f\n",
                   tag, axis, static_cast<double>(actual[axis]),
                   static_cast<double>(expected[axis]));
      std::exit(1);
    }
  }
}

void expect_identity(std::string_view residue_name, char expected_one_letter,
                     std::string_view expected_three_letter,
                     std::string_view expected_canonical,
                     bool expected_modified) {
  const auto identity = pio::classify_residue_name(residue_name);
  if (identity.one_letter != expected_one_letter) {
    fail("one-letter residue identity changed");
  }
  if (identity.three_letter != expected_three_letter) {
    fail("three-letter residue identity changed");
  }
  if (identity.canonical_three_letter != expected_canonical) {
    fail("canonical residue identity changed");
  }
  if (pio::is_modified_amino_acid(residue_name) != expected_modified) {
    fail("modified-residue whitelist changed");
  }
}

void test_canonical_residue_identities_are_frozen() {
  struct Case {
    std::string_view three_letter;
    char one_letter;
  };

  constexpr std::array<Case, 20> cases = {{
      {"ALA", 'A'}, {"ARG", 'R'}, {"ASN", 'N'}, {"ASP", 'D'}, {"CYS", 'C'},
      {"GLN", 'Q'}, {"GLU", 'E'}, {"GLY", 'G'}, {"HIS", 'H'}, {"ILE", 'I'},
      {"LEU", 'L'}, {"LYS", 'K'}, {"MET", 'M'}, {"PHE", 'F'}, {"PRO", 'P'},
      {"SER", 'S'}, {"THR", 'T'}, {"TRP", 'W'}, {"TYR", 'Y'}, {"VAL", 'V'},
  }};

  for (const auto& test_case : cases) {
    expect_identity(test_case.three_letter, test_case.one_letter,
                    test_case.three_letter, test_case.three_letter, false);
  }

  const auto unknown = pio::classify_residue_name("ZZZ");
  if (unknown.one_letter != pio::kUnknownResidueCode) {
    fail("unknown residue code changed");
  }
}

void test_modified_residue_whitelist_is_frozen() {
  expect_identity("MSE", 'M', "MSE", "MET", true);
  expect_identity("SEC", 'C', "SEC", "CYS", true);
  expect_identity("PYL", 'K', "PYL", "LYS", true);
  expect_identity("SEP", 'S', "SEP", "SER", true);
  expect_identity("TPO", 'T', "TPO", "THR", true);
  expect_identity("PTR", 'Y', "PTR", "TYR", true);
  expect_identity("HYP", 'P', "HYP", "PRO", true);

  expect_identity(" mse ", 'M', "MSE", "MET", true);
}

void test_residue_template_coordinates_are_frozen() {
  const auto& tmpl = pio::canonical_residue_template();

  expect_point(atom_point(tmpl, pu::CanonicalAtom::N),
               Point{-1.4580000F, 0.0F, 0.0F}, "template N");
  expect_point(atom_point(tmpl, pu::CanonicalAtom::CA),
               Point{0.0F, 0.0F, 0.0F}, "template CA");
  expect_point(atom_point(tmpl, pu::CanonicalAtom::C),
               Point{1.4213556F, 0.5526060F, 0.0F}, "template C");
  expect_point(atom_point(tmpl, pu::CanonicalAtom::O),
               Point{1.6192990F, 1.7675873F, 0.0F}, "template O");
  expect_point(atom_point(tmpl, pu::CanonicalAtom::CB),
               Point{0.0596943F, -0.2987801F, -0.4695088F},
               "template CB");
}

void test_template_source_literals_are_frozen() {
  const auto& tmpl = pio::canonical_residue_template();
  const Point n = atom_point(tmpl, pu::CanonicalAtom::N);
  const Point ca = atom_point(tmpl, pu::CanonicalAtom::CA);
  const Point c = atom_point(tmpl, pu::CanonicalAtom::C);
  const Point o = atom_point(tmpl, pu::CanonicalAtom::O);

  expect_near(distance(n, ca), 1.458F, "N-CA length");
  expect_near(distance(ca, c), 1.525F, "CA-C length");
  expect_near(distance(c, o), 1.231F, "C-O length");
}

void test_virtual_cb_coefficients_are_frozen() {
  const float n[3] = {0.0F, 0.0F, 0.0F};
  const float ca[3] = {1.0F, 0.0F, 0.0F};
  const float c[3] = {1.0F, 1.0F, 0.0F};
  float cb[3] = {0.0F, 0.0F, 0.0F};

  if (!pio::compute_glycine_virtual_cb(n, ca, c, cb)) {
    fail("virtual CB helper unexpectedly failed");
  }

  expect_point(Point{cb[0], cb[1], cb[2]},
               Point{1.5680283F, -0.5406747F, -0.5827343F},
               "virtual CB coefficients");
}

}  // namespace

int main() {
  test_canonical_residue_identities_are_frozen();
  test_modified_residue_whitelist_is_frozen();
  test_residue_template_coordinates_are_frozen();
  test_template_source_literals_are_frozen();
  test_virtual_cb_coefficients_are_frozen();
  std::fprintf(stdout, "residue_template_frozen_tests: ok\n");
  return 0;
}

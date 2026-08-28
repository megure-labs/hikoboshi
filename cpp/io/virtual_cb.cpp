// Glycine virtual-CB constants and helper. Implements exactly the chartered
// formula from STRUCTURE_INPUT_CHARTER.md so the inferred atom is marked as
// Virtual rather than Inferred. The coefficient literals are frozen alongside
// the residue template constants by tests/cpp/residue_template_frozen_tests.cpp.

#include <hikoboshi/io/atom_inference.hpp>

#include <cstddef>

#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

namespace {

constexpr float kGlycineCbAlpha = -0.58273431F;
constexpr float kGlycineCbBeta = 0.56802827F;
constexpr float kGlycineCbGamma = -0.54067466F;

}  // namespace

bool glycine_virtual_cb(const float n[3],
                        const float ca[3],
                        const float c[3],
                        float out[3]) noexcept {
  const float b[3] = {ca[0] - n[0], ca[1] - n[1], ca[2] - n[2]};
  const float c_vec[3] = {c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]};
  const float a[3] = {b[1] * c_vec[2] - b[2] * c_vec[1],
                      b[2] * c_vec[0] - b[0] * c_vec[2],
                      b[0] * c_vec[1] - b[1] * c_vec[0]};

  out[0] = ca[0] + kGlycineCbAlpha * a[0] + kGlycineCbBeta * b[0] +
           kGlycineCbGamma * c_vec[0];
  out[1] = ca[1] + kGlycineCbAlpha * a[1] + kGlycineCbBeta * b[1] +
           kGlycineCbGamma * c_vec[1];
  out[2] = ca[2] + kGlycineCbAlpha * a[2] + kGlycineCbBeta * b[2] +
           kGlycineCbGamma * c_vec[2];
  return true;
}

bool compute_glycine_virtual_cb(const float n[3],
                                const float ca[3],
                                const float c[3],
                                float out[3]) noexcept {
  return glycine_virtual_cb(n, ca, c, out);
}

}  // namespace hikoboshi::io

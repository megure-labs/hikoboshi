#ifndef HIKOBOSHI_IO_ATOM_INFERENCE_HPP
#define HIKOBOSHI_IO_ATOM_INFERENCE_HPP

#include <cstddef>

#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

struct AtomInferenceKabschUseRequest {
  const float* template_coordinates = nullptr;  // row-major [N, 3]
  const float* coordinates = nullptr;           // row-major [N, 3]
  const universal::AtomSource* atom_sources = nullptr;  // row-major [N]
  std::size_t template_atom_count = 0;
  float rmsd_guard = 0.5F;
};

struct AtomInferenceKabschUseOutput {
  float* coordinates = nullptr;  // row-major [N, 3]
  universal::AtomSource* atom_sources = nullptr;  // row-major [N]
};

bool glycine_virtual_cb(const float n[3],
                        const float ca[3],
                        const float c[3],
                        float out[3]) noexcept;

bool compute_glycine_virtual_cb(const float n[3],
                                const float ca[3],
                                const float c[3],
                                float out[3]) noexcept;

bool atom_inference_kabsch_use(
    const AtomInferenceKabschUseRequest& request,
    const AtomInferenceKabschUseOutput& output) noexcept;

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_ATOM_INFERENCE_HPP

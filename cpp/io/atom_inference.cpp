// Per-residue atom inference using a rigid template fit. Implements the
// chartered guard (>=3 non-collinear anchors and RMSD <= 0.5 A) from
// STRUCTURE_INPUT_CHARTER.md and refuses to infer when the guard fails.
// Glycine residues use the chartered virtual-CB construction and mark CB as
// Virtual rather than Inferred.
//
// Virtual-CB policy: the chartered formula is computed from whatever N, CA,
// and C coordinates are present and written into the CB cell unconditionally
// when the previous CB slot was Missing. The atom_sources mask still records
// CB as Virtual only when N/CA/C were all observed (otherwise CB stays
// Missing), so downstream consumers can mask invalid residues out of the
// KNN/edge graph. This matches PyTorch ProteinMPNN's `_get_cb` which
// computes the formula for every residue and carries validity through
// `mask`, instead of zeroing the CB cell at invalid residues. The previous
// Hikoboshi behavior (zeroing CB when any of N/CA/C was Missing) showed up as
// a 15.94 max-abs virtual-CB miss on `d3ku8a_` residue 134 in
// bench/MPNN64_INPUT_STAGE_PARITY.md, contributing to the d3ku8a_ outlier
// in mp8 synthesis.

#include <hikoboshi/io/structure_loader.hpp>

#include <hikoboshi/io/atom_inference.hpp>
#include <hikoboshi/io/residue_table.hpp>

#include <cstddef>

#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

namespace {

constexpr float kInferRmsdGuard = 0.5F;

}  // namespace

universal::Status infer_missing_atoms(LoadedStructure& structure) {
  auto& impl = structure.impl();
  const ResidueTemplate& tmpl = canonical_residue_template();

  for (std::size_t i = 0; i < impl.normalized.size(); ++i) {
    auto& slot = impl.normalized[i];
    const std::size_t base = i * 5 * 3;
    const std::size_t base_src = i * 5;

    // Glycine virtual CB construction.
    if (slot.residue_code == 'G') {
      const bool need_cb =
          impl.atom_sources[base_src + 4] == universal::AtomSource::Missing;
      const bool have_n =
          impl.atom_sources[base_src + 0] == universal::AtomSource::Observed;
      const bool have_ca =
          impl.atom_sources[base_src + 1] == universal::AtomSource::Observed;
      const bool have_c =
          impl.atom_sources[base_src + 2] == universal::AtomSource::Observed;
      if (need_cb) {
        // Always compute the chartered virtual-CB formula on whatever N, CA,
        // C coordinates are present, mirroring PyTorch ProteinMPNN's
        // `_get_cb`. Validity is carried through atom_sources, not by
        // zeroing the CB cell.
        const float n[3] = {impl.coordinates[base + 0],
                            impl.coordinates[base + 1],
                            impl.coordinates[base + 2]};
        const float ca[3] = {impl.coordinates[base + 3],
                             impl.coordinates[base + 4],
                             impl.coordinates[base + 5]};
        const float c[3] = {impl.coordinates[base + 6],
                            impl.coordinates[base + 7],
                            impl.coordinates[base + 8]};
        float cb[3] = {0.0F, 0.0F, 0.0F};
        glycine_virtual_cb(n, ca, c, cb);
        impl.coordinates[base + 12] = cb[0];
        impl.coordinates[base + 13] = cb[1];
        impl.coordinates[base + 14] = cb[2];
        const bool fully_anchored = have_n && have_ca && have_c;
        const universal::AtomSource cb_source =
            fully_anchored ? universal::AtomSource::Virtual
                           : universal::AtomSource::Missing;
        impl.atom_sources[base_src + 4] = cb_source;
        slot.atom_sources[4] = cb_source;
        slot.coords[4][0] = cb[0];
        slot.coords[4][1] = cb[1];
        slot.coords[4][2] = cb[2];
      }
      continue;
    }

    const bool inferred = atom_inference_kabsch_use(
        AtomInferenceKabschUseRequest{tmpl.coords.data(),
                                      impl.coordinates.data() + base,
                                      impl.atom_sources.data() + base_src,
                                      universal::kCanonicalAtomCount,
                                      kInferRmsdGuard},
        AtomInferenceKabschUseOutput{impl.coordinates.data() + base,
                                     impl.atom_sources.data() + base_src});
    if (!inferred) {
      continue;
    }

    for (std::size_t atom = 0; atom < universal::kCanonicalAtomCount; ++atom) {
      slot.atom_sources[atom] = impl.atom_sources[base_src + atom];
      for (std::size_t axis = 0; axis < universal::kCoordinateAxisCount;
           ++axis) {
        slot.coords[atom][axis] =
            impl.coordinates[base + atom * universal::kCoordinateAxisCount +
                             axis];
      }
    }
  }

  return universal::ok_status();
}

}  // namespace hikoboshi::io

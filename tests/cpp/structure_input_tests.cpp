#include <hikoboshi/io/embedding_loader.hpp>
#include <hikoboshi/io/residue_table.hpp>
#include <hikoboshi/io/structure_loader.hpp>
#include <hikoboshi/universal/structure.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace pio = hikoboshi::io;
namespace pu = hikoboshi::universal;

namespace {

void fail(const char* tag) {
  std::fprintf(stderr, "structure_input_tests: %s\n", tag);
  std::exit(1);
}

bool nearly_equal(float a, float b, float tolerance = 1e-3F) {
  return std::fabs(a - b) <= tolerance;
}

float coord_at(const pu::StructureView& view, std::size_t residue,
               std::size_t atom, std::size_t axis) {
  return view.coordinates.data[residue * 5 * 3 + atom * 3 + axis];
}

pu::AtomSource source_at(const pu::StructureView& view, std::size_t residue,
                         std::size_t atom) {
  return view.atom_sources.data[residue * 5 + atom];
}

const std::string_view kPdbBasic =
    "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1      12.500  13.000  10.500  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1      13.420  14.180  10.250  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1      14.620  13.980  10.350  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1      12.700  12.700  11.980  1.00  0.00           C\n"
    "ATOM      6  N   GLY A   2      13.000  15.250   9.700  1.00  0.00           N\n"
    "ATOM      7  CA  GLY A   2      13.700  16.500   9.300  1.00  0.00           C\n"
    "ATOM      8  C   GLY A   2      13.100  17.200   8.100  1.00  0.00           C\n"
    "ATOM      9  O   GLY A   2      11.900  17.250   7.900  1.00  0.00           O\n"
    "ATOM     10  N   SER A   3      14.000  17.700   7.300  1.00  0.00           N\n"
    "ATOM     11  CA  SER A   3      13.700  18.450   6.100  1.00  0.00           C\n"
    "ATOM     12  C   SER A   3      14.700  19.560   5.860  1.00  0.00           C\n"
    "ATOM     13  O   SER A   3      15.880  19.300   5.610  1.00  0.00           O\n"
    "ATOM     14  CB  SER A   3      13.800  17.500   4.900  1.00  0.00           C\n"
    "TER\n"
    "END\n";

const std::string_view kPdbMultiModel =
    "MODEL        1\n"
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1       1.500   1.500   0.000  1.00  0.00           C\n"
    "ENDMDL\n"
    "MODEL        2\n"
    "ATOM      1  N   ALA A   1     100.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1     101.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1     102.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1     103.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1     101.500   1.500   0.000  1.00  0.00           C\n"
    "ENDMDL\n";

const std::string_view kPdbAltloc =
    "ATOM      1  N  AALA A   1       0.000   0.000   0.000  0.50  0.00           N\n"
    "ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.50  0.00           C\n"
    "ATOM      3  C  AALA A   1       2.000   0.000   0.000  0.50  0.00           C\n"
    "ATOM      4  O  AALA A   1       3.000   0.000   0.000  0.50  0.00           O\n"
    "ATOM      5  CB AALA A   1       1.500   1.500   0.000  0.50  0.00           C\n"
    "ATOM      6  N  BALA A   1     999.999  99.999   0.000  0.50  0.00           N\n"
    "ATOM      7  CA BALA A   1     999.999  99.999   0.000  0.50  0.00           C\n"
    "TER\n";

const std::string_view kPdbMse =
    "HETATM    1  N   MSE A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "HETATM    2  CA  MSE A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "HETATM    3  C   MSE A   1       2.000   0.000   0.000  1.00  0.00           C\n"
    "HETATM    4  O   MSE A   1       3.000   0.000   0.000  1.00  0.00           O\n"
    "HETATM    5  CB  MSE A   1       1.500   1.500   0.000  1.00  0.00           C\n"
    "HETATM    6  O   HOH A 100      50.000  50.000  50.000  1.00  0.00           O\n";

const std::string_view kPdbBadCoord =
    "ATOM      1  N   ALA A   1     XX.XXX   0.000   0.000  1.00  0.00           N\n";

// Coordinates here are the canonical residue template translated by
// (+1.458, 0, 0) so the rigid fit reproduces the template exactly. The CB
// slot is intentionally omitted to exercise the inference path.
const std::string_view kPdbGlycineMissingCb =
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1       2.879   0.553   0.000  1.00  0.00           C\n"
    "ATOM      4  O   GLY A   1       3.076   1.768   0.000  1.00  0.00           O\n";

const std::string_view kPdbMissingCbAla =
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.879   0.553   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.076   1.768   0.000  1.00  0.00           O\n";

const std::string_view kPdbResidueGap =
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1       1.500   1.500   0.000  1.00  0.00           C\n"
    "ATOM      6  N   ALA A   2       4.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      7  CA  ALA A   2       5.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      8  C   ALA A   2       6.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      9  O   ALA A   2       7.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM     10  CB  ALA A   2       5.500   1.500   0.000  1.00  0.00           C\n"
    "ATOM     11  N   ALA A   5      12.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM     12  CA  ALA A   5      13.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM     13  C   ALA A   5      14.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM     14  O   ALA A   5      15.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM     15  CB  ALA A   5      13.500   1.500   0.000  1.00  0.00           C\n";

const std::string_view kCifBasic =
    "data_test\n"
    "loop_\n"
    "_atom_site.group_PDB\n"
    "_atom_site.id\n"
    "_atom_site.type_symbol\n"
    "_atom_site.label_atom_id\n"
    "_atom_site.label_alt_id\n"
    "_atom_site.label_comp_id\n"
    "_atom_site.label_asym_id\n"
    "_atom_site.auth_asym_id\n"
    "_atom_site.auth_seq_id\n"
    "_atom_site.pdbx_PDB_ins_code\n"
    "_atom_site.Cartn_x\n"
    "_atom_site.Cartn_y\n"
    "_atom_site.Cartn_z\n"
    "_atom_site.occupancy\n"
    "_atom_site.auth_comp_id\n"
    "_atom_site.pdbx_PDB_model_num\n"
    "ATOM 1  N N   . ALA A A 1 ? 11.104 13.207 10.000 1.00 ALA 1\n"
    "ATOM 2  C CA  . ALA A A 1 ? 12.500 13.000 10.500 1.00 ALA 1\n"
    "ATOM 3  C C   . ALA A A 1 ? 13.420 14.180 10.250 1.00 ALA 1\n"
    "ATOM 4  O O   . ALA A A 1 ? 14.620 13.980 10.350 1.00 ALA 1\n"
    "ATOM 5  C CB  . ALA A A 1 ? 12.700 12.700 11.980 1.00 ALA 1\n"
    "ATOM 6  N N   . GLY A A 2 ? 13.000 15.250  9.700 1.00 GLY 1\n"
    "ATOM 7  C CA  . GLY A A 2 ? 13.700 16.500  9.300 1.00 GLY 1\n"
    "ATOM 8  C C   . GLY A A 2 ? 13.100 17.200  8.100 1.00 GLY 1\n"
    "ATOM 9  O O   . GLY A A 2 ? 11.900 17.250  7.900 1.00 GLY 1\n"
    "#\n";

void test_residue_classification() {
  const auto ala = pio::classify_residue_name("ALA");
  if (ala.one_letter != 'A') fail("ALA should map to 'A'");
  const auto mse = pio::classify_residue_name("MSE");
  if (mse.one_letter != 'M') fail("MSE should map to 'M' (modified)");
  if (!pio::is_modified_amino_acid("MSE")) fail("MSE should be modified");
  const auto sec = pio::classify_residue_name("SEC");
  if (sec.one_letter != 'C') fail("SEC should map to 'C'");
  const auto unk = pio::classify_residue_name("ZZZ");
  if (unk.one_letter != 'X') fail("unknown residue should map to 'X'");
  const auto leading_space = pio::classify_residue_name(" ALA");
  if (leading_space.one_letter != 'A') fail("leading-space ALA must classify");
}

void test_format_detection() {
  if (pio::detect_structure_format("foo.pdb", "") != pio::StructureFormat::Pdb)
    fail("pdb extension not detected");
  if (pio::detect_structure_format("foo.cif", "") != pio::StructureFormat::MmCif)
    fail("cif extension not detected");
  if (pio::detect_structure_format("", kCifBasic) !=
      pio::StructureFormat::MmCif)
    fail("mmCIF content not detected");
  if (pio::detect_structure_format("", kPdbBasic) != pio::StructureFormat::Pdb)
    fail("PDB content not detected");
  if (pio::detect_structure_format("foo.txt", "completely random text") !=
      pio::StructureFormat::Unknown)
    fail("Unknown format must surface");
}

void test_pdb_basic() {
  auto result = pio::load_pdb_from_string(kPdbBasic, "basic.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("basic PDB parse failed");
  const auto view = result.value.view();
  if (view.residue_count != 3) fail("basic PDB should yield 3 residues");
  if (view.residue_codes.data[0] != 'A') fail("residue 0 should be A");
  if (view.residue_codes.data[1] != 'G') fail("residue 1 should be G");
  if (view.residue_codes.data[2] != 'S') fail("residue 2 should be S");
  if (source_at(view, 0, 0) != pu::AtomSource::Observed)
    fail("residue 0 N must be Observed");
  if (source_at(view, 0, 4) != pu::AtomSource::Observed)
    fail("residue 0 CB must be Observed");
  if (!nearly_equal(coord_at(view, 0, 1, 0), 12.5F))
    fail("residue 0 CA x mismatch");
}

void test_pdb_default_chain() {
  // Two chains, A (ALA) and B (VAL); default should pick A.
  const std::string content = std::string{kPdbBasic} +
      "ATOM     20  N   VAL B   1      30.000  30.000  30.000  1.00  0.00           N\n"
      "ATOM     21  CA  VAL B   1      31.000  30.000  30.000  1.00  0.00           C\n"
      "ATOM     22  C   VAL B   1      32.000  30.000  30.000  1.00  0.00           C\n"
      "ATOM     23  O   VAL B   1      33.000  30.000  30.000  1.00  0.00           O\n"
      "ATOM     24  CB  VAL B   1      31.500  31.500  30.000  1.00  0.00           C\n";
  auto result = pio::load_pdb_from_string(content, "two_chain.pdb");
  if (result.status.code != pu::StatusCode::Ok)
    fail("two-chain PDB parse failed");
  if (result.value.selected_chain_id() != "A")
    fail("default chain should be A");
  if (result.value.residue_count() != 3)
    fail("default chain should drop chain B");
}

void test_pdb_default_chain_ignores_hetatm_only_chain() {
  const std::string_view content =
      "HETATM    1  O   HOH W   1      50.000  50.000  50.000  1.00  0.00           O\n"
      "ATOM      2  N   VAL B   1       0.000   0.000   0.000  1.00  0.00           N\n"
      "ATOM      3  CA  VAL B   1       1.000   0.000   0.000  1.00  0.00           C\n"
      "ATOM      4  C   VAL B   1       2.000   0.000   0.000  1.00  0.00           C\n"
      "ATOM      5  O   VAL B   1       3.000   0.000   0.000  1.00  0.00           O\n"
      "ATOM      6  CB  VAL B   1       1.500   1.500   0.000  1.00  0.00           C\n";
  auto result = pio::load_pdb_from_string(content, "hetatm_first.pdb");
  if (result.status.code != pu::StatusCode::Ok)
    fail("default chain should ignore HETATM-only chains");
  if (result.value.selected_chain_id() != "B")
    fail("default chain should choose first polymer chain after HETATM strip");
  if (result.value.residue_count() != 1)
    fail("default chain should keep the VAL polymer residue");
}

void test_pdb_model_default() {
  auto result = pio::load_pdb_from_string(kPdbMultiModel, "multi_model.pdb");
  if (result.status.code != pu::StatusCode::Ok)
    fail("multi-model PDB parse failed");
  if (result.value.selected_model_index() != 1)
    fail("default model should be 1");
  const auto view = result.value.view();
  if (!nearly_equal(coord_at(view, 0, 0, 0), 0.0F))
    fail("default model should pick model 1 N coordinate");
}

void test_pdb_model_override() {
  pio::StructureLoadOptions opts;
  opts.model_index = 2;
  auto result =
      pio::load_pdb_from_string(kPdbMultiModel, "multi_model.pdb", opts);
  if (result.status.code != pu::StatusCode::Ok) fail("model override failed");
  const auto view = result.value.view();
  if (!nearly_equal(coord_at(view, 0, 0, 0), 100.0F))
    fail("model 2 N coordinate mismatch");
}

void test_pdb_altloc() {
  auto result = pio::load_pdb_from_string(kPdbAltloc, "altloc.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("altloc PDB parse failed");
  const auto view = result.value.view();
  if (view.residue_count != 1) fail("altloc collapse should yield 1 residue");
  // 'A' altloc should win over 'B' altloc.
  if (!nearly_equal(coord_at(view, 0, 0, 0), 0.0F))
    fail("altloc A should win over B");
  if (coord_at(view, 0, 0, 0) > 500.0F) fail("altloc B coords leaked");
}

void test_pdb_altloc_blank_preference_is_per_atom() {
  const std::string_view content =
      "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
      "ATOM      2  CA AALA A   1     100.000   0.000   0.000  0.50  0.00           C\n"
      "ATOM      3  CA  ALA A   1       1.000   0.000   0.000  0.50  0.00           C\n"
      "ATOM      4  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
      "ATOM      5  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n"
      "ATOM      6  CB  ALA A   1       1.500   1.500   0.000  1.00  0.00           C\n";
  auto result = pio::load_pdb_from_string(content, "altloc_per_atom.pdb");
  if (result.status.code != pu::StatusCode::Ok)
    fail("per-atom altloc PDB parse failed");
  const auto view = result.value.view();
  if (!nearly_equal(coord_at(view, 0, 1, 0), 1.0F))
    fail("blank CA altloc should win even after a blank N was seen");
}

void test_pdb_modified_residue() {
  auto result = pio::load_pdb_from_string(kPdbMse, "mse.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("MSE PDB parse failed");
  const auto view = result.value.view();
  if (view.residue_count != 1)
    fail("HETATM HOH must be filtered; only MSE remains");
  if (view.residue_codes.data[0] != 'M')
    fail("MSE must map to one-letter code M");
  if (view.residues.data[0].original_residue_name != "MSE")
    fail("MSE original name must be preserved in metadata");
}

void test_pdb_invalid_coordinate_no_zero_fill() {
  auto result =
      pio::load_pdb_from_string(kPdbBadCoord, "bad.pdb");
  if (result.status.code == pu::StatusCode::Ok)
    fail("invalid coordinate must not silently parse");
  if (result.status.code != pu::StatusCode::InvalidArgument)
    fail("invalid coordinate must surface InvalidArgument");
}

void test_pdb_rejects_nonfinite_coordinate() {
  std::string content{kPdbBasic.substr(0, kPdbBasic.find('\n') + 1)};
  content.replace(30, 8, "     nan");
  auto result = pio::load_pdb_from_string(content, "nan_coord.pdb");
  if (result.status.code != pu::StatusCode::InvalidArgument)
    fail("PDB NaN coordinate must be rejected");
}

void test_pdb_glycine_virtual_cb() {
  auto result =
      pio::load_pdb_from_string(kPdbGlycineMissingCb, "gly.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("glycine parse failed");
  const auto view = result.value.view();
  if (view.residue_count != 1) fail("glycine: one residue expected");
  if (source_at(view, 0, 4) != pu::AtomSource::Virtual)
    fail("glycine CB must be Virtual");
  // CB must not be (0,0,0); that would be a silent zero-fill miss.
  const float x = coord_at(view, 0, 4, 0);
  const float y = coord_at(view, 0, 4, 1);
  const float z = coord_at(view, 0, 4, 2);
  if (x == 0.0F && y == 0.0F && z == 0.0F)
    fail("glycine virtual CB must be computed, not zero-filled");
}

void test_pdb_atom_inference_marks_inferred() {
  auto result =
      pio::load_pdb_from_string(kPdbMissingCbAla, "missing_cb.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("missing-CB ALA parse failed");
  const auto view = result.value.view();
  if (view.residue_count != 1) fail("missing-CB ALA: one residue expected");
  if (source_at(view, 0, 4) != pu::AtomSource::Inferred)
    fail("missing CB on ALA with anchors should be Inferred");
}

void test_pdb_atom_inference_guard() {
  // Only N + CA observed; guard requires >=3 anchors so CB must remain Missing.
  const std::string_view two_anchor =
      "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
      "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n";
  auto result = pio::load_pdb_from_string(two_anchor, "two_anchor.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("two-anchor parse failed");
  const auto view = result.value.view();
  if (source_at(view, 0, 2) != pu::AtomSource::Missing)
    fail("two-anchor C must remain Missing");
  if (source_at(view, 0, 4) != pu::AtomSource::Missing)
    fail("two-anchor CB must remain Missing");
  // Coordinates of missing atoms must remain zero by initialization, but the
  // sources must be Missing so consumers know to skip them.
}

void test_pdb_chain_break_metadata() {
  auto result =
      pio::load_pdb_from_string(kPdbResidueGap, "residue_gap.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("residue gap parse failed");
  const auto view = result.value.view();
  if (view.residue_count != 3) fail("residue gap: should have 3 residues observed");
  if (view.chain_breaks.size != 1)
    fail("residue gap: should report one chain break");
  if (view.chain_breaks.data[0].after_residue_index != 1)
    fail("chain break should be after residue index 1 (between 2 and 5)");
}

void test_mmcif_basic() {
  auto result = pio::load_mmcif_from_string(kCifBasic, "basic.cif");
  if (result.status.code != pu::StatusCode::Ok) {
    std::fprintf(stderr, "mmCIF parse status: code=%d detail=%s\n",
                 static_cast<int>(result.status.code),
                 result.status.detail ? result.status.detail : "(null)");
    fail("basic mmCIF parse failed");
  }
  const auto view = result.value.view();
  if (view.residue_count != 2)
    fail("mmCIF basic should yield 2 residues (ALA, GLY)");
  if (view.residue_codes.data[0] != 'A') fail("mmCIF residue 0 should be A");
  if (view.residue_codes.data[1] != 'G') fail("mmCIF residue 1 should be G");
  if (source_at(view, 1, 4) != pu::AtomSource::Virtual)
    fail("mmCIF GLY CB should be Virtual after inference");
}

void test_mmcif_rejects_trailing_coordinate_garbage() {
  std::string content{kCifBasic};
  const std::size_t pos = content.find("11.104");
  if (pos == std::string::npos) fail("test fixture missing coordinate token");
  content.replace(pos, 6, "11.104abc");
  auto result = pio::load_mmcif_from_string(content, "bad_coord.cif");
  if (result.status.code != pu::StatusCode::InvalidArgument)
    fail("mmCIF coordinate with trailing garbage must be rejected");
}

void test_atom_source_propagation() {
  // Build a residue with everything observed except O. Inference should fill
  // O and mark it Inferred while leaving the other slots unchanged.
  // Coordinates here are the canonical residue template translated by
  // (+1.458, 0, 0) so the fit is exact.
  const std::string_view content =
      "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
      "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
      "ATOM      3  C   ALA A   1       2.879   0.553   0.000  1.00  0.00           C\n"
      "ATOM      4  CB  ALA A   1       1.518  -0.299  -0.470  1.00  0.00           C\n";
  auto result = pio::load_pdb_from_string(content, "missing_o.pdb");
  if (result.status.code != pu::StatusCode::Ok) fail("missing-O parse failed");
  const auto view = result.value.view();
  if (source_at(view, 0, 0) != pu::AtomSource::Observed)
    fail("N should remain Observed");
  if (source_at(view, 0, 1) != pu::AtomSource::Observed)
    fail("CA should remain Observed");
  if (source_at(view, 0, 2) != pu::AtomSource::Observed)
    fail("C should remain Observed");
  if (source_at(view, 0, 4) != pu::AtomSource::Observed)
    fail("CB should remain Observed");
  if (source_at(view, 0, 3) != pu::AtomSource::Inferred)
    fail("missing O should be Inferred");
}

void test_inference_disabled() {
  pio::StructureLoadOptions opts;
  opts.infer_missing_atoms = false;
  auto result = pio::load_pdb_from_string(kPdbMissingCbAla, "no_infer.pdb",
                                          opts);
  if (result.status.code != pu::StatusCode::Ok) fail("no-infer parse failed");
  const auto view = result.value.view();
  if (source_at(view, 0, 4) != pu::AtomSource::Missing)
    fail("CB must remain Missing when inference is disabled");
}

void test_embedding_only_metadata_round_trip() {
  // Construct a synthetic .npy float32 array with shape (3, 4).
  std::vector<std::uint8_t> bytes;
  const char magic[] = "\x93NUMPY";
  bytes.insert(bytes.end(), magic, magic + 6);
  bytes.push_back(1);  // major version
  bytes.push_back(0);  // minor version
  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }";
  std::size_t total_header_size = 10 + header.size();
  // Pad to multiple of 16 for npy spec; whitespace plus newline at end.
  const std::size_t pad = (16 - (total_header_size % 16)) % 16;
  for (std::size_t i = 0; i < pad; ++i) {
    header.push_back(' ');
  }
  if (header.empty() || header.back() != '\n') {
    header.back() = '\n';
  }
  std::uint16_t header_len = static_cast<std::uint16_t>(header.size());
  bytes.push_back(static_cast<std::uint8_t>(header_len & 0xFF));
  bytes.push_back(static_cast<std::uint8_t>((header_len >> 8) & 0xFF));
  bytes.insert(bytes.end(), header.begin(), header.end());
  // Append 12 floats.
  for (std::size_t i = 0; i < 12; ++i) {
    float value = static_cast<float>(i) + 0.25F;
    std::uint8_t buf[sizeof(float)];
    std::memcpy(buf, &value, sizeof(float));
    bytes.insert(bytes.end(), buf, buf + sizeof(float));
  }

  pu::Span<const std::uint8_t> span{bytes.data(), bytes.size()};
  auto result =
      pio::load_embedding_from_npy_bytes(span, "fake.npy", {});
  if (result.status.code != pu::StatusCode::Ok)
    fail("embedding npy parse failed");
  const auto view = result.value.view();
  if (view.residue_count != 3) fail("embedding residue_count mismatch");
  if (view.dimension != 4) fail("embedding dimension mismatch");
  if (!nearly_equal(view.values.data[0], 0.25F))
    fail("embedding value[0] mismatch");

  // attach_residue_metadata should mark the embedding as carrying metadata.
  const char codes[] = {'A', 'G', 'S'};
  pu::Span<const char> code_span{const_cast<char*>(codes), 3};
  pio::attach_residue_metadata(result.value, code_span);
  if (!result.value.has_residue_metadata())
    fail("attach_residue_metadata should flag presence");
  if (result.value.view().residues.size != 3)
    fail("residue metadata count mismatch");
  if (result.value.view().residue_codes.data[1] != 'G')
    fail("residue code propagation mismatch");
}

void test_embedding_rejects_non_float32() {
  std::vector<std::uint8_t> bytes;
  const char magic[] = "\x93NUMPY";
  bytes.insert(bytes.end(), magic, magic + 6);
  bytes.push_back(1);
  bytes.push_back(0);
  std::string header =
      "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 1), }";
  std::size_t total_header_size = 10 + header.size();
  const std::size_t pad = (16 - (total_header_size % 16)) % 16;
  for (std::size_t i = 0; i < pad; ++i) header.push_back(' ');
  std::uint16_t header_len = static_cast<std::uint16_t>(header.size());
  bytes.push_back(static_cast<std::uint8_t>(header_len & 0xFF));
  bytes.push_back(static_cast<std::uint8_t>((header_len >> 8) & 0xFF));
  bytes.insert(bytes.end(), header.begin(), header.end());
  double v = 1.0;
  std::uint8_t buf[sizeof(double)];
  std::memcpy(buf, &v, sizeof(double));
  bytes.insert(bytes.end(), buf, buf + sizeof(double));

  pu::Span<const std::uint8_t> span{bytes.data(), bytes.size()};
  auto result = pio::load_embedding_from_npy_bytes(span, "fake.npy", {});
  if (result.status.code != pu::StatusCode::InvalidArgument)
    fail("non-float32 npy must be rejected");
}

void test_embedding_rejects_big_endian_float32() {
  std::vector<std::uint8_t> bytes;
  const char magic[] = "\x93NUMPY";
  bytes.insert(bytes.end(), magic, magic + 6);
  bytes.push_back(1);
  bytes.push_back(0);
  std::string header =
      "{'descr': '>f4', 'fortran_order': False, 'shape': (1, 1), }";
  std::size_t total_header_size = 10 + header.size();
  const std::size_t pad = (16 - (total_header_size % 16)) % 16;
  for (std::size_t i = 0; i < pad; ++i) header.push_back(' ');
  std::uint16_t header_len = static_cast<std::uint16_t>(header.size());
  bytes.push_back(static_cast<std::uint8_t>(header_len & 0xFF));
  bytes.push_back(static_cast<std::uint8_t>((header_len >> 8) & 0xFF));
  bytes.insert(bytes.end(), header.begin(), header.end());
  const std::uint8_t one_big_endian[] = {0x3F, 0x80, 0x00, 0x00};
  bytes.insert(bytes.end(), one_big_endian, one_big_endian + 4);

  pu::Span<const std::uint8_t> span{bytes.data(), bytes.size()};
  auto result = pio::load_embedding_from_npy_bytes(span, "big.npy", {});
  if (result.status.code != pu::StatusCode::InvalidArgument)
    fail("big-endian float32 npy must be rejected");
}

}  // namespace

int main() {
  test_residue_classification();
  test_format_detection();
  test_pdb_basic();
  test_pdb_default_chain();
  test_pdb_default_chain_ignores_hetatm_only_chain();
  test_pdb_model_default();
  test_pdb_model_override();
  test_pdb_altloc();
  test_pdb_altloc_blank_preference_is_per_atom();
  test_pdb_modified_residue();
  test_pdb_invalid_coordinate_no_zero_fill();
  test_pdb_rejects_nonfinite_coordinate();
  test_pdb_glycine_virtual_cb();
  test_pdb_atom_inference_marks_inferred();
  test_pdb_atom_inference_guard();
  test_pdb_chain_break_metadata();
  test_mmcif_basic();
  test_mmcif_rejects_trailing_coordinate_garbage();
  test_atom_source_propagation();
  test_inference_disabled();
  test_embedding_only_metadata_round_trip();
  test_embedding_rejects_non_float32();
  test_embedding_rejects_big_endian_float32();
  std::fprintf(stdout, "structure_input_tests: ok\n");
  return 0;
}

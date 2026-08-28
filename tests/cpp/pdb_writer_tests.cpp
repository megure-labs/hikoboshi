#include <hikoboshi/io/pdb_writer.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::api;
namespace pio = hikoboshi::io;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "pdb_writer_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double a, double b, double tolerance = 1.0e-6) {
  return std::fabs(a - b) <= tolerance;
}

std::size_t count_occurrences(const std::string& text,
                              std::string_view needle) {
  std::size_t count = 0;
  std::size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

struct StructureFixture {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;
  std::vector<hiko_u::ResidueMetadataView> residues;
  std::string input_id;

  StructureFixture(std::string id,
                   char chain,
                   int model,
                   std::size_t residue_count)
      : coordinates(residue_count * hiko_u::kCanonicalAtomCount *
                        hiko_u::kCoordinateAxisCount,
                    0.0F),
        atom_sources(residue_count * hiko_u::kCanonicalAtomCount,
                     hiko_u::AtomSource::Missing),
        residue_codes(residue_count, 'A'),
        residues(residue_count),
        input_id(std::move(id)) {
    static constexpr const char* kNames[] = {"ALA", "CYS", "ASP", "GLU"};
    static constexpr char kCodes[] = {'A', 'C', 'D', 'E'};
    const char* chain_id = chain == 'B' ? "B" : "A";
    const char* model_id = model == 2 ? "2" : "1";
    for (std::size_t i = 0; i < residue_count; ++i) {
      residue_codes[i] = kCodes[i % 4];
      residues[i] = {residue_codes[i],
                     kNames[i % 4],
                     chain_id,
                     model_id,
                     model,
                     static_cast<std::int32_t>(10 + i),
                     '\0',
                     input_id,
                     static_cast<std::int64_t>(i),
                     input_id,
                     static_cast<std::int64_t>(i + 1)};
    }
  }

  void set_atom(std::size_t residue,
                hiko_u::CanonicalAtom atom,
                double x,
                double y,
                double z,
                hiko_u::AtomSource source = hiko_u::AtomSource::Observed) {
    const std::size_t atom_index = static_cast<std::size_t>(atom);
    const std::size_t source_offset =
        residue * hiko_u::kCanonicalAtomCount + atom_index;
    atom_sources[source_offset] = source;
    const std::size_t coord_offset =
        source_offset * hiko_u::kCoordinateAxisCount;
    coordinates[coord_offset] = static_cast<float>(x);
    coordinates[coord_offset + 1] = static_cast<float>(y);
    coordinates[coord_offset + 2] = static_cast<float>(z);
  }

  hiko_u::StructureView view() const {
    return {residue_codes.size(),
            {coordinates.data(), coordinates.size()},
            {atom_sources.data(), atom_sources.size()},
            {residue_codes.data(), residue_codes.size()},
            {residues.data(), residues.size()},
            input_id,
            input_id,
            {nullptr, 0}};
  }
};

hiko::AlignmentPath diagonal_path(std::size_t count) {
  hiko::AlignmentPath path{};
  for (std::size_t i = 0; i < count; ++i) {
    path.steps.push_back({static_cast<std::int32_t>(i),
                          static_cast<std::int32_t>(i),
                          1.0F});
  }
  path.aligned_pairs = count;
  return path;
}

hiko_u::MetricValue valid(double value) {
  return {value, true, hiko_u::MetricInvalidReason::None};
}

hiko_u::MetricValue invalid(hiko_u::MetricInvalidReason reason) {
  return {0.0, false, reason};
}

hiko::PairwiseResult pairwise_result(std::size_t count) {
  hiko::PairwiseResult result{};
  result.path = diagonal_path(count);
  result.metrics.raw_sw_score = 42.5;
  result.metrics.coverage_query = valid(1.0);
  result.metrics.coverage_target = valid(1.0);
  result.metrics.coverage_mean = valid(1.0);
  result.metrics.identity =
      invalid(hiko_u::MetricInvalidReason::MissingSequenceMetadata);
  result.metrics.rmsd = valid(0.0);
  result.metrics.tm_score_query = valid(1.0);
  result.metrics.tm_score_target = valid(1.0);
  result.metrics.lddt = invalid(hiko_u::MetricInvalidReason::Unimplemented);
  result.metrics.lddt_byA = invalid(hiko_u::MetricInvalidReason::Unimplemented);
  result.metrics.lddt_byB = invalid(hiko_u::MetricInvalidReason::Unimplemented);
  result.metrics.lddt_aln = invalid(hiko_u::MetricInvalidReason::Unimplemented);
  result.metrics.coverage_byA =
      invalid(hiko_u::MetricInvalidReason::Unimplemented);
  result.metrics.coverage_byB =
      invalid(hiko_u::MetricInvalidReason::Unimplemented);
  result.metrics.ecs = invalid(hiko_u::MetricInvalidReason::Unimplemented);
  return result;
}

void populate_translated_ca_pair(StructureFixture& query,
                                 StructureFixture& target) {
  const double query_points[3][3] = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
  };
  for (std::size_t i = 0; i < 3; ++i) {
    query.set_atom(i, hiko_u::CanonicalAtom::CA, query_points[i][0],
                   query_points[i][1], query_points[i][2]);
    target.set_atom(i, hiko_u::CanonicalAtom::CA, query_points[i][0] + 10.0,
                    query_points[i][1] + 20.0, query_points[i][2] + 30.0);
  }
  query.set_atom(0, hiko_u::CanonicalAtom::N, 0.0, 0.0, 0.0);
  target.set_atom(0, hiko_u::CanonicalAtom::N, 10.0, 20.0, 30.0);
  query.set_atom(0, hiko_u::CanonicalAtom::CB, 999.0, 999.0, 999.0,
                 hiko_u::AtomSource::Missing);
}

void test_superposed_two_model_pdb_and_remarks() {
  StructureFixture query("query struct", 'A', 1, 3);
  StructureFixture target("target struct", 'B', 2, 3);
  populate_translated_ca_pair(query, target);

  pio::PdbWriterOptions options{};
  options.backend = hiko_u::Backend::Scalar;
  const auto rendered = pio::render_superposed_pdb(
      pairwise_result(3), query.view(), target.view(), options);
  if (rendered.status.code != hiko_u::StatusCode::Ok) {
    fail("superposed PDB render should succeed");
  }
  if (!rendered.value.transform_valid || rendered.value.transform_pair_count != 3) {
    fail("PDB writer must report the observed-CA transform");
  }
  if (!nearly_equal(rendered.value.target_to_query.translation[0], -10.0) ||
      !nearly_equal(rendered.value.target_to_query.translation[1], -20.0) ||
      !nearly_equal(rendered.value.target_to_query.translation[2], -30.0)) {
    fail("target-to-query translation mismatch");
  }

  const std::string& pdb = rendered.value.contents;
  if (pdb.find("REMARK HIKOBOSHI_VERSION 0.1.0") == std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_INPUT_QUERY query_struct") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_INPUT_TARGET target_struct") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_BACKEND scalar") == std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_GAP_PARAMETERS open=-1.4 extension=-0.15") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC raw_sw_score 42.5") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC identity NA reason=missing_sequence_metadata") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC lddt NA reason=unimplemented") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC lddt_byA NA reason=unimplemented") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC lddt_byB NA reason=unimplemented") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC lddt_aln NA reason=unimplemented") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC coverage_byA NA reason=unimplemented") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_METRIC coverage_byB NA reason=unimplemented") ==
          std::string::npos ||
      pdb.find("REMARK HIKOBOSHI_TRANSFORM_TRANSLATION -10.000000 -20.000000 -30.000000") ==
          std::string::npos) {
    fail("PDB REMARK content is missing chartered metadata");
  }
  if (count_occurrences(pdb, "MODEL") != 2 ||
      count_occurrences(pdb, "ENDMDL") != 2 ||
      pdb.find("END\n") == std::string::npos) {
    fail("PDB output must contain two MODEL records and END");
  }
  if (pdb.find("   0.000   0.000   0.000") == std::string::npos) {
    fail("observed zero coordinates must not be treated as missing");
  }
  if (pdb.find("999.000") != std::string::npos) {
    fail("missing atom slots must not be emitted from coordinate placeholders");
  }
  if (pdb.find("open=nan") != std::string::npos ||
      pdb.find("extension=nan") != std::string::npos) {
    fail("PDB gap metadata must render resolved finite gap values");
  }
}

void test_transform_invalid_reason_is_surfaced() {
  StructureFixture query("query", 'A', 1, 2);
  StructureFixture target("target", 'A', 1, 2);
  query.set_atom(0, hiko_u::CanonicalAtom::CA, 0.0, 0.0, 0.0);
  query.set_atom(1, hiko_u::CanonicalAtom::CA, 1.0, 0.0, 0.0);
  target.set_atom(0, hiko_u::CanonicalAtom::CA, 10.0, 0.0, 0.0);
  target.set_atom(1, hiko_u::CanonicalAtom::CA, 11.0, 0.0, 0.0);

  const auto rendered = pio::render_superposed_pdb(
      pairwise_result(2), query.view(), target.view());
  if (rendered.status.code != hiko_u::StatusCode::Unavailable ||
      rendered.value.transform_invalid_reason !=
          hiko_u::MetricInvalidReason::InsufficientAlignedPairs) {
    fail("invalid superposition must surface the metric invalid reason");
  }
}

void test_package_default_gap_sentinel_does_not_leak_to_remarks() {
  StructureFixture query("query", 'A', 1, 3);
  StructureFixture target("target", 'A', 1, 3);
  populate_translated_ca_pair(query, target);

  pio::PdbWriterOptions options{};
  options.alignment = {};
  const auto rendered = pio::render_superposed_pdb(
      pairwise_result(3), query.view(), target.view(), options);
  if (rendered.status.code != hiko_u::StatusCode::Ok) {
    fail("PDB render with package-default gap sentinels should succeed");
  }
  const std::string& pdb = rendered.value.contents;
  if (pdb.find("REMARK HIKOBOSHI_GAP_PARAMETERS open=-1.4 extension=-0.15") ==
      std::string::npos) {
    fail("PDB writer must resolve package-default gap sentinels to hard gaps");
  }
  if (pdb.find("open=nan") != std::string::npos ||
      pdb.find("extension=nan") != std::string::npos) {
    fail("PDB writer must not emit package-default gap sentinels as NaN");
  }
}

}  // namespace

int main() {
  test_superposed_two_model_pdb_and_remarks();
  test_transform_invalid_reason_is_surfaced();
  test_package_default_gap_sentinel_does_not_leak_to_remarks();
  return 0;
}

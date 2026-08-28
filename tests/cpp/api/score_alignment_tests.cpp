#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/api/score_alignment.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "score_alignment_tests: %s\n", message);
  std::exit(1);
}

bool metric_bit_equal(hiko_u::MetricValue a, hiko_u::MetricValue b) {
  if (a.valid != b.valid) return false;
  if (!a.valid) {
    return a.reason == b.reason;
  }
  // Bit-equal value comparison: equal IEEE754 representations. Since both
  // metrics come from the same metric block over the same path/structures,
  // every numeric field must be exactly equal, not "nearly" equal.
  return a.value == b.value;
}

hiko_u::StructureView structure_view(
    const std::vector<float>& coordinates,
    const std::vector<hiko_u::AtomSource>& atom_sources,
    const std::vector<char>& residue_codes,
    const std::vector<hiko_u::ResidueMetadataView>& residues) {
  return {residue_codes.size(),
          {coordinates.data(), coordinates.size()},
          {atom_sources.data(), atom_sources.size()},
          {residue_codes.data(), residue_codes.size()},
          {residues.data(), residues.size()},
          "synthetic",
          {},
          {}};
}

std::vector<float> synthetic_coordinates() {
  return {
      0.0F, 0.0F, 0.0F,  1.0F, 0.0F, 0.0F,  2.0F, 0.0F, 0.0F,
      2.0F, 1.0F, 0.0F,  1.0F, 1.0F, 0.0F,  0.0F, 0.0F, 1.5F,
      1.0F, 0.0F, 1.5F,  2.0F, 0.0F, 1.5F,  2.0F, 1.0F, 1.5F,
      1.0F, 1.0F, 1.5F,  0.0F, 0.0F, 3.0F,  1.0F, 0.0F, 3.0F,
      2.0F, 0.0F, 3.0F,  2.0F, 1.0F, 3.0F,  1.0F, 1.0F, 3.0F,
  };
}

hiko::Engine make_default_weight_engine() {
  const auto package_result = hiko_w::default_mpnn_d64_package();
  if (package_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.descriptor == nullptr) {
    fail("default Hikoboshi-MPNN-64 package must validate");
  }
  hiko::EngineConfig config{};
  config.weights =
      package_result.value.descriptor->compatibility_views.weights;
  config.execution.backend = hiko_u::Backend::Scalar;
  config.package = package_result.value;
  return hiko::Engine(config);
}

struct Fixture {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;
  std::vector<hiko_u::ResidueMetadataView> residues;
};

Fixture three_residue_fixture() {
  Fixture out{};
  out.coordinates = synthetic_coordinates();
  out.atom_sources.assign(3 * hiko_u::kCanonicalAtomCount,
                          hiko_u::AtomSource::Observed);
  out.residue_codes = {'A', 'C', 'D'};
  out.residues = {
      {'A', "ALA", "A", "1", 1, 1, '\0', "synthetic", 0, {}, -1},
      {'C', "CYS", "A", "2", 1, 2, '\0', "synthetic", 1, {}, -1},
      {'D', "ASP", "A", "3", 1, 3, '\0', "synthetic", 2, {}, -1},
  };
  return out;
}

void assert_bit_equal_to_pairwise(const hiko::PairwiseResult& pairwise,
                                  const hiko::ScoreAlignmentResult& scored) {
  if (scored.aligned_pairs != pairwise.path.aligned_pairs) {
    fail("score_alignment aligned_pairs must match pairwise path aligned_pairs");
  }
  if (!metric_bit_equal(scored.rmsd, pairwise.metrics.rmsd) ||
      !metric_bit_equal(scored.tm_score_query,
                        pairwise.metrics.tm_score_query) ||
      !metric_bit_equal(scored.tm_score_target,
                        pairwise.metrics.tm_score_target) ||
      !metric_bit_equal(scored.lddt, pairwise.metrics.lddt) ||
      !metric_bit_equal(scored.lddt_byA, pairwise.metrics.lddt_byA) ||
      !metric_bit_equal(scored.lddt_byB, pairwise.metrics.lddt_byB) ||
      !metric_bit_equal(scored.lddt_aln, pairwise.metrics.lddt_aln) ||
      !metric_bit_equal(scored.identity, pairwise.metrics.identity) ||
      !metric_bit_equal(scored.coverage_query,
                        pairwise.metrics.coverage_query) ||
      !metric_bit_equal(scored.coverage_target,
                        pairwise.metrics.coverage_target) ||
      !metric_bit_equal(scored.coverage_mean,
                        pairwise.metrics.coverage_mean) ||
      !metric_bit_equal(scored.coverage_byA,
                        pairwise.metrics.coverage_byA) ||
      !metric_bit_equal(scored.coverage_byB,
                        pairwise.metrics.coverage_byB) ||
      !metric_bit_equal(scored.ecs, pairwise.metrics.ecs)) {
    fail("score_alignment metrics must be bit-equal to pairwise metrics");
  }
}

void test_bit_equality_with_pairwise() {
  const Fixture fixture = three_residue_fixture();
  const hiko_u::StructureView view = structure_view(
      fixture.coordinates, fixture.atom_sources, fixture.residue_codes,
      fixture.residues);

  const hiko::Engine engine = make_default_weight_engine();
  hiko::PairwiseStructureRequest pairwise_request{view, view};
  pairwise_request.mode = hiko::AlignmentMode::Hard;
  const auto pairwise = engine.pairwise(pairwise_request);
  if (pairwise.status.code != hiko_u::StatusCode::Ok) {
    fail("pairwise must return Ok before score_alignment regression check");
  }

  hiko::ScoreAlignmentRequest request{};
  request.query_structure = view;
  request.target_structure = view;
  request.correspondences = pairwise.value.path;
  hiko::ScoreAlignmentResult result{};
  const hiko_u::Status status = hiko::score_alignment(request, result);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("score_alignment must return Ok for the pairwise path");
  }
  assert_bit_equal_to_pairwise(pairwise.value, result);
}

void test_diagonal_path_matches_pairwise() {
  const Fixture fixture = three_residue_fixture();
  const hiko_u::StructureView view = structure_view(
      fixture.coordinates, fixture.atom_sources, fixture.residue_codes,
      fixture.residues);

  const hiko::Engine engine = make_default_weight_engine();
  hiko::PairwiseStructureRequest pairwise_request{view, view};
  pairwise_request.mode = hiko::AlignmentMode::Hard;
  const auto pairwise = engine.pairwise(pairwise_request);
  if (pairwise.status.code != hiko_u::StatusCode::Ok) {
    fail("pairwise must return Ok for self-self structure");
  }

  // Externally constructed diagonal path: every residue aligns to its
  // counterpart in the (same) target structure. residue_score is zeroed
  // because score_alignment does not run SW.
  hiko_u::AlignmentPath diagonal{};
  for (std::size_t i = 0; i < view.residue_count; ++i) {
    diagonal.steps.push_back(
        {static_cast<std::int32_t>(i), static_cast<std::int32_t>(i), 0.0F});
  }
  diagonal.aligned_pairs = view.residue_count;
  diagonal.query_start = 0;
  diagonal.query_end = static_cast<std::int32_t>(view.residue_count - 1);
  diagonal.target_start = 0;
  diagonal.target_end = diagonal.query_end;

  hiko::ScoreAlignmentRequest request{};
  request.query_structure = view;
  request.target_structure = view;
  request.correspondences = diagonal;
  hiko::ScoreAlignmentResult result{};
  if (hiko::score_alignment(request, result).code != hiko_u::StatusCode::Ok) {
    fail("score_alignment must accept a manually constructed diagonal path");
  }
  // Self-self alignment over the synthetic 3-residue fixture is what the
  // pairwise hard-SW pass also recovers; the panels must match bit-equally.
  assert_bit_equal_to_pairwise(pairwise.value, result);
}

void test_empty_correspondences_return_invalid_metrics() {
  const Fixture fixture = three_residue_fixture();
  const hiko_u::StructureView view = structure_view(
      fixture.coordinates, fixture.atom_sources, fixture.residue_codes,
      fixture.residues);

  hiko::ScoreAlignmentRequest request{};
  request.query_structure = view;
  request.target_structure = view;
  hiko::ScoreAlignmentResult result{};
  if (hiko::score_alignment(request, result).code != hiko_u::StatusCode::Ok) {
    fail("score_alignment with empty correspondences must succeed");
  }
  if (result.aligned_pairs != 0) {
    fail("empty correspondences must report aligned_pairs == 0");
  }
  // Kabsch needs >= 3 aligned pairs, so RMSD and the Kabsch-driven TM-scores
  // surface InsufficientAlignedPairs when the path is empty. Canonical
  // (Mariani) lDDT and structure-length coverage are defined as 0 over an
  // empty path with non-zero reference graphs, so they remain valid; that
  // is the same state pairwise() would produce when SW finds no aligned
  // pairs.
  if (result.rmsd.valid ||
      result.rmsd.reason !=
          hiko_u::MetricInvalidReason::InsufficientAlignedPairs) {
    fail("empty correspondences must surface InsufficientAlignedPairs for rmsd");
  }
  if (result.tm_score_query.valid || result.tm_score_target.valid) {
    fail("empty correspondences must invalidate the Kabsch-derived TM-scores");
  }
}

void test_single_pair_below_kabsch_threshold() {
  const Fixture fixture = three_residue_fixture();
  const hiko_u::StructureView view = structure_view(
      fixture.coordinates, fixture.atom_sources, fixture.residue_codes,
      fixture.residues);

  hiko_u::AlignmentPath single{};
  single.steps.push_back({0, 0, 0.0F});
  single.aligned_pairs = 1;
  single.query_start = 0;
  single.query_end = 0;
  single.target_start = 0;
  single.target_end = 0;

  hiko::ScoreAlignmentRequest request{};
  request.query_structure = view;
  request.target_structure = view;
  request.correspondences = single;
  hiko::ScoreAlignmentResult result{};
  if (hiko::score_alignment(request, result).code != hiko_u::StatusCode::Ok) {
    fail("score_alignment must accept single-pair correspondences");
  }
  if (result.aligned_pairs != 1) {
    fail("single-pair correspondences must report aligned_pairs == 1");
  }
  if (result.rmsd.valid) {
    fail("single-pair RMSD must be invalid (Kabsch needs >= 3 pairs)");
  }
  if (!result.coverage_query.valid || !result.identity.valid) {
    fail("single-pair must still report valid coverage and identity");
  }
}

void test_out_of_range_index_rejected() {
  const Fixture fixture = three_residue_fixture();
  const hiko_u::StructureView view = structure_view(
      fixture.coordinates, fixture.atom_sources, fixture.residue_codes,
      fixture.residues);

  hiko_u::AlignmentPath bad{};
  bad.steps.push_back({0, 5, 0.0F});  // target_index 5 out of range for L=3
  bad.aligned_pairs = 1;

  hiko::ScoreAlignmentRequest request{};
  request.query_structure = view;
  request.target_structure = view;
  request.correspondences = bad;
  hiko::ScoreAlignmentResult result{};
  const hiko_u::Status status = hiko::score_alignment(request, result);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("out-of-range correspondence index must return InvalidArgument");
  }
}

void test_empty_structure_rejected() {
  hiko::ScoreAlignmentRequest request{};
  hiko::ScoreAlignmentResult result{};
  const hiko_u::Status status = hiko::score_alignment(request, result);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("empty structure must return InvalidArgument");
  }
}

}  // namespace

int main() {
  test_bit_equality_with_pairwise();
  test_diagonal_path_matches_pairwise();
  test_empty_correspondences_return_invalid_metrics();
  test_single_pair_below_kabsch_threshold();
  test_out_of_range_index_rejected();
  test_empty_structure_rejected();
  return 0;
}

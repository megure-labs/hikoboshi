#include <hikoboshi/algorithms/metrics.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "metrics_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double a, double b, double tolerance = 1.0e-6) {
  return std::fabs(a - b) <= tolerance;
}

hiko_u::AlignmentPath diagonal_path(std::size_t count) {
  hiko_u::AlignmentPath path{};
  for (std::size_t i = 0; i < count; ++i) {
    path.steps.push_back({static_cast<std::int32_t>(i),
                          static_cast<std::int32_t>(i),
                          1.0F});
  }
  path.aligned_pairs = count;
  path.query_start = count == 0 ? hiko_u::kAlignmentGapSentinel : 0;
  path.target_start = count == 0 ? hiko_u::kAlignmentGapSentinel : 0;
  path.query_end = count == 0 ? hiko_u::kAlignmentGapSentinel
                              : static_cast<std::int32_t>(count - 1);
  path.target_end = path.query_end;
  return path;
}

struct StructureFixture {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;

  explicit StructureFixture(std::size_t residue_count)
      : coordinates(residue_count * hiko_u::kCanonicalAtomCount *
                        hiko_u::kCoordinateAxisCount,
                    0.0F),
        atom_sources(residue_count * hiko_u::kCanonicalAtomCount,
                     hiko_u::AtomSource::Missing),
        residue_codes(residue_count, 'A') {}

  void set_ca(std::size_t residue, hiko::Point3 point) {
    const std::size_t atom = static_cast<std::size_t>(hiko_u::CanonicalAtom::CA);
    const std::size_t source_offset = residue * hiko_u::kCanonicalAtomCount + atom;
    atom_sources[source_offset] = hiko_u::AtomSource::Observed;
    const std::size_t coord_offset =
        source_offset * hiko_u::kCoordinateAxisCount;
    coordinates[coord_offset] = static_cast<float>(point.x);
    coordinates[coord_offset + 1] = static_cast<float>(point.y);
    coordinates[coord_offset + 2] = static_cast<float>(point.z);
  }

  hiko_u::StructureView view() const {
    return {residue_codes.size(),
            {coordinates.data(), coordinates.size()},
            {atom_sources.data(), atom_sources.size()},
            {residue_codes.data(), residue_codes.size()},
            {nullptr, 0},
            {},
            {},
            {}};
  }
};

void test_coverage_formulas() {
  const hiko_u::AlignmentPath path = diagonal_path(2);
  const hiko::CoverageMetrics coverage = hiko::compute_coverage(path, 4, 5);
  if (!coverage.query.valid || !coverage.target.valid || !coverage.mean.valid) {
    fail("coverage metrics must be valid for non-zero lengths");
  }
  if (!nearly_equal(coverage.query.value, 0.5) ||
      !nearly_equal(coverage.target.value, 0.4) ||
      !nearly_equal(coverage.mean.value, 4.0 / 9.0)) {
    fail("coverage formulas must match METRICS_CHARTER");
  }
}

void test_identity_formula_and_x_exclusion() {
  hiko_u::AlignmentPath path = diagonal_path(4);
  const std::vector<char> query = {'A', 'C', 'X', 'D'};
  const std::vector<char> target = {'A', 'T', 'Y', 'D'};
  const hiko_u::MetricValue identity =
      hiko::compute_identity(path, {query.data(), query.size()},
                            {target.data(), target.size()});
  if (!identity.valid || !nearly_equal(identity.value, 2.0 / 3.0)) {
    fail("identity must exclude X and count exact standard-code matches");
  }
}

void test_identity_invalid_reasons() {
  const hiko_u::AlignmentPath path = diagonal_path(1);
  const hiko_u::MetricValue missing =
      hiko::compute_identity(path, {nullptr, 0}, {nullptr, 0});
  if (missing.valid ||
      missing.reason != hiko_u::MetricInvalidReason::MissingSequenceMetadata) {
    fail("missing sequence metadata must invalidate identity explicitly");
  }
  const std::vector<char> unknown = {'X'};
  const hiko_u::MetricValue zero =
      hiko::compute_identity(path, {unknown.data(), unknown.size()},
                            {unknown.data(), unknown.size()});
  if (zero.valid || zero.reason != hiko_u::MetricInvalidReason::ZeroDenominator) {
    fail("identity denominator zero must not return a fabricated zero");
  }
}

void test_kabsch_alignment_and_rmsd() {
  const std::vector<hiko::Point3> query = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
  };
  const std::vector<hiko::Point3> target = {
      {5.0, -2.0, 1.0},
      {5.0, -1.0, 1.0},
      {4.0, -2.0, 1.0},
  };
  const hiko::KabschResult kabsch =
      hiko::kabsch_superpose({query.data(), query.size()},
                            {target.data(), target.size()});
  if (!kabsch.valid || !nearly_equal(kabsch.rmsd, 0.0, 1.0e-5)) {
    fail("Kabsch must superpose a rigidly transformed triangle");
  }

  StructureFixture query_structure(3);
  StructureFixture target_structure(3);
  for (std::size_t i = 0; i < query.size(); ++i) {
    query_structure.set_ca(i, query[i]);
    target_structure.set_ca(i, target[i]);
  }
  const hiko_u::MetricValue rmsd =
      hiko::compute_rmsd(diagonal_path(3), query_structure.view(),
                        target_structure.view());
  if (!rmsd.valid || !nearly_equal(rmsd.value, 0.0, 1.0e-5)) {
    fail("RMSD must use Kabsch-superposed observed CA pairs");
  }
}

void test_tm_score_directional_normalization() {
  StructureFixture query_structure(3);
  StructureFixture target_structure(3);
  const hiko::Point3 points[3] = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
  };
  for (std::size_t i = 0; i < 3; ++i) {
    query_structure.set_ca(i, points[i]);
    target_structure.set_ca(i, points[i]);
  }
  const hiko::TmScoreMetrics tm =
      hiko::compute_tm_scores(diagonal_path(3), query_structure.view(),
                             target_structure.view(), 3, 6);
  if (!tm.query_norm.valid || !tm.target_norm.valid ||
      !nearly_equal(tm.query_norm.value, 1.0) ||
      !nearly_equal(tm.target_norm.value, 0.5)) {
    fail("TM-score must report directional query and target normalization");
  }
}

void check_canonical_identity(const hiko::LddtMetrics& metrics, const char* label) {
  if (!metrics.lddt_aln.valid) {
    return;
  }
  if (metrics.lddt_byA.valid && metrics.coverage_byA.valid) {
    const double expected =
        metrics.lddt_aln.value * metrics.coverage_byA.value;
    if (std::fabs(metrics.lddt_byA.value - expected) > 1.0e-9) {
      std::fprintf(stderr,
                   "metrics_tests: %s identity byA failed: %.17g != %.17g\n",
                   label, metrics.lddt_byA.value, expected);
      fail("canonical identity lddt_byA = lddt_aln * coverage_byA must hold");
    }
  }
  if (metrics.lddt_byB.valid && metrics.coverage_byB.valid) {
    const double expected =
        metrics.lddt_aln.value * metrics.coverage_byB.value;
    if (std::fabs(metrics.lddt_byB.value - expected) > 1.0e-9) {
      std::fprintf(stderr,
                   "metrics_tests: %s identity byB failed: %.17g != %.17g\n",
                   label, metrics.lddt_byB.value, expected);
      fail("canonical identity lddt_byB = lddt_aln * coverage_byB must hold");
    }
  }
}

void test_lddt_threshold_behavior() {
  // 3-residue structures, fully aligned, all pairs in R0.
  // Expected canonical lDDT (Mariani):
  //   pair distances A: 1, 2, 3 ; B: 1.6, 1.4, 3
  //   delta:           0.6, 0.6, 0
  //   threshold passes (per pair, 4 thresholds = 0.5/1/2/4):
  //     (0,1): delta 0.6 -> 3 passes (1, 2, 4)
  //     (1,2): delta 0.6 -> 3 passes
  //     (0,2): delta 0   -> 4 passes
  //   total 10 passes; 3 pairs * 4 thresholds = 12
  //   lddt_byA = lddt_byB = 10/12 ; lddt_aln = 10/12 ; coverage_byA = byB = 1.0
  StructureFixture query_structure(3);
  StructureFixture target_structure(3);
  query_structure.set_ca(0, {0.0, 0.0, 0.0});
  query_structure.set_ca(1, {1.0, 0.0, 0.0});
  query_structure.set_ca(2, {3.0, 0.0, 0.0});
  target_structure.set_ca(0, {0.0, 0.0, 0.0});
  target_structure.set_ca(1, {1.6, 0.0, 0.0});
  target_structure.set_ca(2, {3.0, 0.0, 0.0});

  const hiko::LddtMetrics metrics =
      hiko::compute_lddt(diagonal_path(3), query_structure.view(),
                        target_structure.view());
  if (!metrics.lddt.valid || !metrics.lddt_byA.valid ||
      !metrics.lddt_byB.valid || !metrics.lddt_aln.valid ||
      !metrics.coverage_byA.valid || !metrics.coverage_byB.valid) {
    fail("canonical lDDT must report all six fields valid for fully aligned input");
  }
  const double expected = 10.0 / 12.0;
  if (!nearly_equal(metrics.lddt.value, expected) ||
      !nearly_equal(metrics.lddt_byA.value, expected) ||
      !nearly_equal(metrics.lddt_byB.value, expected) ||
      !nearly_equal(metrics.lddt_aln.value, expected)) {
    fail("canonical lDDT must equal hand-computed 10/12 on the threshold fixture");
  }
  if (!nearly_equal(metrics.coverage_byA.value, 1.0) ||
      !nearly_equal(metrics.coverage_byB.value, 1.0)) {
    fail("coverage must be 1.0 when every reference pair is aligned-aligned");
  }
  check_canonical_identity(metrics, "test_lddt_threshold_behavior");
}

void test_lddt_canonical_identical_structures() {
  // Hand-computed regression fixture #1: 4-residue identical structures.
  // Every pair distance matches exactly, so every threshold passes; both
  // directions and lddt_aln must be 1.0 with full coverage.
  StructureFixture query_structure(4);
  StructureFixture target_structure(4);
  for (std::size_t i = 0; i < 4; ++i) {
    const hiko::Point3 point = {static_cast<double>(i), 0.0, 0.0};
    query_structure.set_ca(i, point);
    target_structure.set_ca(i, point);
  }

  const hiko::LddtMetrics metrics =
      hiko::compute_lddt(diagonal_path(4), query_structure.view(),
                        target_structure.view());
  if (!metrics.lddt.valid || !metrics.lddt_byA.valid ||
      !metrics.lddt_byB.valid || !metrics.lddt_aln.valid) {
    fail("identical structures must produce a fully valid canonical lDDT");
  }
  if (!nearly_equal(metrics.lddt.value, 1.0) ||
      !nearly_equal(metrics.lddt_byA.value, 1.0) ||
      !nearly_equal(metrics.lddt_byB.value, 1.0) ||
      !nearly_equal(metrics.lddt_aln.value, 1.0) ||
      !nearly_equal(metrics.coverage_byA.value, 1.0) ||
      !nearly_equal(metrics.coverage_byB.value, 1.0)) {
    fail("identical 4-residue structures must score 1.0 in every lDDT field");
  }
  check_canonical_identity(metrics, "test_lddt_canonical_identical_structures");
}

void test_lddt_canonical_perturbed_residue() {
  // Hand-computed regression fixture #2: one residue displaced by 2.0 Å in B.
  //   A: linear chain (0,1,2,3) along x.
  //   B: same chain but residue 1 raised by 2.0 along y.
  //   d_A pairs:  (0,1)=1   (1,2)=1   (1,3)=2   (0,2)=2  (0,3)=3  (2,3)=1
  //   d_B pairs:  (0,1)=√5  (1,2)=√5  (1,3)=2√2 (0,2)=2  (0,3)=3  (2,3)=1
  //   |delta|:    1.2360    1.2360    0.8284    0        0        0
  //   passes (thresholds 0.5/1/2/4):
  //     (0,1) 1.2360 -> 2 passes (2,4)
  //     (1,2) 1.2360 -> 2 passes
  //     (1,3) 0.8284 -> 3 passes (1,2,4)
  //     (0,2) 0      -> 4 passes
  //     (0,3) 0      -> 4 passes
  //     (2,3) 0      -> 4 passes
  //   total 19 passes ; 6 pairs * 4 = 24 ; lddt = 19/24.
  StructureFixture query_structure(4);
  StructureFixture target_structure(4);
  query_structure.set_ca(0, {0.0, 0.0, 0.0});
  query_structure.set_ca(1, {1.0, 0.0, 0.0});
  query_structure.set_ca(2, {2.0, 0.0, 0.0});
  query_structure.set_ca(3, {3.0, 0.0, 0.0});
  target_structure.set_ca(0, {0.0, 0.0, 0.0});
  target_structure.set_ca(1, {1.0, 2.0, 0.0});
  target_structure.set_ca(2, {2.0, 0.0, 0.0});
  target_structure.set_ca(3, {3.0, 0.0, 0.0});

  const hiko::LddtMetrics metrics =
      hiko::compute_lddt(diagonal_path(4), query_structure.view(),
                        target_structure.view());
  const double expected = 19.0 / 24.0;
  if (!metrics.lddt.valid || !nearly_equal(metrics.lddt.value, expected) ||
      !nearly_equal(metrics.lddt_byA.value, expected) ||
      !nearly_equal(metrics.lddt_byB.value, expected) ||
      !nearly_equal(metrics.lddt_aln.value, expected)) {
    fail("perturbed-residue lDDT must equal 19/24 across all canonical fields");
  }
  if (!nearly_equal(metrics.coverage_byA.value, 1.0) ||
      !nearly_equal(metrics.coverage_byB.value, 1.0)) {
    fail("perturbed-residue fixture has full alignment coverage in both directions");
  }
  check_canonical_identity(metrics, "test_lddt_canonical_perturbed_residue");
}

void test_lddt_canonical_partial_coverage() {
  // Hand-computed regression fixture #3: identical 4-residue structures but
  // only the first three residues are aligned. Residue 3 is unaligned in both
  // structures, so its contributions show up only in the canonical denominator,
  // never in the aligned-only score.
  //   total pairs in R0_A = 6 (and = 6 in R0_B by symmetry)
  //   aligned-aligned pairs in R0_A = 3 (the (0,1), (0,2), (1,2) triangle)
  //   pass count = 3 pairs * 4 passes = 12 (identical pair distances)
  //   lddt_byA = lddt_byB = 12 / (6 * 4) = 0.5
  //   lddt_aln = 12 / (3 * 4) = 1.0 ; coverage_byA = coverage_byB = 0.5
  StructureFixture query_structure(4);
  StructureFixture target_structure(4);
  for (std::size_t i = 0; i < 4; ++i) {
    const hiko::Point3 point = {static_cast<double>(i), 0.0, 0.0};
    query_structure.set_ca(i, point);
    target_structure.set_ca(i, point);
  }

  const hiko::LddtMetrics metrics =
      hiko::compute_lddt(diagonal_path(3), query_structure.view(),
                        target_structure.view());
  if (!nearly_equal(metrics.lddt.value, 0.5) ||
      !nearly_equal(metrics.lddt_byA.value, 0.5) ||
      !nearly_equal(metrics.lddt_byB.value, 0.5) ||
      !nearly_equal(metrics.lddt_aln.value, 1.0) ||
      !nearly_equal(metrics.coverage_byA.value, 0.5) ||
      !nearly_equal(metrics.coverage_byB.value, 0.5)) {
    fail("partial-coverage canonical lDDT must match hand-computed expectations");
  }
  check_canonical_identity(metrics, "test_lddt_canonical_partial_coverage");
}

void test_lddt_canonical_asymmetric_coverage() {
  // Regression fixture #4: structures with different residue counts.
  //   A has 5 residues; B has 4 residues.
  //   Path aligns A[0..3] to B[0..3]; A residue 4 is unaligned.
  //   total pairs in R0_A = 10 ; aligned-aligned in R0_A = 6 ; coverage_byA = 0.6
  //   total pairs in R0_B =  6 ; aligned-aligned in R0_B = 6 ; coverage_byB = 1.0
  //   identical aligned coordinates -> 4 passes per aligned-aligned pair -> 24 passes
  //   lddt_byA = 24/40 = 0.6 ; lddt_byB = 24/24 = 1.0 ; lddt = 0.8 ; lddt_aln = 1.0
  StructureFixture query_structure(5);
  StructureFixture target_structure(4);
  for (std::size_t i = 0; i < 4; ++i) {
    const hiko::Point3 point = {static_cast<double>(i), 0.0, 0.0};
    query_structure.set_ca(i, point);
    target_structure.set_ca(i, point);
  }
  query_structure.set_ca(4, {4.0, 0.0, 0.0});

  const hiko::LddtMetrics metrics =
      hiko::compute_lddt(diagonal_path(4), query_structure.view(),
                        target_structure.view());
  if (!nearly_equal(metrics.lddt_byA.value, 0.6) ||
      !nearly_equal(metrics.lddt_byB.value, 1.0) ||
      !nearly_equal(metrics.lddt.value, 0.8) ||
      !nearly_equal(metrics.lddt_aln.value, 1.0) ||
      !nearly_equal(metrics.coverage_byA.value, 0.6) ||
      !nearly_equal(metrics.coverage_byB.value, 1.0)) {
    fail("asymmetric-coverage canonical lDDT must match hand-computed expectations");
  }
  check_canonical_identity(metrics, "test_lddt_canonical_asymmetric_coverage");
}

void test_structural_invalid_reasons() {
  const hiko_u::MetricValue missing =
      hiko::compute_rmsd(diagonal_path(3), {}, {});
  if (missing.valid ||
      missing.reason != hiko_u::MetricInvalidReason::MissingStructureMetadata) {
    fail("missing coordinates must invalidate structural metrics explicitly");
  }

  StructureFixture query_structure(2);
  StructureFixture target_structure(2);
  query_structure.set_ca(0, {0.0, 0.0, 0.0});
  query_structure.set_ca(1, {1.0, 0.0, 0.0});
  target_structure.set_ca(0, {0.0, 0.0, 0.0});
  target_structure.set_ca(1, {1.0, 0.0, 0.0});

  const hiko_u::MetricValue rmsd =
      hiko::compute_rmsd(diagonal_path(2), query_structure.view(),
                        target_structure.view());
  if (rmsd.valid ||
      rmsd.reason != hiko_u::MetricInvalidReason::InsufficientAlignedPairs) {
    fail("RMSD fewer than three observed CA pairs must be invalid");
  }

  // Single-residue structures have no residue pairs in any reference at all,
  // so every canonical lDDT field becomes InsufficientAlignedPairs.
  StructureFixture single_query(1);
  StructureFixture single_target(1);
  single_query.set_ca(0, {0.0, 0.0, 0.0});
  single_target.set_ca(0, {0.0, 0.0, 0.0});
  const hiko::LddtMetrics empty =
      hiko::compute_lddt(diagonal_path(1), single_query.view(),
                        single_target.view());
  if (empty.lddt.valid || empty.lddt_byA.valid || empty.lddt_byB.valid ||
      empty.lddt_aln.valid || empty.coverage_byA.valid ||
      empty.coverage_byB.valid ||
      empty.lddt.reason != hiko_u::MetricInvalidReason::InsufficientAlignedPairs) {
    fail("lDDT with no residue pairs in either reference must be invalid");
  }

  // Aligned-aligned subset is empty (only one residue aligned out of two), but
  // both structures still contain one R0 pair, so the directional lDDTs report
  // a valid 0.0 score while lddt_aln stays InsufficientAlignedPairs.
  const hiko::LddtMetrics partial =
      hiko::compute_lddt(diagonal_path(1), query_structure.view(),
                        target_structure.view());
  if (!partial.lddt.valid ||
      !nearly_equal(partial.lddt.value, 0.0) ||
      !partial.coverage_byA.valid ||
      !nearly_equal(partial.coverage_byA.value, 0.0) ||
      partial.lddt_aln.valid ||
      partial.lddt_aln.reason !=
          hiko_u::MetricInvalidReason::InsufficientAlignedPairs) {
    fail("lDDT with no aligned-aligned pairs must invalidate lddt_aln only");
  }

  // Missing structure data short-circuits to MissingStructureMetadata across
  // every canonical field.
  const hiko::LddtMetrics missing_struct =
      hiko::compute_lddt(diagonal_path(3), {}, {});
  if (missing_struct.lddt.valid || missing_struct.lddt_byA.valid ||
      missing_struct.lddt_byB.valid || missing_struct.lddt_aln.valid ||
      missing_struct.coverage_byA.valid || missing_struct.coverage_byB.valid ||
      missing_struct.lddt.reason !=
          hiko_u::MetricInvalidReason::MissingStructureMetadata) {
    fail("missing coordinates must invalidate every canonical lDDT field");
  }
}

}  // namespace

int main() {
  test_coverage_formulas();
  test_identity_formula_and_x_exclusion();
  test_identity_invalid_reasons();
  test_kabsch_alignment_and_rmsd();
  test_tm_score_directional_normalization();
  test_lddt_threshold_behavior();
  test_lddt_canonical_identical_structures();
  test_lddt_canonical_perturbed_residue();
  test_lddt_canonical_partial_coverage();
  test_lddt_canonical_asymmetric_coverage();
  test_structural_invalid_reasons();
  return 0;
}

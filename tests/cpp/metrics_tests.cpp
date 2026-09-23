#include <hikoboshi/algorithms/metrics.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

void require_same_metric(hiko_u::MetricValue a, hiko_u::MetricValue b) {
  if (a.valid != b.valid || a.reason != b.reason ||
      (a.valid && a.value != b.value))
    fail("shared superposition must retain exact standalone metric values");
}

void test_shared_superposition() {
  StructureFixture query(19), target(19);
  for (std::size_t i = 0; i < 19; ++i) {
    query.set_ca(i, {double(i), double(i % 3), double(i % 5)});
    target.set_ca(i, {double(i) + 2.5, double(i % 4), double(i % 6)});
  }
  for (const std::size_t aligned : {0U, 1U, 2U, 3U, 19U}) {
    auto path = diagonal_path(aligned);
    path.steps.push_back({-1, 3, 1.0F});
    path.steps.push_back({99, 99, 1.0F});
    for (const std::size_t qlen : {0U, 19U, 50U}) {
      for (const std::size_t tlen : {0U, 19U, 71U}) {
        auto shared = hiko::compute_superposition_metrics(
            path, query.view(), target.view(), qlen, tlen);
        auto tm = hiko::compute_tm_scores(path, query.view(), target.view(), qlen, tlen);
        require_same_metric(shared.rmsd,
            hiko::compute_rmsd(path, query.view(), target.view()));
        require_same_metric(shared.tm.query_norm, tm.query_norm);
        require_same_metric(shared.tm.target_norm, tm.target_norm);
      }
    }
  }
  const auto missing = hiko::compute_superposition_metrics(diagonal_path(3), {}, {}, 0, 0);
  if (missing.rmsd.reason != hiko_u::MetricInvalidReason::MissingStructureMetadata ||
      missing.tm.query_norm.reason != missing.rmsd.reason ||
      missing.tm.target_norm.reason != missing.rmsd.reason)
    fail("missing metadata must precede zero TM denominator");
}

void test_lddt_reuse_with_mutable_inputs() {
  StructureFixture query(3), target(3);
  for (std::size_t i = 0; i < 3; ++i) {
    const double x = i == 2 ? 3.0 : double(i);
    query.set_ca(i, {x, 0, 0});
    target.set_ca(i, {x, 0, 0});
  }
  const auto path = diagonal_path(3);
  // Exercise repeated contents, FIFO eviction and changes at the same address.
  // Two deltas are |x-1| and the endpoint delta is zero.
  for (unsigned pass = 0; pass < 3; ++pass) {
    for (unsigned step = 0; step < 25; ++step) {
      const float x = static_cast<float>(step) / 8.0F;
      target.set_ca(1, {x, 0, 0});
      unsigned passes = 0;
      for (const double threshold : {0.5, 1.0, 2.0, 4.0})
        if (std::fabs(double(x) - 1.0) <= threshold) ++passes;
      const double expected = double(4 + 2 * passes) / 12.0;
      for (unsigned repeat = 0; repeat < 3; ++repeat) {
        auto result = hiko::compute_lddt(path, query.view(), target.view());
        if (!result.lddt.valid || result.lddt.value != expected)
          fail("lDDT reuse must validate coordinates, including reused addresses");
      }
    }
  }
  target.set_ca(1, {1.6, 0, 0});
  const auto narrow = hiko::compute_lddt(path, query.view(), target.view(), 1.0);
  if (!narrow.lddt_byA.valid || narrow.lddt_byA.value != 0.75 || narrow.lddt_byB.valid)
    fail("lDDT contacts must honor changed radius and directional denominators");
  const auto wide = hiko::compute_lddt(path, query.view(), target.view(), 15.0);
  if (wide.lddt.value != 10.0 / 12.0) fail("lDDT radius changes must invalidate reuse");
  target.atom_sources[hiko_u::kCanonicalAtomCount +
                      static_cast<std::size_t>(hiko_u::CanonicalAtom::CA)] =
      hiko_u::AtomSource::Missing;
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    const auto partial = hiko::compute_lddt(path, query.view(), target.view());
    if (!partial.lddt.valid || partial.lddt_byA.value != 1.0 / 3.0 ||
        partial.lddt_byB.value != 1.0 || partial.lddt_aln.value != 1.0)
      fail("observed-mask changes must invalidate cached reference contacts");
  }
  const auto short_path = hiko::compute_lddt(diagonal_path(1), query.view(), target.view());
  if (!short_path.lddt.valid || short_path.lddt.value != 0 || short_path.lddt_aln.valid)
    fail("contact reuse must rebuild alignment-dependent counters each call");
  for (const double radius : {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    const auto nonfinite = hiko::compute_lddt(path, query.view(), target.view(), radius);
    if (!nonfinite.lddt.valid || nonfinite.lddt_byA.value != 1.0 / 3.0)
      fail("nonfinite radius must retain the direct traversal's behavior");
  }
}

void test_lddt_bounded_contact_fallback() {
  // 257 coincident residues exceed the bounded contact count. The fallback
  // must score every pair, including after mutation and repeated calls.
  StructureFixture query(257), target(257);
  for (std::size_t i = 0; i < 257; ++i) {
    query.set_ca(i, {0, 0, 0});
    target.set_ca(i, {0, 0, 0});
  }
  const auto path = diagonal_path(257);
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    auto result = hiko::compute_lddt(path, query.view(), target.view());
    if (!result.lddt.valid || result.lddt.value != 1.0)
      fail("dense lDDT fallback must retain all contacts");
  }
  target.set_ca(0, {100, 0, 0});
  auto changed = hiko::compute_lddt(path, query.view(), target.view());
  if (changed.lddt_byA.value != 255.0 / 257.0 || changed.lddt_byB.value != 1.0)
    fail("dense fallback must honor coordinate changes");
  StructureFixture long_query(4097), long_target(4097);
  for (std::size_t i = 0; i < 3; ++i) {
    long_query.set_ca(i, {double(i), 0, 0});
    long_target.set_ca(i, {double(i), 0, 0});
  }
  auto large = hiko::compute_lddt(diagonal_path(3), long_query.view(), long_target.view());
  if (!large.lddt.valid || large.lddt.value != 1.0)
    fail("long structure fallback must preserve observed-contact semantics");
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
  test_shared_superposition();
  test_lddt_reuse_with_mutable_inputs();
  test_lddt_bounded_contact_fallback();
  return 0;
}

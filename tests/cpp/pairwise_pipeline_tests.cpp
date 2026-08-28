#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "pairwise_pipeline_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double a, double b, double tolerance = 1.0e-6) {
  return std::fabs(a - b) <= tolerance;
}

hiko_u::StructureView structure_view(const std::vector<float>& coordinates,
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
      0.0F, 0.0F, 0.0F,
      1.0F, 0.0F, 0.0F,
      2.0F, 0.0F, 0.0F,
      2.0F, 1.0F, 0.0F,
      1.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.5F,
      1.0F, 0.0F, 1.5F,
      2.0F, 0.0F, 1.5F,
      2.0F, 1.0F, 1.5F,
      1.0F, 1.0F, 1.5F,
      0.0F, 0.0F, 3.0F,
      1.0F, 0.0F, 3.0F,
      2.0F, 0.0F, 3.0F,
      2.0F, 1.0F, 3.0F,
      1.0F, 1.0F, 3.0F,
  };
}

hiko::Engine make_default_weight_engine() {
  const auto package_result = hiko_w::default_mpnn_d64_package();
  if (package_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.descriptor == nullptr) {
    fail("default Hikoboshi-MPNN-64 package must validate");
  }
  const auto weights_result = hiko_w::default_mpnn_d64();
  if (weights_result.status.code != hiko_u::StatusCode::Ok) {
    fail("default Hikoboshi-MPNN-64 weights must resolve through package provider");
  }
  const hiko_u::PackageDescriptor& descriptor =
      *package_result.value.descriptor;
  if (weights_result.value.opaque != package_result.value.opaque ||
      weights_result.value.opaque !=
          descriptor.compatibility_views.weights.opaque ||
      weights_result.value.view != descriptor.compatibility_views.weights.view) {
    fail("default weights must be the compatibility view from the default package");
  }

  hiko::EngineConfig config{};
  config.weights = weights_result.value;
  config.execution.backend = hiko_u::Backend::Scalar;
  return hiko::Engine(config);
}

hiko_u::EmbeddingView encoded_embedding_view(const hiko::EncodeResult& encoded) {
  return {encoded.embedding.residue_count,
          encoded.embedding.dimension,
          {encoded.embedding.values.data(), encoded.embedding.values.size()},
          {encoded.embedding.residue_codes.data(),
           encoded.embedding.residue_codes.size()},
          {encoded.embedding.residues.data(), encoded.embedding.residues.size()}};
}

void test_structure_coords_and_embedding_paths_are_deterministic() {
  const std::vector<float> coordinates = synthetic_coordinates();
  std::vector<hiko_u::AtomSource> atom_sources(3 * hiko_u::kCanonicalAtomCount,
                                            hiko_u::AtomSource::Observed);
  const std::vector<char> residue_codes = {'A', 'C', 'D'};
  const std::vector<hiko_u::ResidueMetadataView> residues = {
      {'A', "ALA", "A", "1", 1, 1, '\0', "synthetic", 0, {}, -1},
      {'C', "CYS", "A", "2", 1, 2, '\0', "synthetic", 1, {}, -1},
      {'D', "ASP", "A", "3", 1, 3, '\0', "synthetic", 2, {}, -1},
  };
  const hiko_u::StructureView structure =
      structure_view(coordinates, atom_sources, residue_codes, residues);
  const hiko::CoordsInputView coords{
      structure.residue_count,
      structure.coordinates,
      structure.atom_sources,
      structure.residue_codes,
      structure.residues,
  };

  const hiko::Engine engine = make_default_weight_engine();

  const auto encoded = engine.encode(hiko::EncodeStructureRequest{structure});
  if (encoded.status.code != hiko_u::StatusCode::Ok) {
    fail("structure encode must return Ok with default Hikoboshi-MPNN-64 weights");
  }
  if (encoded.value.embedding.residue_count != structure.residue_count ||
      encoded.value.embedding.dimension != hiko_d::kMpnn64HiddenDimension ||
      encoded.value.embedding.values.size() !=
          structure.residue_count * hiko_d::kMpnn64HiddenDimension) {
    fail("encoded embedding shape mismatch");
  }

  const hiko_u::EmbeddingView embedding = encoded_embedding_view(encoded.value);
  // Pin all three paths to hard mode: this test's golden values
  // (kExpectedRawSwScore, per-step residue scores) are hard-SW outputs.
  // Hikoboshi 0.1.0's default remains hard Smith-Waterman, and these goldens
  // validate that branch explicitly.
  hiko::PairwiseEmbeddingRequest embedding_request{embedding, embedding};
  embedding_request.mode = hiko::AlignmentMode::Hard;
  const auto embedding_pairwise = engine.pairwise(embedding_request);
  if (embedding_pairwise.status.code != hiko_u::StatusCode::Ok) {
    fail("embedding pairwise must return Ok");
  }

  hiko::PairwiseStructureRequest structure_request{structure, structure};
  structure_request.mode = hiko::AlignmentMode::Hard;
  const auto structure_pairwise = engine.pairwise(structure_request);
  if (structure_pairwise.status.code != hiko_u::StatusCode::Ok) {
    fail("structure pairwise must return Ok");
  }
  hiko::PairwiseCoordsRequest coords_request{coords, coords};
  coords_request.mode = hiko::AlignmentMode::Hard;
  const auto coords_pairwise = engine.pairwise(coords_request);
  if (coords_pairwise.status.code != hiko_u::StatusCode::Ok) {
    fail("coords pairwise must return Ok");
  }

  if (!nearly_equal(structure_pairwise.value.metrics.raw_sw_score,
                    embedding_pairwise.value.metrics.raw_sw_score) ||
      !nearly_equal(coords_pairwise.value.metrics.raw_sw_score,
                    embedding_pairwise.value.metrics.raw_sw_score)) {
    fail("structure, coords, and embedding paths must share deterministic score");
  }
  if (structure_pairwise.value.path.aligned_pairs !=
          embedding_pairwise.value.path.aligned_pairs ||
      coords_pairwise.value.path.aligned_pairs !=
          embedding_pairwise.value.path.aligned_pairs) {
    fail("structure, coords, and embedding paths must share aligned-pair count");
  }
  if (embedding_pairwise.value.path.aligned_pairs == 0 ||
      embedding_pairwise.value.metrics.raw_sw_score <= 0.0) {
    fail("synthetic end-to-end pairwise output must be non-empty");
  }
  if (!structure_pairwise.value.metrics.coverage_mean.valid ||
      !structure_pairwise.value.metrics.identity.valid) {
    fail("public result must propagate coverage and identity metrics");
  }

  const auto encoded_coords = engine.encode(hiko::EncodeCoordsRequest{coords});
  if (encoded_coords.status.code != hiko_u::StatusCode::Ok) {
    fail("coords encode must return Ok with default Hikoboshi-MPNN-64 weights");
  }
  if (encoded_coords.value.embedding.values.size() !=
      encoded.value.embedding.values.size()) {
    fail("coords encode shape must match structure encode shape");
  }

  // Real-package pairwise golden from hiko_w::default_mpnn_d64_package() on the
  // 3-residue structure above; residue scores were refreshed after the MF
  // wave (mf1 KNN+vCb policy, mf2 distance floor, mf3 W_e GEMM accumulation
  // order, mf5 per-residue mask gate).
  constexpr double kExpectedRawSwScore = 10.543645858764648;
  constexpr std::size_t kExpectedAlignedPairs = 3;
  if (!nearly_equal(structure_pairwise.value.metrics.raw_sw_score,
                    kExpectedRawSwScore, 5.0e-6) ||
      structure_pairwise.value.path.aligned_pairs != kExpectedAlignedPairs ||
      structure_pairwise.value.path.query_start != 0 ||
      structure_pairwise.value.path.query_end != 2 ||
      structure_pairwise.value.path.target_start != 0 ||
      structure_pairwise.value.path.target_end != 2) {
    fail("default-weight pairwise real-package golden drifted");
  }
  constexpr std::array<double, kExpectedAlignedPairs> kExpectedResidueScores = {
      4.6901264191,
      4.3586835861,
      1.4948357344,
  };
  if (structure_pairwise.value.path.steps.size() != kExpectedAlignedPairs) {
    fail("default-weight pairwise path length drifted");
  }
  for (std::size_t index = 0; index < kExpectedAlignedPairs; ++index) {
    const hiko_u::AlignmentStep& step = structure_pairwise.value.path.steps[index];
    if (step.query_index != static_cast<std::int32_t>(index) ||
        step.target_index != static_cast<std::int32_t>(index) ||
        !nearly_equal(step.residue_score, kExpectedResidueScores[index],
                      5.0e-6)) {
      fail("default-weight pairwise path step golden drifted");
    }
  }
}

}  // namespace

int main() {
  test_structure_coords_and_embedding_paths_are_deterministic();
  return 0;
}

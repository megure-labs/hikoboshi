#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "pairwise_parallel_encode_tests: %s\n", message);
  std::exit(1);
}

struct StructureBundle {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;
  std::vector<std::string> residue_numbers;
  std::vector<hiko_u::ResidueMetadataView> residues;
};

StructureBundle make_chain(std::size_t length, float coord_offset,
                           char residue_kind) {
  StructureBundle bundle;
  bundle.residue_codes.assign(length, residue_kind);
  bundle.residue_numbers.resize(length);
  bundle.residues.resize(length);
  bundle.coordinates.assign(length * hiko_u::kCanonicalAtomCount * 3, 0.0F);
  bundle.atom_sources.assign(length * hiko_u::kCanonicalAtomCount,
                             hiko_u::AtomSource::Observed);
  for (std::size_t i = 0; i < length; ++i) {
    bundle.residue_numbers[i] = std::to_string(static_cast<int>(i + 1));
    bundle.residues[i] = hiko_u::ResidueMetadataView{
        residue_kind,
        std::string_view{"ALA"},
        std::string_view{"A"},
        std::string_view{bundle.residue_numbers[i]},
        1,
        static_cast<std::int32_t>(i + 1),
        '\0',
        std::string_view{"synthetic"},
        static_cast<std::int64_t>(i),
        {},
        -1};
    for (std::size_t atom = 0; atom < hiko_u::kCanonicalAtomCount; ++atom) {
      const std::size_t base = (i * hiko_u::kCanonicalAtomCount + atom) * 3;
      bundle.coordinates[base + 0] =
          static_cast<float>(i) + coord_offset;
      bundle.coordinates[base + 1] =
          static_cast<float>(atom) * 0.5F + 0.1F * coord_offset;
      bundle.coordinates[base + 2] =
          coord_offset + 0.25F * static_cast<float>(i % 3);
    }
  }
  return bundle;
}

hiko_u::StructureView view_of(const StructureBundle& bundle) {
  return {bundle.residue_codes.size(),
          {bundle.coordinates.data(), bundle.coordinates.size()},
          {bundle.atom_sources.data(), bundle.atom_sources.size()},
          {bundle.residue_codes.data(), bundle.residue_codes.size()},
          {bundle.residues.data(), bundle.residues.size()},
          "synthetic",
          {},
          {}};
}

hiko::Engine make_engine() {
  const auto package_result = hiko_w::default_mpnn_d64_package();
  if (package_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.descriptor == nullptr) {
    fail("default Hikoboshi-MPNN-64 package must validate");
  }
  const auto weights_result = hiko_w::default_mpnn_d64();
  if (weights_result.status.code != hiko_u::StatusCode::Ok) {
    fail("default Hikoboshi-MPNN-64 weights must resolve");
  }
  hiko::EngineConfig config{};
  config.weights = weights_result.value;
  config.execution.backend = hiko_u::Backend::Scalar;
  return hiko::Engine(config);
}

void set_parallel_encode(bool enabled) {
  if (enabled) {
    ::unsetenv("HIKOBOSHI_PAIRWISE_DISABLE_PARALLEL_ENCODE");
  } else {
    ::setenv("HIKOBOSHI_PAIRWISE_DISABLE_PARALLEL_ENCODE", "1", 1);
  }
}

hiko::PairwiseResult run_pairwise(const hiko::Engine& engine,
                                 const hiko_u::StructureView& query,
                                 const hiko_u::StructureView& target,
                                 bool parallel_encode) {
  set_parallel_encode(parallel_encode);
  hiko::PairwiseStructureRequest request{query, target};
  request.mode = hiko::AlignmentMode::Hard;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("pairwise must return Ok");
  }
  return result.value;
}

void require_bytewise_equal(const std::vector<float>& parallel,
                            const std::vector<float>& sequential,
                            const char* label) {
  if (parallel.size() != sequential.size()) {
    std::fprintf(stderr,
                 "pairwise_parallel_encode_tests: %s vector length differs "
                 "(parallel=%zu sequential=%zu)\n",
                 label, parallel.size(), sequential.size());
    std::exit(1);
  }
  if (std::memcmp(parallel.data(), sequential.data(),
                  parallel.size() * sizeof(float)) != 0) {
    std::fprintf(stderr,
                 "pairwise_parallel_encode_tests: %s bytes differ between "
                 "parallel and sequential encode paths\n",
                 label);
    std::exit(1);
  }
}

void require_paths_byte_identical(const hiko::PairwiseResult& parallel,
                                  const hiko::PairwiseResult& sequential) {
  if (std::memcmp(&parallel.metrics.raw_sw_score,
                  &sequential.metrics.raw_sw_score,
                  sizeof(parallel.metrics.raw_sw_score)) != 0) {
    fail("raw_sw_score bytes differ");
  }
  if (parallel.path.aligned_pairs != sequential.path.aligned_pairs ||
      parallel.path.query_start != sequential.path.query_start ||
      parallel.path.query_end != sequential.path.query_end ||
      parallel.path.target_start != sequential.path.target_start ||
      parallel.path.target_end != sequential.path.target_end) {
    fail("alignment path summary differs");
  }
  if (parallel.path.steps.size() != sequential.path.steps.size()) {
    fail("alignment path step count differs");
  }
  for (std::size_t i = 0; i < parallel.path.steps.size(); ++i) {
    const auto& a = parallel.path.steps[i];
    const auto& b = sequential.path.steps[i];
    if (a.query_index != b.query_index || a.target_index != b.target_index ||
        std::memcmp(&a.residue_score, &b.residue_score,
                    sizeof(a.residue_score)) != 0) {
      fail("alignment path step bytes differ");
    }
  }
}

// Verify that the parallel two-encode path produces byte-identical output
// to the sequential fallback for the same input. The fallback is selected
// via the HIKOBOSHI_PAIRWISE_DISABLE_PARALLEL_ENCODE environment variable so
// no API surface changes.
void test_parallel_encode_byte_identity() {
  const StructureBundle query_bundle = make_chain(7, 0.0F, 'A');
  const StructureBundle target_bundle = make_chain(9, 1.5F, 'C');
  const hiko_u::StructureView query = view_of(query_bundle);
  const hiko_u::StructureView target = view_of(target_bundle);

  const hiko::Engine engine = make_engine();

  // Single-structure encode is sequential by construction (one MPNN forward
  // per call). Captures the canonical embedding bytes the parallel pairwise
  // path must reproduce.
  set_parallel_encode(true);
  const auto query_encoded =
      engine.encode(hiko::EncodeStructureRequest{query});
  if (query_encoded.status.code != hiko_u::StatusCode::Ok) {
    fail("reference query encode must return Ok");
  }
  const auto target_encoded =
      engine.encode(hiko::EncodeStructureRequest{target});
  if (target_encoded.status.code != hiko_u::StatusCode::Ok) {
    fail("reference target encode must return Ok");
  }

  const hiko::PairwiseResult parallel_result =
      run_pairwise(engine, query, target, /*parallel_encode=*/true);
  const hiko::PairwiseResult sequential_result =
      run_pairwise(engine, query, target, /*parallel_encode=*/false);
  require_paths_byte_identical(parallel_result, sequential_result);

  // Feed the canonical encoded embeddings into the embedding-only pairwise
  // entry point; the resulting alignment must also match the parallel
  // structure pairwise. This pins both the embedding bytes and the alignment
  // to the same reference.
  hiko_u::EmbeddingView query_view{
      query_encoded.value.embedding.residue_count,
      query_encoded.value.embedding.dimension,
      {query_encoded.value.embedding.values.data(),
       query_encoded.value.embedding.values.size()},
      {query_encoded.value.embedding.residue_codes.data(),
       query_encoded.value.embedding.residue_codes.size()},
      {query_encoded.value.embedding.residues.data(),
       query_encoded.value.embedding.residues.size()}};
  hiko_u::EmbeddingView target_view{
      target_encoded.value.embedding.residue_count,
      target_encoded.value.embedding.dimension,
      {target_encoded.value.embedding.values.data(),
       target_encoded.value.embedding.values.size()},
      {target_encoded.value.embedding.residue_codes.data(),
       target_encoded.value.embedding.residue_codes.size()},
      {target_encoded.value.embedding.residues.data(),
       target_encoded.value.embedding.residues.size()}};
  hiko::PairwiseEmbeddingRequest embedding_request{query_view, target_view};
  embedding_request.mode = hiko::AlignmentMode::Hard;
  const auto reference = engine.pairwise(embedding_request);
  if (reference.status.code != hiko_u::StatusCode::Ok) {
    fail("reference embedding pairwise must return Ok");
  }
  require_paths_byte_identical(parallel_result, reference.value);

  // Re-encode after the parallel pairwise call to confirm the structure
  // encode path agrees with the parallel two-encode bytes.
  const auto query_reencoded =
      engine.encode(hiko::EncodeStructureRequest{query});
  if (query_reencoded.status.code != hiko_u::StatusCode::Ok) {
    fail("verification query encode must return Ok");
  }
  const auto target_reencoded =
      engine.encode(hiko::EncodeStructureRequest{target});
  if (target_reencoded.status.code != hiko_u::StatusCode::Ok) {
    fail("verification target encode must return Ok");
  }
  require_bytewise_equal(query_reencoded.value.embedding.values,
                         query_encoded.value.embedding.values,
                         "query embedding");
  require_bytewise_equal(target_reencoded.value.embedding.values,
                         target_encoded.value.embedding.values,
                         "target embedding");
}

// Opt-in wall-time smoke (gated by HIKOBOSHI_PAIRWISE_PARALLEL_TEST_BENCH=1).
// Times parallel vs sequential pairwise on synthetic L=64 chains so the
// 2x speedup target can be observed locally. The chain length is large
// enough that the encoder cost dominates the per-call wall, while still
// keeping the bench under a few seconds.
double measure_back_to_back_wall(const hiko::Engine& engine,
                                 const hiko_u::StructureView& query,
                                 const hiko_u::StructureView& target,
                                 bool parallel_encode,
                                 std::size_t iterations) {
  set_parallel_encode(parallel_encode);
  hiko::PairwiseStructureRequest request{query, target};
  request.mode = hiko::AlignmentMode::Hard;
  // Warmup so JIT/allocator paths are amortized before timing.
  const auto warmup = engine.pairwise(request);
  if (warmup.status.code != hiko_u::StatusCode::Ok) {
    fail("wall-bench warmup must return Ok");
  }
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto result = engine.pairwise(request);
    if (result.status.code != hiko_u::StatusCode::Ok) {
      fail("wall-bench pairwise must return Ok");
    }
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

void test_wall_speedup_smoke() {
  const char* enabled = std::getenv("HIKOBOSHI_PAIRWISE_PARALLEL_TEST_BENCH");
  if (enabled == nullptr || enabled[0] == '\0' ||
      std::strcmp(enabled, "0") == 0) {
    return;
  }
  std::size_t length = 64;
  if (const char* override_length =
          std::getenv("HIKOBOSHI_PAIRWISE_PARALLEL_TEST_L")) {
    const long parsed = std::strtol(override_length, nullptr, 10);
    if (parsed > 0) {
      length = static_cast<std::size_t>(parsed);
    }
  }
  std::size_t iterations = 4;
  if (const char* override_iterations =
          std::getenv("HIKOBOSHI_PAIRWISE_PARALLEL_TEST_ITERATIONS")) {
    const long parsed = std::strtol(override_iterations, nullptr, 10);
    if (parsed > 0) {
      iterations = static_cast<std::size_t>(parsed);
    }
  }

  const StructureBundle query_bundle = make_chain(length, 0.0F, 'V');
  const StructureBundle target_bundle = make_chain(length, 0.5F, 'I');
  const hiko_u::StructureView query = view_of(query_bundle);
  const hiko_u::StructureView target = view_of(target_bundle);

  const hiko::Engine engine = make_engine();

  const double parallel_wall = measure_back_to_back_wall(
      engine, query, target, /*parallel_encode=*/true, iterations);
  const double sequential_wall = measure_back_to_back_wall(
      engine, query, target, /*parallel_encode=*/false, iterations);

  std::fprintf(stderr,
               "pairwise_parallel_encode_tests: wall-bench L=%zu "
               "iterations=%zu parallel=%.4fs sequential=%.4fs speedup=%.3fx\n",
               length, iterations, parallel_wall, sequential_wall,
               sequential_wall / parallel_wall);
  if (parallel_wall >= sequential_wall) {
    fail("wall-bench: parallel wall must beat sequential wall");
  }
}

// Back-to-back parallel pairwise calls must not leak resources, deadlock,
// or drift; every iteration produces the same bytes as the first.
void test_back_to_back_no_leaks_or_drift() {
  const StructureBundle query_bundle = make_chain(11, 0.0F, 'G');
  const StructureBundle target_bundle = make_chain(13, 0.75F, 'L');
  const hiko_u::StructureView query = view_of(query_bundle);
  const hiko_u::StructureView target = view_of(target_bundle);

  const hiko::Engine engine = make_engine();
  set_parallel_encode(true);

  hiko::PairwiseStructureRequest request{query, target};
  request.mode = hiko::AlignmentMode::Hard;
  const auto baseline = engine.pairwise(request);
  if (baseline.status.code != hiko_u::StatusCode::Ok) {
    fail("baseline back-to-back pairwise must return Ok");
  }

  constexpr std::size_t kIterations = 8;
  for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
    const auto result = engine.pairwise(request);
    if (result.status.code != hiko_u::StatusCode::Ok) {
      fail("back-to-back parallel pairwise must return Ok");
    }
    require_paths_byte_identical(result.value, baseline.value);
  }
}

}  // namespace

int main() {
  test_parallel_encode_byte_identity();
  test_back_to_back_no_leaks_or_drift();
  test_wall_speedup_smoke();
  return 0;
}

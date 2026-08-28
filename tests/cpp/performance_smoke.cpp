#define HIKOBOSHI_TEST_ALLOC_COUNTER_IMPLEMENTATION
#include "support/test_alloc_counter.hpp"

#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/path_builder.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_api = hikoboshi::api;
namespace hiko_m = hikoboshi::modules;
namespace hiko_md = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

// Chartered smoke baseline: ada6000-local, scalar CPU backend, Lq=Lt=200,
// hidden_dim=64 embedding-only pairwise. Budget: <= 500 ms per steady-state
// pairwise call and <= 2 MiB prepared pairwise workspace for this shape.
constexpr std::size_t kLength = 200;
constexpr std::size_t kHiddenDim = 64;
constexpr std::size_t kMeasuredIterations = 5;
constexpr std::size_t kMpnnMeasuredIterations = 5;
constexpr double kPairwiseBudgetMs = 500.0;
constexpr std::size_t kWorkspaceBudgetBytes = 2U * 1024U * 1024U;

template <typename T, typename = void>
struct HasSimilarityMethod : std::false_type {};

template <typename T>
struct HasSimilarityMethod<T, std::void_t<decltype(&T::similarity)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasScoreMethod : std::false_type {};

template <typename T>
struct HasScoreMethod<T, std::void_t<decltype(&T::score)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasScoreOnlyMember : std::false_type {};

template <typename T>
struct HasScoreOnlyMember<T, std::void_t<decltype(std::declval<T&>().score_only)>>
    : std::true_type {};

static_assert(!HasSimilarityMethod<hiko_api::Engine>::value,
              "Hikoboshi 0.1.0 must not expose Engine::similarity");
static_assert(!HasScoreMethod<hiko_api::Engine>::value,
              "Hikoboshi 0.1.0 must not expose Engine::score");
static_assert(!HasScoreOnlyMember<hiko_api::AllVsAllOptions>::value,
              "Hikoboshi 0.1.0 must not expose all-vs-all score_only");

void fail(const char* message) {
  std::fprintf(stderr, "performance_smoke: %s\n", message);
  std::exit(1);
}

void require_ok(hiko_u::Status status, const char* message) {
  if (status.code != hiko_u::StatusCode::Ok) {
    std::fprintf(stderr, "performance_smoke: %s: %s\n", message,
                 status.detail == nullptr ? "" : status.detail);
    std::exit(1);
  }
}

bool nearly_equal(double actual, double expected, double tolerance = 1.0e-6) {
  return std::fabs(actual - expected) <= tolerance;
}

std::vector<char> make_residue_codes() {
  constexpr char kResidues[] = {
      'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
  };
  std::vector<char> codes(kLength);
  for (std::size_t i = 0; i < kLength; ++i) {
    codes[i] = kResidues[i % (sizeof(kResidues) / sizeof(kResidues[0]))];
  }
  return codes;
}

std::vector<char> make_residue_codes(std::size_t residue_count) {
  constexpr char kResidues[] = {
      'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
  };
  std::vector<char> codes(residue_count);
  for (std::size_t i = 0; i < residue_count; ++i) {
    codes[i] = kResidues[i % (sizeof(kResidues) / sizeof(kResidues[0]))];
  }
  return codes;
}

std::vector<float> make_identity_embeddings() {
  std::vector<float> values(kLength * kHiddenDim, 0.0F);
  for (std::size_t residue = 0; residue < kLength; ++residue) {
    values[residue * kHiddenDim + (residue % kHiddenDim)] = 1.0F;
  }
  return values;
}

struct OwnedStructure {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;

  hiko_u::StructureView view() const noexcept {
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

struct OwnedMpnnWorkspace {
  std::vector<float> ca_coordinates;
  std::vector<float> residue_features;
  std::vector<std::int32_t> neighbor_indices;
  std::vector<float> neighbor_squared_distances;
  std::vector<float> rbf_features;
  std::vector<float> residue_state;
  std::vector<float> gathered_state;
  std::vector<float> edge_state;
  std::vector<float> message_state;
  std::vector<float> projected_message_state;
  std::vector<float> residue_scratch;
  std::vector<float> ffn_hidden;
  hiko_md::Mpnn64Workspace view{};
};

OwnedMpnnWorkspace make_mpnn_workspace(const hiko_md::Mpnn64MemoryPlan& plan) {
  OwnedMpnnWorkspace owned{};
  owned.ca_coordinates.resize(hiko_md::mpnn64_ca_coordinate_count(plan));
  owned.residue_features.resize(hiko_md::mpnn64_residue_feature_count(plan));
  owned.neighbor_indices.resize(hiko_md::mpnn64_neighbor_slot_count(plan));
  owned.neighbor_squared_distances.resize(hiko_md::mpnn64_neighbor_slot_count(plan));
  owned.rbf_features.resize(hiko_md::mpnn64_neighbor_rbf_count(plan));
  owned.residue_state.resize(hiko_md::mpnn64_residue_hidden_count(plan));
  owned.gathered_state.resize(hiko_md::mpnn64_neighbor_hidden_count(plan));
  owned.edge_state.resize(hiko_md::mpnn64_neighbor_hidden_count(plan));
  owned.message_state.resize(hiko_md::mpnn64_neighbor_hidden_count(plan));
  owned.projected_message_state.resize(hiko_md::mpnn64_neighbor_hidden_count(plan));
  owned.residue_scratch.resize(hiko_md::mpnn64_residue_hidden_count(plan));
  owned.ffn_hidden.resize(hiko_md::mpnn64_ffn_hidden_count(plan));
  owned.view = {
      plan,
      {owned.ca_coordinates.data(), owned.ca_coordinates.size()},
      {owned.residue_features.data(), owned.residue_features.size()},
      {owned.neighbor_indices.data(), owned.neighbor_indices.size()},
      {owned.neighbor_squared_distances.data(),
       owned.neighbor_squared_distances.size()},
      {owned.rbf_features.data(), owned.rbf_features.size()},
      {owned.residue_state.data(), owned.residue_state.size()},
      {owned.gathered_state.data(), owned.gathered_state.size()},
      {owned.edge_state.data(), owned.edge_state.size()},
      {owned.message_state.data(), owned.message_state.size()},
      {owned.projected_message_state.data(),
       owned.projected_message_state.size()},
      {owned.residue_scratch.data(), owned.residue_scratch.size()},
      {owned.ffn_hidden.data(), owned.ffn_hidden.size()},
  };
  return owned;
}

std::vector<float> make_mpnn_coordinates(std::size_t residue_count) {
  std::vector<float> coordinates(residue_count * hiko_md::kMpnn64AtomCount *
                                     hiko_md::kMpnn64AxisCount,
                                 0.0F);
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    const float z = 1.5F * static_cast<float>(residue);
    constexpr float atoms[hiko_md::kMpnn64AtomCount][hiko_md::kMpnn64AxisCount] = {
        {0.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F},
        {2.0F, 0.0F, 0.0F},
        {2.0F, 1.0F, 0.0F},
        {1.0F, 1.0F, 0.0F},
    };
    for (std::size_t atom = 0; atom < hiko_md::kMpnn64AtomCount; ++atom) {
      for (std::size_t axis = 0; axis < hiko_md::kMpnn64AxisCount; ++axis) {
        coordinates[(residue * hiko_md::kMpnn64AtomCount + atom) *
                        hiko_md::kMpnn64AxisCount +
                    axis] = atoms[atom][axis] + (axis == 2 ? z : 0.0F);
      }
    }
  }
  return coordinates;
}

OwnedStructure make_observed_structure(std::size_t residue_count) {
  OwnedStructure structure{};
  structure.coordinates = make_mpnn_coordinates(residue_count);
  structure.atom_sources.assign(residue_count * hiko_u::kCanonicalAtomCount,
                                hiko_u::AtomSource::Observed);
  structure.residue_codes = make_residue_codes(residue_count);
  return structure;
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values,
                                  const std::vector<char>& codes) {
  return {kLength,
          kHiddenDim,
          {values.data(), values.size()},
          {codes.data(), codes.size()},
          {nullptr, 0}};
}

class CountingAllVsAllSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    if (record.result.path.steps.empty()) {
      fail("all-vs-all hot path must emit non-empty pairwise paths");
    }
    ++count;
    return hiko_u::ok_status();
  }

  std::size_t count = 0;
};

hiko_ad::PairwiseWorkspacePlan embedding_plan() {
  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = kLength;
  plan.max_target_length = kLength;
  plan.embedding_dimension = kHiddenDim;
  plan.allocate_mpnn = false;
  return plan;
}

std::size_t estimated_workspace_bytes() {
  const std::size_t embedding_bytes = 2U * kLength * kHiddenDim * sizeof(float);
  const std::size_t similarity_bytes = kLength * kLength * sizeof(float);
  const std::size_t sw_cells = (kLength + 1U) * (kLength + 1U);
  const std::size_t sw_bytes = 3U * sw_cells * sizeof(float);
  const std::size_t trace_bytes = 3U * kLength * kLength *
                                  sizeof(hiko_ad::SwTraceDirection);
  const std::size_t path_bytes =
      (kLength + kLength) * sizeof(hiko_u::AlignmentStep);
  return embedding_bytes + similarity_bytes + sw_bytes + trace_bytes +
         path_bytes;
}

hiko::PairwiseResult run_algorithm_pairwise(const hiko_u::EmbeddingView& query,
                                           const hiko_u::EmbeddingView& target,
                                           hiko_ad::PairwiseWorkspace& workspace) {
  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = query;
  request.target_embedding = target;

  hiko::PairwiseResult result{};
  require_ok(hiko::run_pairwise_embeddings(request, workspace, result),
             "algorithm pairwise");
  return result;
}

void test_cpp_api_parity_with_algorithm() {
  const std::vector<float> values = make_identity_embeddings();
  const std::vector<char> codes = make_residue_codes();
  const hiko_u::EmbeddingView embedding = embedding_view(values, codes);

  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(embedding_plan()), "workspace prepare");
  const hiko::PairwiseResult algorithm_result =
      run_algorithm_pairwise(embedding, embedding, workspace);

  const hiko_api::Engine engine({{nullptr, nullptr}, {hiko_u::Backend::Scalar, 0}});
  const auto api_result =
      engine.pairwise(hiko_api::PairwiseEmbeddingRequest{embedding, embedding});
  if (api_result.status.code != hiko_u::StatusCode::Ok) {
    fail("C++ API pairwise embedding request must return Ok");
  }
  if (!nearly_equal(api_result.value.metrics.raw_sw_score,
                    algorithm_result.raw_sw_score) ||
      api_result.value.path.aligned_pairs != algorithm_result.path.aligned_pairs) {
    fail("C++ API and algorithm pairwise results must stay in parity");
  }
  if (algorithm_result.path.aligned_pairs != kLength ||
      !nearly_equal(algorithm_result.raw_sw_score, static_cast<double>(kLength))) {
    fail("representative identity embeddings must align the full diagonal");
  }
}

void test_workspace_budget() {
  const std::size_t bytes = estimated_workspace_bytes();
  if (bytes > kWorkspaceBudgetBytes) {
    fail("estimated prepared workspace exceeds chartered smoke budget");
  }
}

void test_no_allocation_after_prepare_for_pairwise_hot_path() {
  const std::vector<float> values = make_identity_embeddings();
  const std::vector<char> codes = make_residue_codes();
  const hiko_u::EmbeddingView embedding = embedding_view(values, codes);

  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(embedding_plan()), "workspace prepare");

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = embedding;
  request.target_embedding = embedding;
  hiko::PairwiseResult result{};
  require_ok(hiko::run_pairwise_embeddings(request, workspace, result),
             "pairwise result handoff warmup");
  if (result.path.steps.size() != kLength ||
      result.path.aligned_pairs != kLength) {
    fail("pairwise hot-path warmup must produce a non-empty traceback path");
  }

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t i = 0; i < 20; ++i) {
    require_ok(hiko::run_pairwise_embeddings(request, workspace, result),
               "steady-state identity pairwise");
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0U) {
    fail("pairwise hot path allocated after workspace preparation");
  }
}

void test_no_allocation_after_prepare_for_pairwise_structure_hot_path() {
  constexpr std::size_t kStructureResidues = 10;
  const hiko_w::WeightManifestView& manifest = hiko_w::default_mpnn_d64_manifest();
  const hiko_m::Mpnn64Descriptor descriptor{manifest.hidden_dimension,
                                         manifest.neighbor_count,
                                         manifest.rbf_count,
                                         manifest.layer_count,
                                         manifest.message_scale};
  const auto weights_result = hiko_w::default_mpnn_d64();
  require_ok(weights_result.status, "default MPNN weights");
  const auto* prepared =
      static_cast<const hiko_md::Mpnn64Weights*>(weights_result.value.opaque);
  if (prepared == nullptr) {
    fail("default MPNN weights must expose prepared module state");
  }

  const OwnedStructure query = make_observed_structure(kStructureResidues);
  const OwnedStructure target = make_observed_structure(kStructureResidues);

  hiko_ad::PairwiseWorkspacePlan plan{};
  plan.max_query_length = kStructureResidues;
  plan.max_target_length = kStructureResidues;
  plan.embedding_dimension = descriptor.hidden_dimension;
  plan.allocate_mpnn = true;
  plan.mpnn_descriptor = descriptor;

  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(plan), "structure workspace prepare");

  hiko::PairwiseStructureRequest request{};
  request.query = query.view();
  request.target = target.view();
  request.descriptor = descriptor;
  request.weights = prepared;

  hiko::PairwiseResult result{};
  require_ok(hiko::run_pairwise_structures(request, workspace, result),
             "pairwise structure warmup");
  if (!result.metrics.rmsd.valid || !result.metrics.tm_score_query.valid ||
      !result.metrics.tm_score_target.valid || !result.metrics.lddt.valid) {
    fail("pairwise structure warmup must exercise structural metrics");
  }

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    require_ok(hiko::run_pairwise_structures(request, workspace, result),
               "steady-state pairwise structure");
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0U) {
    fail("pairwise structure hot path allocated after workspace preparation");
  }
}

void test_no_allocation_after_prepare_for_all_vs_all_single_worker_hot_path() {
  const std::vector<float> values = make_identity_embeddings();
  const std::vector<char> codes = make_residue_codes();
  const hiko_u::EmbeddingView embedding = embedding_view(values, codes);
  const std::vector<hiko_u::EmbeddingView> embeddings = {
      embedding,
      embedding,
      embedding,
  };

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_ad::AllVsAllWorkerWorkspace worker{};
  CountingAllVsAllSink sink;
  require_ok(hiko::run_all_vs_all_embeddings(request, sink, nullptr, 1U,
                                            {&worker, 1U}),
             "all-vs-all single-worker warmup");

  const std::size_t expected_pairs =
      embeddings.size() * (embeddings.size() - 1U) / 2U;
  if (sink.count != expected_pairs) {
    fail("all-vs-all warmup emitted the wrong pair count");
  }

  sink.count = 0;
  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    require_ok(hiko::run_all_vs_all_embeddings(request, sink, nullptr, 1U,
                                              {&worker, 1U}),
               "steady-state all-vs-all single-worker");
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0U) {
    fail("all-vs-all single-worker hot path allocated after workspace warmup");
  }
  if (sink.count != expected_pairs * 5U) {
    fail("all-vs-all steady-state emitted the wrong pair count");
  }
}

void test_no_allocation_for_con1_traceback_path() {
  hiko_ad::PathBuilder builder;
  builder.prepare(kLength);
  hiko_u::AlignmentPath output;
  output.steps.reserve(kLength);

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  for (std::size_t iteration = 0; iteration < 20; ++iteration) {
    builder.reset();
    for (std::size_t i = 0; i < kLength; ++i) {
      const std::int32_t index = static_cast<std::int32_t>(kLength - 1U - i);
      if (!builder.push_reverse({index, index, 1.0F})) {
        fail("CON-1 traceback path scratch capacity is too small");
      }
    }
    builder.set_span(0, static_cast<std::int32_t>(kLength - 1U), 0,
                     static_cast<std::int32_t>(kLength - 1U), kLength);
    builder.write_ordered_to(output);
  }
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0U) {
    fail("CON-1 traceback path allocated after PathBuilder preparation");
  }
  if (output.steps.size() != kLength || output.aligned_pairs != kLength) {
    fail("CON-1 traceback path output shape mismatch");
  }
}

double test_no_allocation_after_prepare_for_mpnn_hot_path() {
  constexpr std::size_t kMpnnResidues = 3;
  const hiko_w::WeightManifestView& manifest = hiko_w::default_mpnn_d64_manifest();
  const hiko_m::Mpnn64Descriptor descriptor{manifest.hidden_dimension,
                                         manifest.neighbor_count,
                                         manifest.rbf_count,
                                         manifest.layer_count,
                                         manifest.message_scale};
  const auto weights_result = hiko_w::default_mpnn_d64();
  require_ok(weights_result.status, "default MPNN weights");
  const auto* prepared =
      static_cast<const hiko_md::Mpnn64Weights*>(weights_result.value.opaque);
  if (prepared == nullptr) {
    fail("default MPNN weights must expose prepared module state");
  }

  const hiko_md::Mpnn64MemoryPlan plan{kMpnnResidues,
                                    descriptor.hidden_dimension,
                                    descriptor.neighbor_count,
                                    descriptor.rbf_count,
                                    descriptor.layer_count};
  OwnedMpnnWorkspace workspace = make_mpnn_workspace(plan);
  const std::vector<float> coordinates = make_mpnn_coordinates(kMpnnResidues);
  std::vector<hiko_u::AtomSource> atom_sources(kMpnnResidues *
                                                hiko_md::kMpnn64AtomCount,
                                            hiko_u::AtomSource::Observed);
  std::vector<float> embeddings(kMpnnResidues * descriptor.hidden_dimension,
                                0.0F);

  require_ok(hiko_m::mpnn64_forward_scalar(
                 {coordinates.data(), atom_sources.data(), kMpnnResidues,
                  descriptor, prepared, &workspace.view},
                 {embeddings.data(), kMpnnResidues,
                  descriptor.hidden_dimension}),
             "MPNN warmup");

  hikoboshi::tests::AllocationCounter::reset();
  hikoboshi::tests::AllocationCounter::set_enabled(true);
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < kMpnnMeasuredIterations;
       ++iteration) {
    require_ok(hiko_m::mpnn64_forward_scalar(
                   {coordinates.data(), atom_sources.data(), kMpnnResidues,
                    descriptor, prepared, &workspace.view},
                   {embeddings.data(), kMpnnResidues,
                    descriptor.hidden_dimension}),
               "steady-state MPNN");
  }
  const auto stop = std::chrono::steady_clock::now();
  hikoboshi::tests::AllocationCounter::set_enabled(false);

  if (hikoboshi::tests::AllocationCounter::allocations() != 0U) {
    fail("MPNN hot path allocated after workspace preparation");
  }
  for (float value : embeddings) {
    if (!std::isfinite(value)) {
      fail("MPNN hot path produced a non-finite embedding value");
    }
  }
  const std::chrono::duration<double, std::milli> elapsed = stop - start;
  return elapsed.count() / static_cast<double>(kMpnnMeasuredIterations);
}

double measure_pairwise_ms() {
  const std::vector<float> values = make_identity_embeddings();
  const std::vector<char> codes = make_residue_codes();
  const hiko_u::EmbeddingView embedding = embedding_view(values, codes);

  hiko_ad::PairwiseWorkspace workspace;
  require_ok(workspace.prepare(embedding_plan()), "workspace prepare");
  (void)run_algorithm_pairwise(embedding, embedding, workspace);

  hiko::PairwiseEmbeddingRequest request{};
  request.query_embedding = embedding;
  request.target_embedding = embedding;
  hiko::PairwiseResult result{};

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < kMeasuredIterations; ++i) {
    require_ok(hiko::run_pairwise_embeddings(request, workspace, result),
               "measured pairwise");
  }
  const auto stop = std::chrono::steady_clock::now();
  const std::chrono::duration<double, std::milli> elapsed = stop - start;
  const double measured_ms = elapsed.count() /
                             static_cast<double>(kMeasuredIterations);
  if (measured_ms > kPairwiseBudgetMs) {
    fail("representative pairwise smoke exceeded chartered time budget");
  }
  return measured_ms;
}

}  // namespace

int main() {
  test_cpp_api_parity_with_algorithm();
  test_workspace_budget();
  test_no_allocation_after_prepare_for_pairwise_hot_path();
  test_no_allocation_after_prepare_for_pairwise_structure_hot_path();
  test_no_allocation_after_prepare_for_all_vs_all_single_worker_hot_path();
  test_no_allocation_for_con1_traceback_path();
  const double mpnn_measured_ms =
      test_no_allocation_after_prepare_for_mpnn_hot_path();
  const double measured_ms = measure_pairwise_ms();
  std::printf(
      "hikoboshi_performance_smoke representative_length=%zu hidden_dim=%zu "
      "iterations=%zu measured_ms=%.6f budget_ms=%.3f "
      "workspace_bytes_estimate=%zu workspace_budget_bytes=%zu "
      "real_mpnn_residues=%zu real_mpnn_iterations=%zu "
      "real_mpnn_measured_ms=%.6f\n",
      kLength, kHiddenDim, kMeasuredIterations, measured_ms, kPairwiseBudgetMs,
      estimated_workspace_bytes(), kWorkspaceBudgetBytes,
      static_cast<std::size_t>(3), kMpnnMeasuredIterations, mpnn_measured_ms);
  return 0;
}

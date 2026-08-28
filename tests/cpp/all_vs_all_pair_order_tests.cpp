#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_d = hikoboshi::algorithms::detail;
namespace hiko_m = hikoboshi::modules;
namespace hiko_md = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_ud = hikoboshi::universal::detail;
namespace hiko_w = hikoboshi::weights;

static std::size_t g_mpnn64_forward_calls = 0;

extern "C" hiko_u::Status
__real__ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE(
    const hiko_m::Mpnn64ForwardRequest& request,
    const hiko_m::Mpnn64ForwardOutput& output);

extern "C" hiko_u::Status
__wrap__ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE(
    const hiko_m::Mpnn64ForwardRequest& request,
    const hiko_m::Mpnn64ForwardOutput& output) {
  ++g_mpnn64_forward_calls;
  return
      __real__ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE(
          request, output);
}

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_pair_order_tests: %s\n", message);
  std::exit(1);
}

void expect_pair(hiko_d::PairIndex actual,
                 std::size_t query_index,
                 std::size_t target_index) {
  if (actual.query_index != query_index || actual.target_index != target_index) {
    fail("pair index mapping mismatch");
  }
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values) {
  return {values.size(), 1, {values.data(), values.size()}, {nullptr, 0}, {nullptr, 0}};
}

struct StructureFixture {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;

  hiko_u::StructureView view() const {
    return {residue_codes.size(),
            {coordinates.data(), coordinates.size()},
            {atom_sources.data(), atom_sources.size()},
            {residue_codes.data(), residue_codes.size()},
            {nullptr, 0},
            "all_vs_all_pair_order_fixture",
            {},
            {}};
  }
};

StructureFixture structure_fixture(float offset) {
  StructureFixture fixture{};
  fixture.coordinates = {
      offset + 0.0F, 0.0F, 0.0F,
      offset + 1.0F, 0.0F, 0.0F,
      offset + 2.0F, 0.0F, 0.0F,
      offset + 2.0F, 1.0F, 0.0F,
      offset + 1.0F, 1.0F, 0.0F,
  };
  fixture.atom_sources.assign(hiko_u::kCanonicalAtomCount,
                              hiko_u::AtomSource::Observed);
  fixture.residue_codes = {'A'};
  return fixture;
}

hiko_m::Mpnn64Descriptor mpnn64_descriptor_fixture() {
  const hiko_w::WeightManifestView& manifest = hiko_w::default_mpnn_d64_manifest();
  return {manifest.hidden_dimension,
          manifest.neighbor_count,
          manifest.rbf_count,
          manifest.layer_count,
          manifest.message_scale};
}

const hiko_md::Mpnn64Weights* default_prepared_weights() {
  const hiko_u::Result<hiko_u::WeightsHandle> weights = hiko_w::default_mpnn_d64();
  if (weights.status.code != hiko_u::StatusCode::Ok) {
    fail("default Hikoboshi-MPNN-64 weights must be available");
  }
  const auto* prepared =
      static_cast<const hiko_md::Mpnn64Weights*>(weights.value.opaque);
  if (prepared == nullptr) {
    fail("default Hikoboshi-MPNN-64 weights must expose prepared state");
  }
  return prepared;
}

class RecordingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    pairs.push_back({record.query_index, record.target_index});
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::vector<hiko_d::PairIndex> pairs;
};

void test_pair_counts() {
  if (hiko_d::symmetric_pair_count(0, false) != 0 ||
      hiko_d::symmetric_pair_count(1, false) != 0 ||
      hiko_d::symmetric_pair_count(2, false) != 1 ||
      hiko_d::symmetric_pair_count(3, false) != 3 ||
      hiko_d::symmetric_pair_count(0, true) != 0 ||
      hiko_d::symmetric_pair_count(1, true) != 1 ||
      hiko_d::symmetric_pair_count(2, true) != 3 ||
      hiko_d::symmetric_pair_count(3, true) != 6) {
    fail("symmetric pair count mismatch");
  }
}

void test_default_pair_index_mapping() {
  const hiko_d::PairIndex expected[] = {
      {0, 1},
      {0, 2},
      {0, 3},
      {1, 2},
      {1, 3},
      {2, 3},
  };
  for (std::size_t index = 0; index < 6; ++index) {
    expect_pair(hiko_d::pair_index_to_ij(index, 4, false),
                expected[index].query_index,
                expected[index].target_index);
  }
  expect_pair(hiko_d::pair_index_to_ij(6, 4, false), 4, 4);
}

void test_include_self_pair_index_mapping() {
  const hiko_d::PairIndex expected[] = {
      {0, 0},
      {0, 1},
      {0, 2},
      {0, 3},
      {1, 1},
      {1, 2},
      {1, 3},
      {2, 2},
      {2, 3},
      {3, 3},
  };
  for (std::size_t index = 0; index < 10; ++index) {
    expect_pair(hiko_d::pair_index_to_ij(index, 4, true),
                expected[index].query_index,
                expected[index].target_index);
  }
  expect_pair(hiko_d::pair_index_to_ij(10, 4, true), 4, 4);
}

void test_partition_ranges_are_stable() {
  const hiko_d::PairRange first = hiko_d::partition_pair_range(10, 0, 3);
  const hiko_d::PairRange second = hiko_d::partition_pair_range(10, 1, 3);
  const hiko_d::PairRange third = hiko_d::partition_pair_range(10, 2, 3);
  if (first.begin != 0 || first.end != 4 || second.begin != 4 ||
      second.end != 7 || third.begin != 7 || third.end != 10) {
    fail("pair range partitioning mismatch");
  }
}

void test_algorithm_emits_lexicographic_order(bool include_self) {
  const std::vector<std::vector<float>> storage = {{1.0F}, {2.0F}, {3.0F}, {4.0F}};
  std::vector<hiko_u::EmbeddingView> embeddings;
  embeddings.reserve(storage.size());
  for (const auto& values : storage) {
    embeddings.push_back(embedding_view(values));
  }

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.include_self = include_self;

  RecordingSink sink;
  const hiko_u::Status status = hiko::run_all_vs_all_embeddings(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("all-vs-all embedding order run must return Ok");
  }
  const std::size_t expected_count =
      hiko_d::symmetric_pair_count(embeddings.size(), include_self);
  if (sink.pairs.size() != expected_count) {
    fail("all-vs-all emitted pair count mismatch");
  }
  for (std::size_t index = 0; index < expected_count; ++index) {
    const hiko_d::PairIndex expected =
        hiko_d::pair_index_to_ij(index, embeddings.size(), include_self);
    expect_pair(sink.pairs[index], expected.query_index, expected.target_index);
  }
}

void test_parallel_algorithm_emits_lexicographic_order(bool include_self) {
  constexpr std::size_t kInputCount = 10;
  std::vector<std::vector<float>> storage;
  storage.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    storage.push_back({static_cast<float>(index + 1U)});
  }
  std::vector<hiko_u::EmbeddingView> embeddings;
  embeddings.reserve(storage.size());
  for (const auto& values : storage) {
    embeddings.push_back(embedding_view(values));
  }

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.include_self = include_self;

  hiko_ud::ThreadPool pool(4);
  std::vector<hiko_d::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  RecordingSink sink;
  const hiko_u::Status status =
      hiko::run_all_vs_all_embeddings(request,
                                     sink,
                                     &pool,
                                     pool.thread_count(),
                                     {workers.data(), workers.size()});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("parallel all-vs-all embedding order run must return Ok");
  }
  const std::size_t expected_count =
      hiko_d::symmetric_pair_count(embeddings.size(), include_self);
  if (sink.pairs.size() != expected_count) {
    fail("parallel all-vs-all emitted pair count mismatch");
  }
  for (std::size_t index = 0; index < expected_count; ++index) {
    const hiko_d::PairIndex expected =
        hiko_d::pair_index_to_ij(index, embeddings.size(), include_self);
    expect_pair(sink.pairs[index], expected.query_index, expected.target_index);
  }
}

void test_structure_all_vs_all_encodes_each_input_once(bool include_self) {
  const std::vector<StructureFixture> storage = {
      structure_fixture(0.0F),
      structure_fixture(1.0F),
      structure_fixture(2.0F),
  };
  std::vector<hiko_u::StructureView> structures;
  structures.reserve(storage.size());
  for (const StructureFixture& fixture : storage) {
    structures.push_back(fixture.view());
  }

  hiko::AllVsAllStructureRequest request{};
  request.structures = {structures.data(), structures.size()};
  request.descriptor = mpnn64_descriptor_fixture();
  request.weights = default_prepared_weights();
  request.options.include_self = include_self;

  RecordingSink sink;
  g_mpnn64_forward_calls = 0;
  const hiko_u::Status status = hiko::run_all_vs_all_structures(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("structure all-vs-all run must return Ok");
  }

  const std::size_t expected_count =
      hiko_d::symmetric_pair_count(structures.size(), include_self);
  if (sink.pairs.size() != expected_count) {
    fail("structure all-vs-all emitted pair count mismatch");
  }
  if (g_mpnn64_forward_calls != structures.size()) {
    fail("structure all-vs-all must encode each input exactly once");
  }
}

}  // namespace

int main() {
  test_pair_counts();
  test_default_pair_index_mapping();
  test_include_self_pair_index_mapping();
  test_partition_ranges_are_stable();
  test_algorithm_emits_lexicographic_order(false);
  test_algorithm_emits_lexicographic_order(true);
  test_parallel_algorithm_emits_lexicographic_order(false);
  test_parallel_algorithm_emits_lexicographic_order(true);
  test_structure_all_vs_all_encodes_each_input_once(false);
  test_structure_all_vs_all_encodes_each_input_once(true);
  return 0;
}

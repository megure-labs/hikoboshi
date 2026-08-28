#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_api = hikoboshi::api;
namespace hiko_m = hikoboshi::modules;
namespace hiko_md = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_ud = hikoboshi::universal::detail;
namespace hiko_w = hikoboshi::weights;

namespace {

std::mutex g_forward_mutex;
std::vector<std::thread::id> g_forward_threads;

}  // namespace

extern "C" hiko_u::Status
__wrap__ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE(
    const hiko_m::Mpnn64ForwardRequest& request,
    const hiko_m::Mpnn64ForwardOutput& output) {
  {
    std::lock_guard<std::mutex> lock(g_forward_mutex);
    g_forward_threads.push_back(std::this_thread::get_id());
  }

  const float base = request.coordinates == nullptr ? 0.0F
                                                    : request.coordinates[0];
  for (std::size_t residue = 0; residue < request.residue_count; ++residue) {
    for (std::size_t dim = 0; dim < output.hidden_dimension; ++dim) {
      output.embeddings[residue * output.hidden_dimension + dim] =
          1.0F + base + static_cast<float>(residue) * 0.25F +
          static_cast<float>(dim + 1U) * 0.001F;
    }
  }
  return hiko_u::ok_status();
}

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_parallel_encoding_tests: %s\n", message);
  std::exit(1);
}

void reset_forward_tracking() {
  std::lock_guard<std::mutex> lock(g_forward_mutex);
  g_forward_threads.clear();
}

std::size_t forward_call_count() {
  std::lock_guard<std::mutex> lock(g_forward_mutex);
  return g_forward_threads.size();
}

std::size_t unique_forward_thread_count() {
  std::lock_guard<std::mutex> lock(g_forward_mutex);
  std::vector<std::thread::id> unique = g_forward_threads;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  return unique.size();
}

bool nearly_equal(double actual, double expected, double tolerance = 1.0e-6) {
  return std::fabs(actual - expected) <= tolerance;
}

struct StructureFixture {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;

  hiko_u::StructureView structure_view() const {
    return {residue_codes.size(),
            {coordinates.data(), coordinates.size()},
            {atom_sources.data(), atom_sources.size()},
            {residue_codes.data(), residue_codes.size()},
            {nullptr, 0},
            "parallel_encoding_fixture",
            {},
            {}};
  }

  hiko_api::CoordsInputView coords_view() const {
    return {residue_codes.size(),
            {coordinates.data(), coordinates.size()},
            {atom_sources.data(), atom_sources.size()},
            {residue_codes.data(), residue_codes.size()},
            {nullptr, 0}};
  }
};

StructureFixture make_structure_fixture(std::size_t residue_count,
                                        float offset) {
  StructureFixture fixture{};
  fixture.coordinates.resize(residue_count * hiko_u::kCanonicalAtomCount *
                             hiko_u::kCoordinateAxisCount);
  fixture.atom_sources.assign(residue_count * hiko_u::kCanonicalAtomCount,
                              hiko_u::AtomSource::Observed);
  fixture.residue_codes.resize(residue_count);
  constexpr char kResidues[] = {'A', 'C', 'D', 'E', 'F', 'G'};
  for (std::size_t residue = 0; residue < residue_count; ++residue) {
    fixture.residue_codes[residue] = kResidues[residue % 6U];
    for (std::size_t atom = 0; atom < hiko_u::kCanonicalAtomCount; ++atom) {
      for (std::size_t axis = 0; axis < hiko_u::kCoordinateAxisCount; ++axis) {
        const std::size_t flat =
            (residue * hiko_u::kCanonicalAtomCount + atom) *
                hiko_u::kCoordinateAxisCount +
            axis;
        fixture.coordinates[flat] =
            offset + static_cast<float>(residue) * 0.1F +
            static_cast<float>(atom) * 0.01F +
            static_cast<float>(axis) * 0.001F;
      }
    }
  }
  return fixture;
}

std::vector<StructureFixture> make_structure_fixtures() {
  std::vector<StructureFixture> fixtures;
  fixtures.reserve(4);
  for (std::size_t index = 0; index < 4; ++index) {
    fixtures.push_back(
        make_structure_fixture(3, static_cast<float>(index) * 0.5F));
  }
  return fixtures;
}

std::vector<hiko_u::StructureView> make_structure_views(
    const std::vector<StructureFixture>& fixtures) {
  std::vector<hiko_u::StructureView> structures;
  structures.reserve(fixtures.size());
  for (const StructureFixture& fixture : fixtures) {
    structures.push_back(fixture.structure_view());
  }
  return structures;
}

std::vector<hiko_api::CoordsInputView> make_coords_views(
    const std::vector<StructureFixture>& fixtures) {
  std::vector<hiko_api::CoordsInputView> coords;
  coords.reserve(fixtures.size());
  for (const StructureFixture& fixture : fixtures) {
    coords.push_back(fixture.coords_view());
  }
  return coords;
}

class AlgorithmCollectingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    records.push_back(record);
    return hiko_u::ok_status();
  }

  std::vector<hiko::PairwiseResultRecord> records;
};

void require_algorithm_records_match(
    const std::vector<hiko::PairwiseResultRecord>& serial,
    const std::vector<hiko::PairwiseResultRecord>& parallel) {
  if (serial.size() != parallel.size()) {
    fail("serial and parallel structure record counts differ");
  }
  for (std::size_t index = 0; index < serial.size(); ++index) {
    const hiko::PairwiseResultRecord& lhs = serial[index];
    const hiko::PairwiseResultRecord& rhs = parallel[index];
    if (lhs.query_index != rhs.query_index ||
        lhs.target_index != rhs.target_index ||
        !nearly_equal(lhs.result.metrics.raw_sw_score,
                      rhs.result.metrics.raw_sw_score) ||
        lhs.result.path.aligned_pairs != rhs.result.path.aligned_pairs ||
        lhs.result.path.steps.size() != rhs.result.path.steps.size()) {
      fail("serial and parallel structure records differ");
    }
  }
}

void require_api_records_match(const hiko_api::AllVsAllResult& serial,
                               const hiko_api::AllVsAllResult& parallel) {
  if (serial.records.size() != parallel.records.size()) {
    fail("serial and parallel coords record counts differ");
  }
  for (std::size_t index = 0; index < serial.records.size(); ++index) {
    const hiko_api::PairwiseResultRecord& lhs = serial.records[index];
    const hiko_api::PairwiseResultRecord& rhs = parallel.records[index];
    if (lhs.query_index != rhs.query_index ||
        lhs.target_index != rhs.target_index ||
        !nearly_equal(lhs.result.metrics.raw_sw_score,
                      rhs.result.metrics.raw_sw_score) ||
        lhs.result.path.aligned_pairs != rhs.result.path.aligned_pairs ||
        lhs.result.path.steps.size() != rhs.result.path.steps.size()) {
      fail("serial and parallel coords records differ");
    }
  }
}

hiko_m::Mpnn64Descriptor fast_descriptor() {
  return {64, 2, 2, 1, 1.0F};
}

hiko_api::EngineConfig engine_config(std::uint32_t thread_count) {
  const hiko_u::Result<hiko_u::WeightsHandle> weights = hiko_w::default_mpnn_d64();
  if (weights.status.code != hiko_u::StatusCode::Ok) {
    fail("default Hikoboshi-MPNN-64 weights must be available");
  }
  hiko_api::EngineConfig config{};
  config.weights = weights.value;
  config.execution.thread_count = thread_count;
  return config;
}

void test_algorithm_parallel_structure_encoding_matches_serial() {
  const std::vector<StructureFixture> fixtures = make_structure_fixtures();
  const std::vector<hiko_u::StructureView> structures =
      make_structure_views(fixtures);
  hiko_md::Mpnn64Weights dummy_weights{};

  hiko::AllVsAllStructureRequest request{};
  request.structures = {structures.data(), structures.size()};
  request.descriptor = fast_descriptor();
  request.weights = &dummy_weights;

  reset_forward_tracking();
  AlgorithmCollectingSink serial_sink;
  const hiko_u::Status serial_status =
      hiko::run_all_vs_all_structures(request, serial_sink);
  if (serial_status.code != hiko_u::StatusCode::Ok ||
      forward_call_count() != structures.size()) {
    fail("serial structure encoding must encode each input once");
  }

  reset_forward_tracking();
  hiko_ud::ThreadPool pool(4);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  AlgorithmCollectingSink parallel_sink;
  const hiko_u::Status parallel_status =
      hiko::run_all_vs_all_structures(request,
                                     parallel_sink,
                                     &pool,
                                     pool.thread_count(),
                                     {workers.data(), workers.size()});
  if (parallel_status.code != hiko_u::StatusCode::Ok ||
      forward_call_count() != structures.size()) {
    fail("parallel structure encoding must encode each input once");
  }
  if (unique_forward_thread_count() < 2U) {
    fail("N=4 structure encoding must use the pool before serial Phase 2");
  }

  require_algorithm_records_match(serial_sink.records, parallel_sink.records);
}

void test_engine_parallel_coords_encoding_matches_serial() {
  const std::vector<StructureFixture> fixtures = make_structure_fixtures();
  const std::vector<hiko_api::CoordsInputView> coords =
      make_coords_views(fixtures);

  hiko_api::AllVsAllCoordsRequest request{};
  request.coords = {coords.data(), coords.size()};

  reset_forward_tracking();
  const hiko_api::Engine serial_engine(engine_config(1));
  const auto serial = serial_engine.collect_all_vs_all(request);
  if (serial.status.code != hiko_u::StatusCode::Ok ||
      forward_call_count() != coords.size()) {
    fail("serial coords Engine all-vs-all must encode each input once");
  }

  reset_forward_tracking();
  const hiko_api::Engine parallel_engine(engine_config(4));
  const auto parallel = parallel_engine.collect_all_vs_all(request);
  if (parallel.status.code != hiko_u::StatusCode::Ok ||
      forward_call_count() != coords.size()) {
    fail("parallel coords Engine all-vs-all must encode each input once");
  }
  if (unique_forward_thread_count() < 2U) {
    fail("N=4 coords encoding must use the Engine pool before serial Phase 2");
  }

  require_api_records_match(serial.value, parallel.value);
}

void test_phase1_thread_policy_downscales_deterministically() {
  std::size_t workspace_bytes = 0;
  if (!hiko_ad::estimate_all_vs_all_structure_encoder_workspace_bytes(
          3, fast_descriptor(), workspace_bytes) ||
      workspace_bytes == 0U) {
    fail("structure encoder workspace byte estimate must be available");
  }

  if (hiko_ad::select_all_vs_all_phase1_thread_count_for_budget(
          8, 10, workspace_bytes, workspace_bytes * 8U) != 8U) {
    fail("phase1 policy must keep requested threads when budget allows");
  }
  if (hiko_ad::select_all_vs_all_phase1_thread_count_for_budget(
          8, 10, workspace_bytes, workspace_bytes * 3U) != 3U) {
    fail("phase1 policy must downscale to deterministic budget capacity");
  }
  if (hiko_ad::select_all_vs_all_phase1_thread_count_for_budget(
          8, 10, workspace_bytes, workspace_bytes - 1U) != 1U) {
    fail("phase1 policy must fall back to serial when one workspace does not fit");
  }
  if (hiko_ad::select_all_vs_all_phase1_thread_count_for_budget(
          8, 3, workspace_bytes, workspace_bytes * 8U) != 1U) {
    fail("phase1 policy must keep N<4 inputs serial");
  }
}

void test_auto_thread_policy_clamps_pair_workload() {
  if (hiko_ad::resolve_all_vs_all_auto_thread_count(0, 6, 12) != 3U) {
    fail("auto thread policy must clamp six pairs to three workers on hw=12");
  }
  if (hiko_ad::resolve_all_vs_all_auto_thread_count(4, 6, 12) != 4U) {
    fail("explicit thread policy must not clamp by pair workload");
  }
  if (hiko_ad::resolve_all_vs_all_auto_thread_count(0, 1000, 12) != 12U) {
    fail("auto thread policy must keep hardware count for large workloads");
  }
  if (hiko_ad::resolve_all_vs_all_auto_thread_count(0, 0, 12) != 1U) {
    fail("auto thread policy must return one worker for empty workloads");
  }
}

}  // namespace

int main() {
  test_algorithm_parallel_structure_encoding_matches_serial();
  test_engine_parallel_coords_encoding_matches_serial();
  test_phase1_thread_policy_downscales_deterministically();
  test_auto_thread_policy_clamps_pair_workload();
  return 0;
}

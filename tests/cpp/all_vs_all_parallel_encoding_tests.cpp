#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/encoding_scheduler.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/structure_embedding_cache.hpp>
#include <unistd.h>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cstring>
#include <limits>
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
bool g_inject_failures = false;

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
  if (g_inject_failures && base == 1.0F)
    return hiko_u::invalid_argument_status("earlier injected encoder error");
  if (g_inject_failures && base == 6.0F)
    return hiko_u::failed_precondition_status("later injected encoder error");
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

void test_weighted_encoding_ranges() {
  // Count-based ranges concentrate all long inputs in one worker.
  std::vector<std::size_t> lengths(32, 100);
  std::fill_n(lengths.begin(), 8, 1000);
  const auto cost = [&](std::size_t i) { return hiko_ad::mpnn_encoding_cost(lengths[i], 64); };
  const auto ranges = hiko_ad::partition_encoding_ranges(lengths.size(), 4, cost);
  long double maximum = 0;
  std::size_t next = 0;
  for (const auto range : ranges) {
    if (range.begin != next || range.end <= range.begin) fail("encoding ranges must be contiguous and nonempty");
    long double work = 0;
    for (std::size_t i = range.begin; i < range.end; ++i) work += cost(i);
    maximum = std::max(maximum, work);
    next = range.end;
  }
  if (next != lengths.size() || maximum >= 0.6L * 8 * cost(0))
    fail("weighted encoding must reduce a skewed count-partition tail");
  for (std::size_t n : {0U, 1U, 3U, 16U}) {
    for (std::size_t workers : {1U, 2U, 4U, 32U}) {
      const auto zero = hiko_ad::partition_encoding_ranges(n, workers, [](std::size_t) { return 0; });
      if (zero.size() != std::min(n, workers)) fail("small/empty encoding plan worker count");
      std::size_t covered = 0;
      for (const auto range : zero) {
        if (range.begin != covered || range.end <= range.begin || range.end > n)
          fail("zero-cost input coverage");
        covered = range.end;
      }
      if (covered != n) fail("zero-cost inputs must not disappear");
    }
  }
  const auto huge = hiko_ad::partition_encoding_ranges(10, 3, [](std::size_t) {
    return hiko_ad::mpnn_encoding_cost(std::numeric_limits<std::size_t>::max(),
                                      std::numeric_limits<std::size_t>::max());
  });
  if (huge.size() != 3 || huge.back().end != 10 || huge.front().end < 3)
    fail("encoding cost arithmetic must not overflow integer lengths");
  if (hiko_ad::mpnn_encoding_cost(5, 64) != 25 ||
      hiko_ad::mpnn_encoding_cost(100, 64) != 6400)
    fail("MPNN scheduling must account for active neighbors");
}

void test_weighted_encoding_error_order_and_recovery() {
  std::vector<StructureFixture> fixtures;
  for (std::size_t i = 0; i < 8; ++i)
    fixtures.push_back(make_structure_fixture(i < 4 ? 8 : 2, static_cast<float>(i)));
  const auto structures = make_structure_views(fixtures);
  hiko_md::Mpnn64Weights dummy_weights{};
  hiko::AllVsAllStructureRequest request{};
  request.structures = {structures.data(), structures.size()};
  request.descriptor = fast_descriptor();
  request.weights = &dummy_weights;
  hiko_ud::ThreadPool pool(4);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(4);
  g_inject_failures = true;
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    AlgorithmCollectingSink sink;
    const auto status = hiko::run_all_vs_all_structures(
        request, sink, &pool, 4, {workers.data(), workers.size()});
    if (status.code != hiko_u::StatusCode::InvalidArgument ||
        std::strcmp(status.detail, "earlier injected encoder error") || !sink.records.empty())
      fail("parallel encoding must report the earliest input error before emitting pairs");
    for (const auto& worker : workers)
      if (worker.encoder.has_mpnn_workspaces()) fail("failed encoding must release scratch");
  }
  g_inject_failures = false;
  reset_forward_tracking();
  AlgorithmCollectingSink recovered, serial;
  const auto status = hiko::run_all_vs_all_structures(
      request, recovered, &pool, 4, {workers.data(), workers.size()});
  if (!hiko_u::is_ok(status) || forward_call_count() != structures.size())
    fail("workers must recover and encode every uneven input once");
  if (!hiko_u::is_ok(hiko::run_all_vs_all_structures(request, serial))) fail("serial uneven encode");
  require_algorithm_records_match(serial.records, recovered.records);
}

class StreamingCheckSink final : public hiko_api::PairwiseResultSink {
 public:
  hiko_api::AllVsAllResult result;
  std::size_t stop_at = std::numeric_limits<std::size_t>::max();
  const std::thread::id caller = std::this_thread::get_id();
  hiko_u::Status receive(const hiko_api::PairwiseResultRecord& record) override {
    if (std::this_thread::get_id() != caller) fail("pair-list callbacks must run on caller thread");
    if (result.records.size() == stop_at) return hiko_u::unavailable_status("injected sink failure");
    result.records.push_back(record);
    return hiko_u::ok_status();
  }
};

template<class Request>
void check_streaming_request(Request request) {
  hiko_api::Engine serial(engine_config(1)), parallel(engine_config(4));
  for (const auto mode : {hiko_api::AlignmentMode::Hard, hiko_api::AlignmentMode::Both}) {
    request.options.mode = mode;
    const auto expected = serial.collect_pair_list(request);
    if (!expected.status.ok()) fail("serial pair-list fixture");
    StreamingCheckSink streamed;
    if (!parallel.pair_list(request, streamed).ok()) fail("parallel streaming pair-list");
    require_api_records_match(expected.value, streamed.result);
    for (std::size_t i = 0; i < expected.value.records.size(); ++i) {
      const auto& a = expected.value.records[i].result;
      const auto& b = streamed.result.records[i].result;
      if (a.metrics.soft_sw_score.valid != b.metrics.soft_sw_score.valid ||
          a.metrics.soft_sw_score.value != b.metrics.soft_sw_score.value ||
          a.warnings.size() != b.warnings.size()) fail("streaming scores/warnings changed");
      for (std::size_t j = 0; j < a.path.steps.size(); ++j) {
        const auto x = a.path.steps[j], y = b.path.steps[j];
        if (x.query_index != y.query_index || x.target_index != y.target_index ||
            x.residue_score != y.residue_score) fail("streaming traceback changed");
      }
    }
    StreamingCheckSink rejected;
    rejected.stop_at = 1030; // Cross the first bounded-batch boundary.
    const auto status = parallel.pair_list(request, rejected);
    if (status.code != hiko_u::StatusCode::Unavailable ||
        std::strcmp(status.detail, "injected sink failure") || rejected.result.records.size() != 1030)
      fail("streaming sink failure must stop and propagate unchanged");
    StreamingCheckSink recovered;
    if (!parallel.pair_list(request, recovered).ok()) fail("streaming must recover after sink failure");
    require_api_records_match(expected.value, recovered.result);
  }
  request.pairs.push_back({"missing", "a"});
  StreamingCheckSink invalid;
  if (parallel.pair_list(request, invalid).ok() || !invalid.result.records.empty())
    fail("ID validation must finish before streaming any results");
  request.pairs.clear();
  if (!parallel.pair_list(request, invalid).ok() || !invalid.result.records.empty())
    fail("empty pair-list must not invoke sink");
}

void require_cached_records_exact(const hiko_api::AllVsAllResult& expected,
                                  const hiko_api::AllVsAllResult& actual) {
  require_api_records_match(expected, actual);
  const hiko_u::MetricValue hiko_api::PairwiseMetrics::* metrics[] = {
      &hiko_api::PairwiseMetrics::soft_sw_score, &hiko_api::PairwiseMetrics::rmsd,
      &hiko_api::PairwiseMetrics::tm_score_query, &hiko_api::PairwiseMetrics::tm_score_target,
      &hiko_api::PairwiseMetrics::lddt, &hiko_api::PairwiseMetrics::lddt_byA,
      &hiko_api::PairwiseMetrics::lddt_byB, &hiko_api::PairwiseMetrics::lddt_aln,
      &hiko_api::PairwiseMetrics::coverage_query, &hiko_api::PairwiseMetrics::coverage_target,
      &hiko_api::PairwiseMetrics::coverage_mean, &hiko_api::PairwiseMetrics::identity,
      &hiko_api::PairwiseMetrics::coverage_byA, &hiko_api::PairwiseMetrics::coverage_byB,
      &hiko_api::PairwiseMetrics::ecs};
  for (std::size_t i = 0; i < expected.records.size(); ++i) {
    const auto& a = expected.records[i].result;
    const auto& b = actual.records[i].result;
    if (a.metrics.raw_sw_score != b.metrics.raw_sw_score) fail("cached raw score changed");
    for (const auto member : metrics) {
      const auto x = a.metrics.*member, y = b.metrics.*member;
      if (x.valid != y.valid || x.reason != y.reason ||
          (x.valid && x.value != y.value)) fail("cached geometry/score metric changed");
    }
    for (std::size_t j = 0; j < a.path.steps.size(); ++j) {
      const auto x = a.path.steps[j], y = b.path.steps[j];
      if (x.query_index != y.query_index || x.target_index != y.target_index ||
          x.residue_score != y.residue_score) fail("cached traceback changed");
    }
  }
}

void test_durable_structure_cache_routes() {
  std::string pattern = (std::filesystem::temp_directory_path() / "hikoboshi-engine-cache-XXXXXX").string();
  std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
  if (mkdtemp(name.data()) == nullptr) fail("cache fixture directory");
  const std::filesystem::path directory(name.data());
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
  } cleanup{directory};
  auto fixtures = make_structure_fixtures();
  auto views = make_structure_views(fixtures);
  const char* ids[] = {"a", "b", "c", "d"};
  for (std::size_t i = 0; i < views.size(); ++i) views[i].input_id = ids[i];
  hiko_api::PairListStructureRequest request{};
  request.structures = {views.data(), views.size()};
  for (std::size_t i = 0; i < 80; ++i) request.pairs.push_back({ids[i % 4], ids[(i + 1) % 4]});
  for (std::uint32_t threads : {1U, 4U}) {
    hikoboshi::io::DiskStructureEmbeddingCache cache;
    const auto identity = "stub-encoder/threads-" + std::to_string(threads);
    if (!cache.prepare(directory, identity).ok()) fail("prepare cache");
    auto config = engine_config(threads);
    hiko_api::Engine uncached(config);
    config.structure_embedding_cache = &cache;
    hiko_api::Engine cached(config);
    for (const auto mode : {hiko_api::AlignmentMode::Hard, hiko_api::AlignmentMode::Soft,
                            hiko_api::AlignmentMode::Both}) {
      request.options.mode = mode;
      const auto expected = uncached.collect_pair_list(request);
      if (!expected.status.ok()) fail("uncached fixture");
      reset_forward_tracking();
      const auto cold = cached.collect_pair_list(request);
      if (!cold.status.ok()) fail("cached fixture");
      const auto expected_calls = mode == hiko_api::AlignmentMode::Hard ? views.size() : 0;
      if (forward_call_count() != expected_calls) fail("cache encodes each miss once");
      require_cached_records_exact(expected.value, cold.value);
      hikoboshi::io::DiskStructureEmbeddingCache reopened;
      if (!reopened.prepare(directory, identity).ok()) fail("reopen cache");
      auto warm_config = config; warm_config.structure_embedding_cache = &reopened;
      hiko_api::Engine warm(warm_config);
      reset_forward_tracking();
      StreamingCheckSink output;
      if (!warm.pair_list(request, output).ok() || forward_call_count() != 0 ||
          reopened.statistics().hits != views.size()) fail("warm pair-list must skip every forward");
      require_cached_records_exact(expected.value, output.result);
      // Structure and coords all-vs-all must share the same cached encodings.
      hiko_api::AllVsAllStructureRequest all{}; all.structures = request.structures;
      all.options.mode = mode;
      hiko_api::AllVsAllCoordsRequest coords{};
      const auto cv = make_coords_views(fixtures); coords.coords = {cv.data(), cv.size()};
      coords.options = all.options;
      StreamingCheckSink expected_all, warm_all, warm_coords;
      if (!uncached.all_vs_all(all, expected_all).ok()) fail("uncached all-vs-all fixture");
      reset_forward_tracking();
      if (!warm.all_vs_all(all, warm_all).ok() || !warm.all_vs_all(coords, warm_coords).ok() ||
          forward_call_count() != 0) fail("warm all-vs-all must skip every forward");
      require_cached_records_exact(expected_all.result, warm_all.result);
      require_cached_records_exact(expected_all.result, warm_coords.result);
    }
    fixtures[0].coordinates[0] += 0.125F;
    reset_forward_tracking();
    const auto changed = cached.collect_pair_list(request);
    if (!changed.status.ok() || forward_call_count() != 1) fail("changed protein must reencode alone");
    const auto expected = uncached.collect_pair_list(request);
    if (!expected.status.ok()) fail("changed baseline");
    require_cached_records_exact(expected.value, changed.value);
    fixtures[0].coordinates[0] -= 0.125F;
  }
}

void test_pair_list_streaming_routes() {
  const auto fixtures = make_structure_fixtures();
  auto structures = make_structure_views(fixtures);
  auto coords = make_coords_views(fixtures);
  const std::vector<std::string> names = {"unused", "a", "b", "c"};
  std::vector<std::vector<hiko_u::ResidueMetadataView>> metadata(4);
  std::vector<std::vector<float>> values(4);
  std::vector<hiko_u::EmbeddingView> embeddings;
  for (std::size_t i = 0; i < 4; ++i) {
    structures[i].input_id = names[i];
    metadata[i].resize(structures[i].residue_count);
    for (auto& residue : metadata[i]) residue.source_id = names[i];
    coords[i].residues = {metadata[i].data(), metadata[i].size()};
    values[i].resize(structures[i].residue_count * 4, 1.0F + static_cast<float>(i));
    embeddings.push_back({structures[i].residue_count, 4, {values[i].data(), values[i].size()},
        structures[i].residue_codes, {metadata[i].data(), metadata[i].size()}});
  }
  std::vector<std::pair<std::string, std::string>> pairs;
  for (std::size_t i = 0; i < 2051; ++i)
    pairs.push_back(i % 3 == 0 ? std::make_pair(std::string{"c"}, std::string{"a"}) :
        i % 3 == 1 ? std::make_pair(std::string{"a"}, std::string{"c"}) :
                    std::make_pair(std::string{"b"}, std::string{"b"}));
  hiko_api::PairListStructureRequest sr{};
  sr.structures = {structures.data(), structures.size()}; sr.pairs = pairs;
  check_streaming_request(sr);
  hiko_api::PairListCoordsRequest cr{};
  cr.coords = {coords.data(), coords.size()}; cr.pairs = pairs;
  check_streaming_request(cr);
  hiko_api::PairListEmbeddingRequest er{};
  er.embeddings = {embeddings.data(), embeddings.size()}; er.pairs = pairs;
  check_streaming_request(er);
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
  test_durable_structure_cache_routes();
  test_pair_list_streaming_routes();
  test_weighted_encoding_ranges();
  test_weighted_encoding_error_order_and_recovery();
  test_algorithm_parallel_structure_encoding_matches_serial();
  test_engine_parallel_coords_encoding_matches_serial();
  test_phase1_thread_policy_downscales_deterministically();
  test_auto_thread_policy_clamps_pair_workload();
  return 0;
}

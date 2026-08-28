#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_md = hikoboshi::modules::detail;
namespace hiko_api = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;
namespace hiko_ud = hikoboshi::universal::detail;
namespace hiko_w = hikoboshi::weights;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_parallel_correctness_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double actual, double expected, double tolerance = 1.0e-6) {
  return std::fabs(actual - expected) <= tolerance;
}

struct EmbeddingFixture {
  std::vector<float> values;
  std::vector<char> codes;

  hiko_u::EmbeddingView view(std::size_t residue_count,
                          std::size_t dimension) const {
    return {residue_count,
            dimension,
            {values.data(), values.size()},
            {codes.data(), codes.size()},
            {nullptr, 0}};
  }
};

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
            "parallel_correctness_fixture",
            {},
            {}};
  }
};

StructureFixture make_structure_fixture(float offset) {
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

hiko::AllVsAllStructureRequest make_structure_request(
    const std::vector<hiko_u::StructureView>& structures) {
  hiko::AllVsAllStructureRequest request{};
  request.structures = {structures.data(), structures.size()};
  request.descriptor = {64, 64, 16, 3, 30.0F};
  request.weights = default_prepared_weights();
  return request;
}

std::vector<EmbeddingFixture> make_embedding_fixtures(
    std::size_t count,
    std::size_t residue_count,
    std::size_t dimension) {
  std::vector<EmbeddingFixture> fixtures;
  fixtures.reserve(count);
  constexpr char kResidues[] = {'A', 'C', 'D', 'E', 'F', 'G'};
  for (std::size_t item = 0; item < count; ++item) {
    EmbeddingFixture fixture{};
    fixture.values.resize(residue_count * dimension);
    fixture.codes.resize(residue_count);
    for (std::size_t residue = 0; residue < residue_count; ++residue) {
      fixture.codes[residue] = kResidues[(item + residue) %
                                         (sizeof(kResidues) /
                                          sizeof(kResidues[0]))];
      for (std::size_t dim = 0; dim < dimension; ++dim) {
        const float diagonal = residue == dim ? 0.25F : 0.0F;
        fixture.values[residue * dimension + dim] =
            0.1F * static_cast<float>((item + 1U) * (residue + 1U) +
                                      (dim + 1U)) +
            diagonal;
      }
    }
    fixtures.push_back(std::move(fixture));
  }
  return fixtures;
}

std::vector<hiko_u::EmbeddingView> make_views(
    const std::vector<EmbeddingFixture>& fixtures,
    std::size_t residue_count,
    std::size_t dimension) {
  std::vector<hiko_u::EmbeddingView> views;
  views.reserve(fixtures.size());
  for (const EmbeddingFixture& fixture : fixtures) {
    views.push_back(fixture.view(residue_count, dimension));
  }
  return views;
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
    fail("serial and parallel algorithm record counts differ");
  }
  for (std::size_t index = 0; index < serial.size(); ++index) {
    const hiko::PairwiseResultRecord& lhs = serial[index];
    const hiko::PairwiseResultRecord& rhs = parallel[index];
    if (lhs.query_index != rhs.query_index ||
        lhs.target_index != rhs.target_index) {
      fail("serial and parallel algorithm pair order differs");
    }
    if (!nearly_equal(lhs.result.metrics.raw_sw_score,
                      rhs.result.metrics.raw_sw_score) ||
        !nearly_equal(lhs.result.raw_sw_score, rhs.result.raw_sw_score)) {
      fail("serial and parallel algorithm scores differ");
    }
    if (lhs.result.path.aligned_pairs != rhs.result.path.aligned_pairs ||
        lhs.result.path.steps.size() != rhs.result.path.steps.size()) {
      fail("serial and parallel algorithm paths differ");
    }
    for (std::size_t step = 0; step < lhs.result.path.steps.size(); ++step) {
      const hiko_u::AlignmentStep& lhs_step = lhs.result.path.steps[step];
      const hiko_u::AlignmentStep& rhs_step = rhs.result.path.steps[step];
      if (lhs_step.query_index != rhs_step.query_index ||
          lhs_step.target_index != rhs_step.target_index ||
          !nearly_equal(lhs_step.residue_score, rhs_step.residue_score)) {
        fail("serial and parallel algorithm path steps differ");
      }
    }
  }
}

void require_api_records_match(const hiko_api::AllVsAllResult& serial,
                               const hiko_api::AllVsAllResult& parallel) {
  if (serial.records.size() != parallel.records.size()) {
    fail("serial and parallel API record counts differ");
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
      fail("serial and parallel API records differ");
    }
  }
}

void test_algorithm_serial_and_parallel_embedding_results_match() {
  constexpr std::size_t kInputCount = 10;
  constexpr std::size_t kResidueCount = 4;
  constexpr std::size_t kDimension = 4;
  const std::vector<EmbeddingFixture> fixtures =
      make_embedding_fixtures(kInputCount, kResidueCount, kDimension);
  const std::vector<hiko_u::EmbeddingView> embeddings =
      make_views(fixtures, kResidueCount, kDimension);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  AlgorithmCollectingSink serial_sink;
  const hiko_u::Status serial_status =
      hiko::run_all_vs_all_embeddings(request, serial_sink);
  if (serial_status.code != hiko_u::StatusCode::Ok) {
    fail("serial algorithm all-vs-all must return Ok");
  }

  hiko_ud::ThreadPool pool(4);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  AlgorithmCollectingSink parallel_sink;
  const hiko_u::Status parallel_status =
      hiko::run_all_vs_all_embeddings(request,
                                     parallel_sink,
                                     &pool,
                                     pool.thread_count(),
                                     {workers.data(), workers.size()});
  if (parallel_status.code != hiko_u::StatusCode::Ok) {
    fail("parallel algorithm all-vs-all must return Ok");
  }

  require_algorithm_records_match(serial_sink.records, parallel_sink.records);
}

void test_algorithm_serial_and_parallel_structure_results_match() {
  constexpr std::size_t kInputCount = 10;
  std::vector<StructureFixture> fixtures;
  fixtures.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    fixtures.push_back(make_structure_fixture(static_cast<float>(index)));
  }
  std::vector<hiko_u::StructureView> structures;
  structures.reserve(fixtures.size());
  for (const StructureFixture& fixture : fixtures) {
    structures.push_back(fixture.view());
  }

  const hiko::AllVsAllStructureRequest request =
      make_structure_request(structures);

  AlgorithmCollectingSink serial_sink;
  const hiko_u::Status serial_status =
      hiko::run_all_vs_all_structures(request, serial_sink);
  if (serial_status.code != hiko_u::StatusCode::Ok) {
    fail("serial structure algorithm all-vs-all must return Ok");
  }

  hiko_ud::ThreadPool pool(4);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  AlgorithmCollectingSink parallel_sink;
  const hiko_u::Status parallel_status =
      hiko::run_all_vs_all_structures(request,
                                     parallel_sink,
                                     &pool,
                                     pool.thread_count(),
                                     {workers.data(), workers.size()});
  if (parallel_status.code != hiko_u::StatusCode::Ok) {
    fail("parallel structure algorithm all-vs-all must return Ok");
  }

  require_algorithm_records_match(serial_sink.records, parallel_sink.records);
}

void test_engine_serial_and_parallel_embedding_results_match() {
  constexpr std::size_t kInputCount = 10;
  constexpr std::size_t kResidueCount = 4;
  constexpr std::size_t kDimension = 4;
  const std::vector<EmbeddingFixture> fixtures =
      make_embedding_fixtures(kInputCount, kResidueCount, kDimension);
  const std::vector<hiko_u::EmbeddingView> embeddings =
      make_views(fixtures, kResidueCount, kDimension);

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_api::EngineConfig serial_config{};
  serial_config.execution.thread_count = 1;
  const hiko_api::Engine serial_engine(serial_config);
  const auto serial = serial_engine.collect_all_vs_all(request);
  if (serial.status.code != hiko_u::StatusCode::Ok) {
    fail("serial Engine all-vs-all must return Ok");
  }

  hiko_api::EngineConfig parallel_config{};
  parallel_config.execution.thread_count = 4;
  const hiko_api::Engine parallel_engine(parallel_config);
  const auto parallel = parallel_engine.collect_all_vs_all(request);
  if (parallel.status.code != hiko_u::StatusCode::Ok) {
    fail("parallel Engine all-vs-all must return Ok");
  }

  require_api_records_match(serial.value, parallel.value);
}

void test_parallel_controls_keep_ineligible_runs_serial() {
  constexpr std::size_t kInputCount = 4;
  constexpr std::size_t kResidueCount = 2;
  constexpr std::size_t kDimension = 2;
  const std::vector<EmbeddingFixture> fixtures =
      make_embedding_fixtures(kInputCount, kResidueCount, kDimension);
  const std::vector<hiko_u::EmbeddingView> embeddings =
      make_views(fixtures, kResidueCount, kDimension);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_ud::ThreadPool pool(4);
  AlgorithmCollectingSink small_sink;
  const hiko_u::Status small_status =
      hiko::run_all_vs_all_embeddings(request,
                                     small_sink,
                                     &pool,
                                     pool.thread_count(),
                                     {nullptr, 0});
  if (small_status.code != hiko_u::StatusCode::Ok ||
      small_sink.records.size() !=
          hiko_ad::symmetric_pair_count(kInputCount, false)) {
    fail("small all-vs-all runs must stay serial without worker workspaces");
  }

  hiko_ud::ThreadPool serial_pool(1);
  AlgorithmCollectingSink forced_serial_sink;
  const hiko_u::Status forced_serial_status =
      hiko::run_all_vs_all_embeddings(request,
                                     forced_serial_sink,
                                     &serial_pool,
                                     serial_pool.thread_count(),
                                     {nullptr, 0});
  if (forced_serial_status.code != hiko_u::StatusCode::Ok ||
      forced_serial_sink.records.size() !=
          hiko_ad::symmetric_pair_count(kInputCount, false)) {
    fail("thread_count one must force serial execution");
  }
}

// --- ESM2 sequence-route serial==parallel equivalence (bp4) ---
//
// The sequence route parallelizes the per-protein ESM2 encode and (for
// all-vs-all) the per-pair dispatch under EngineConfig.execution.thread_count.
// Encode is per-protein independent and the deterministic pair-ordered sink
// emits in pair_index order, so a parallel run must be byte-identical to a
// serial run over the same fixture. These tests drive both modes through the
// public Engine and assert every emitted record column matches exactly. The
// file is TSan-instrumented, so a missing per-worker scratch buffer or an
// unguarded shared counter in the parallel encode also surfaces as a race.

std::int32_t aa_token(char c) noexcept {
  switch (c) {
    case 'A': return 0;   case 'R': return 1;   case 'N': return 2;
    case 'D': return 3;   case 'C': return 4;   case 'Q': return 5;
    case 'E': return 6;   case 'G': return 7;   case 'H': return 8;
    case 'I': return 9;   case 'L': return 10;  case 'K': return 11;
    case 'M': return 12;  case 'F': return 13;  case 'P': return 14;
    case 'S': return 15;  case 'T': return 16;  case 'W': return 17;
    case 'Y': return 18;  case 'V': return 19;  case 'B': return 20;
    case 'U': return 21;  case 'Z': return 22;  case 'O': return 23;
    default:  return 24;  // X / unknown
  }
}

std::vector<std::int32_t> tokenize(std::string_view aa) {
  std::vector<std::int32_t> tokens;
  tokens.reserve(aa.size());
  for (char c : aa) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    tokens.push_back(aa_token(c));
  }
  return tokens;
}

struct SequenceSample {
  const char* name;
  std::string_view sequence;
};

// Twelve short/medium peptides — enough that all-vs-all enumerates 66 pairs
// (>= kParallelPairThreshold so the parallel dispatch path engages) and the
// 12-protein encode crosses kAllVsAllParallelEncodingThreshold.
const std::vector<SequenceSample>& sequence_samples() {
  static const std::vector<SequenceSample> kSamples = {
      {"oxytocin",      "CYIQNCPLG"},
      {"vasopressin",   "CYFQNCPRG"},
      {"glucagon_n",    "HSQGTFTSDYSK"},
      {"insulin_a",     "GIVEQCCTSICSLYQLENYCN"},
      {"insulin_b",     "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"},
      {"glucagon_full", "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"},
      {"defensin",      "DCYCRIPACIAGERRYGTCIYQGRLWAFCC"},
      {"calcitonin_n",  "CSNLSTCVLGKLSQELHKLQTYPRTNTGSGTP"},
      {"growth_h_n",    "FPTIPLSRLFDNAMLRAHRLHQLAFDTYQEFEEAYIPKEQK"},
      {"acth_n",        "SYSMEHFRWGKPVGKKRRPVKVYP"},
      {"defensin_x2",
       "DCYCRIPACIAGERRYGTCIYQGRLWAFCCDCYCRIPACIAGERRYGTCIYQGRLWAFCC"},
      {"glucagon_x2",
       "HSQGTFTSDYSKYLDSRRAQDFVQWLMNTHSQGTFTSDYSKYLDSRRAQDFVQWLMNT"},
  };
  return kSamples;
}

std::vector<hiko_api::SequenceEntry> make_sequence_entries(
    const std::vector<SequenceSample>& samples,
    std::vector<std::vector<std::int32_t>>& tokens) {
  tokens.clear();
  tokens.reserve(samples.size());
  for (const SequenceSample& sample : samples) {
    tokens.push_back(tokenize(sample.sequence));
  }
  std::vector<hiko_api::SequenceEntry> entries(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    entries[i] = {std::string_view{samples[i].name},
                  {tokens[i].data(), tokens[i].size()}};
  }
  return entries;
}

class ApiRecordingSink final : public hiko_api::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko_api::PairwiseResultRecord& record) override {
    records.push_back(record);
    return hiko_u::ok_status();
  }
  std::vector<hiko_api::PairwiseResultRecord> records;
};

void require_metric_identical(const hiko_u::MetricValue& lhs,
                              const hiko_u::MetricValue& rhs,
                              const char* what) {
  if (lhs.value != rhs.value || lhs.valid != rhs.valid ||
      lhs.reason != rhs.reason) {
    std::fprintf(stderr,
                 "all_vs_all_parallel_correctness_tests: sequence-route "
                 "serial/parallel %s differs\n",
                 what);
    std::exit(1);
  }
}

// Byte-identical comparison of two sequence-route record streams: pair order,
// the raw SW score, the soft-pass metric columns, and every alignment path
// step are compared bit-exactly (not within a tolerance) because the encode
// is per-protein independent so serial and parallel must agree exactly.
void require_sequence_record_streams_identical(
    const std::vector<hiko_api::PairwiseResultRecord>& serial,
    const std::vector<hiko_api::PairwiseResultRecord>& parallel) {
  if (serial.size() != parallel.size()) {
    fail("sequence-route serial/parallel record counts differ");
  }
  for (std::size_t index = 0; index < serial.size(); ++index) {
    const hiko_api::PairwiseResultRecord& lhs = serial[index];
    const hiko_api::PairwiseResultRecord& rhs = parallel[index];
    if (lhs.query_index != rhs.query_index ||
        lhs.target_index != rhs.target_index) {
      fail("sequence-route serial/parallel pair order differs");
    }
    if (lhs.result.metrics.raw_sw_score != rhs.result.metrics.raw_sw_score) {
      fail("sequence-route serial/parallel raw_sw_score differs");
    }
    require_metric_identical(lhs.result.metrics.soft_sw_score,
                             rhs.result.metrics.soft_sw_score, "soft_sw_score");
    const hiko_u::AlignmentPath& lp = lhs.result.path;
    const hiko_u::AlignmentPath& rp = rhs.result.path;
    if (lp.aligned_pairs != rp.aligned_pairs ||
        lp.query_start != rp.query_start || lp.query_end != rp.query_end ||
        lp.target_start != rp.target_start ||
        lp.target_end != rp.target_end ||
        lp.steps.size() != rp.steps.size()) {
      fail("sequence-route serial/parallel alignment paths differ");
    }
    for (std::size_t step = 0; step < lp.steps.size(); ++step) {
      if (lp.steps[step].query_index != rp.steps[step].query_index ||
          lp.steps[step].target_index != rp.steps[step].target_index ||
          lp.steps[step].residue_score != rp.steps[step].residue_score) {
        fail("sequence-route serial/parallel path steps differ");
      }
    }
  }
}

void test_sequence_all_vs_all_serial_and_parallel_match() {
  const hiko_u::Result<hiko_w::PackageHandle> package =
      hiko_w::default_esm2_8m_package();
  if (package.status.code != hiko_u::StatusCode::Ok) {
    fail("Hikoboshi-ESM2-8M package handle must be resolvable");
  }

  std::vector<std::vector<std::int32_t>> tokens;
  const std::vector<hiko_api::SequenceEntry> entries =
      make_sequence_entries(sequence_samples(), tokens);
  hiko_api::AllVsAllSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};

  hiko_api::EngineConfig serial_config{};
  serial_config.package = package.value;
  serial_config.execution.thread_count = 1;
  const hiko_api::Engine serial_engine(serial_config);
  ApiRecordingSink serial_sink;
  if (serial_engine.all_vs_all(request, serial_sink).code !=
      hiko_u::StatusCode::Ok) {
    fail("serial sequence all-vs-all must return Ok");
  }

  hiko_api::EngineConfig parallel_config{};
  parallel_config.package = package.value;
  parallel_config.execution.thread_count = 4;
  const hiko_api::Engine parallel_engine(parallel_config);
  ApiRecordingSink parallel_sink;
  if (parallel_engine.all_vs_all(request, parallel_sink).code !=
      hiko_u::StatusCode::Ok) {
    fail("parallel sequence all-vs-all must return Ok");
  }

  require_sequence_record_streams_identical(serial_sink.records,
                                            parallel_sink.records);
}

void test_pair_list_sequence_serial_and_parallel_match() {
  const hiko_u::Result<hiko_w::PackageHandle> package =
      hiko_w::default_esm2_8m_package();
  if (package.status.code != hiko_u::StatusCode::Ok) {
    fail("Hikoboshi-ESM2-8M package handle must be resolvable");
  }

  const std::vector<SequenceSample>& kSamples = sequence_samples();
  std::vector<std::vector<std::int32_t>> tokens;
  const std::vector<hiko_api::SequenceEntry> entries =
      make_sequence_entries(kSamples, tokens);

  // A pair list that references every one of the 12 proteins (so the encode-
  // once pass crosses the parallel-encode threshold) without listing every
  // i<j combination.
  std::vector<std::pair<std::string, std::string>> pairs;
  for (std::size_t i = 0; i + 1 < kSamples.size(); ++i) {
    pairs.emplace_back(kSamples[i].name, kSamples[i + 1].name);
  }
  pairs.emplace_back(kSamples[0].name, kSamples[kSamples.size() - 1].name);

  hiko_api::PairListSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};
  request.pairs = pairs;

  hiko_api::EngineConfig serial_config{};
  serial_config.package = package.value;
  serial_config.execution.thread_count = 1;
  const hiko_api::Engine serial_engine(serial_config);
  const hiko_u::Result<hiko_api::AllVsAllResult> serial =
      serial_engine.collect_pair_list(request);
  if (serial.status.code != hiko_u::StatusCode::Ok) {
    fail("serial pair-list sequence must return Ok");
  }

  hiko_api::EngineConfig parallel_config{};
  parallel_config.package = package.value;
  parallel_config.execution.thread_count = 4;
  const hiko_api::Engine parallel_engine(parallel_config);
  const hiko_u::Result<hiko_api::AllVsAllResult> parallel =
      parallel_engine.collect_pair_list(request);
  if (parallel.status.code != hiko_u::StatusCode::Ok) {
    fail("parallel pair-list sequence must return Ok");
  }

  if (serial.value.records.size() != pairs.size()) {
    fail("pair-list must emit one record per input pair");
  }
  require_sequence_record_streams_identical(serial.value.records,
                                            parallel.value.records);
}

}  // namespace

int main() {
  test_algorithm_serial_and_parallel_embedding_results_match();
  test_algorithm_serial_and_parallel_structure_results_match();
  test_engine_serial_and_parallel_embedding_results_match();
  test_parallel_controls_keep_ineligible_runs_serial();
  test_sequence_all_vs_all_serial_and_parallel_match();
  test_pair_list_sequence_serial_and_parallel_match();
  return 0;
}

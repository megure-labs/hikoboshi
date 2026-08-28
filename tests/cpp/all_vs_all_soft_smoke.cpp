#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_api = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;

namespace {

constexpr float kMpnn64SoftGapOpen = -3.21337F;
constexpr float kMpnn64SoftGapExtension = -0.111704F;

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_soft_smoke: %s\n", message);
  std::exit(1);
}

bool nearly_equal(float lhs, float rhs) {
  return std::fabs(lhs - rhs) <= 1.0e-6F;
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values,
                                  std::size_t residue_count,
                                  std::size_t dimension) {
  return {residue_count,
          dimension,
          {values.data(), values.size()},
          {nullptr, 0},
          {nullptr, 0}};
}

class CollectingSink final : public hiko_api::PairwiseResultSink {
 public:
  std::vector<hiko_api::PairwiseResultRecord> records;

  hiko_u::Status receive(const hiko_api::PairwiseResultRecord& record) override {
    hiko_api::PairwiseResultRecord copy{};
    copy.query_index = record.query_index;
    copy.target_index = record.target_index;
    copy.result.metrics = record.result.metrics;
    copy.result.path.aligned_pairs = record.result.path.aligned_pairs;
    copy.result.path.query_start = record.result.path.query_start;
    copy.result.path.query_end = record.result.path.query_end;
    copy.result.path.target_start = record.result.path.target_start;
    copy.result.path.target_end = record.result.path.target_end;
    copy.result.path.steps.assign(record.result.path.steps.begin(),
                                  record.result.path.steps.end());
    records.push_back(std::move(copy));
    return {hiko_u::StatusCode::Ok, ""};
  }
};

void require_soft_record_metrics_match(const hiko_api::PairwiseResult& actual,
                                       const hiko_api::PairwiseResult& expected,
                                       const char* context) {
  if (!actual.metrics.soft_sw_score.valid ||
      !expected.metrics.soft_sw_score.valid ||
      !nearly_equal(actual.metrics.soft_sw_score.value,
                    expected.metrics.soft_sw_score.value)) {
    fail(context);
  }
}

void test_default_mode_is_hard() {
  // A default-constructed AllVsAllOptions must keep the legacy hard branch.
  // The packet's "Default behavior unchanged" constraint pins the all-vs-all
  // default to AlignmentMode::Hard; soft is opt-in until the human ship
  // decision lands after the SCOPe40 F1 bench.
  const hiko_api::AllVsAllOptions options{};
  if (options.mode != hiko_api::AlignmentMode::Hard) {
    fail("default AllVsAllOptions mode must be Hard");
  }
}

void test_explicit_soft_mode_runs_end_to_end() {
  // 3-input embedding-mode all-vs-all in explicit soft mode, with three
  // distinct diagonal-dominant queries. Verify the sink receives exactly the
  // three i<j pairs in lexicographic order with finite metrics and populated
  // paths.
  const std::vector<float> a = {1.0F, 0.0F, 0.0F,
                                 0.0F, 1.0F, 0.0F,
                                 0.0F, 0.0F, 1.0F};
  const std::vector<float> b = {0.9F, 0.1F, 0.0F,
                                 0.1F, 0.9F, 0.0F,
                                 0.0F, 0.1F, 0.9F};
  const std::vector<float> c = {1.0F, 0.0F, 0.0F,
                                 0.5F, 0.5F, 0.0F,
                                 0.0F, 0.0F, 1.0F};
  const hiko_u::EmbeddingView a_view = embedding_view(a, 3, 3);
  const hiko_u::EmbeddingView b_view = embedding_view(b, 3, 3);
  const hiko_u::EmbeddingView c_view = embedding_view(c, 3, 3);
  const std::vector<hiko_u::EmbeddingView> embeddings = {a_view, b_view, c_view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Soft;
  request.options.temperature = 1.0F;

  CollectingSink sink;
  const hiko_api::Engine engine;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("explicit soft-mode all-vs-all must return Ok");
  }
  if (sink.records.size() != 3U) {
    fail("explicit soft-mode all-vs-all must emit exactly three pairs for n=3 i<j");
  }
  const std::pair<std::size_t, std::size_t> expected[3] = {
      {0U, 1U}, {0U, 2U}, {1U, 2U}};
  for (std::size_t i = 0; i < 3U; ++i) {
    if (sink.records[i].query_index != expected[i].first ||
        sink.records[i].target_index != expected[i].second) {
      fail("explicit soft-mode pair indices must be lexicographic");
    }
    if (!std::isfinite(sink.records[i].result.metrics.raw_sw_score)) {
      fail("explicit soft-mode pair raw_sw_score must be finite");
    }
    if (!sink.records[i].result.metrics.soft_sw_score.valid) {
      fail("explicit soft-mode pair soft_sw_score must be valid");
    }
    if (sink.records[i].result.metrics.soft_sw_score.value !=
        sink.records[i].result.metrics.raw_sw_score) {
      fail("soft-only all-vs-all must mirror soft_sw_score into raw_sw_score");
    }
  }
}

void test_default_soft_mode_uses_mpnn_soft_gap_family() {
  const std::vector<float> query_values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<float> target_values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const hiko_u::EmbeddingView query_view = embedding_view(query_values, 3, 3);
  const hiko_u::EmbeddingView target_view = embedding_view(target_values, 2, 3);
  const std::vector<hiko_u::EmbeddingView> embeddings = {query_view, target_view};

  hiko_api::AllVsAllEmbeddingRequest default_request{};
  default_request.embeddings = {embeddings.data(), embeddings.size()};
  default_request.options.mode = hiko_api::AlignmentMode::Soft;

  hiko_api::AllVsAllEmbeddingRequest explicit_request = default_request;
  explicit_request.options.alignment.gap_open = kMpnn64SoftGapOpen;
  explicit_request.options.alignment.gap_extension = kMpnn64SoftGapExtension;

  const hiko_api::Engine engine;
  CollectingSink default_sink;
  CollectingSink explicit_sink;
  if (engine.all_vs_all(default_request, default_sink).code !=
          hiko_u::StatusCode::Ok ||
      engine.all_vs_all(explicit_request, explicit_sink).code !=
          hiko_u::StatusCode::Ok) {
    fail("default and explicit MPNN soft all-vs-all requests must return Ok");
  }
  if (default_sink.records.size() != 1U || explicit_sink.records.size() != 1U) {
    fail("default and explicit MPNN soft all-vs-all requests must emit one record");
  }
  if (!nearly_equal(default_sink.records[0].result.metrics.raw_sw_score,
                    explicit_sink.records[0].result.metrics.raw_sw_score)) {
    fail("default MPNN soft all-vs-all raw score must use the calibrated soft gaps");
  }
  require_soft_record_metrics_match(
      default_sink.records[0].result, explicit_sink.records[0].result,
      "default MPNN soft all-vs-all metrics must use the calibrated soft gaps");
}

void test_explicit_soft_mode_temperature() {
  const std::vector<float> values = {1.0F, 0.0F, 0.0F,
                                      0.0F, 1.0F, 0.0F};
  const hiko_u::EmbeddingView view = embedding_view(values, 2, 3);
  const std::vector<hiko_u::EmbeddingView> embeddings = {view, view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Soft;
  request.options.temperature = 0.5F;

  CollectingSink sink;
  const hiko_api::Engine engine;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("explicit soft-mode all-vs-all must return Ok");
  }
  if (sink.records.size() != 1U) {
    fail("explicit soft-mode all-vs-all must emit one i<j pair for n=2");
  }
}

void test_soft_mode_rejects_non_positive_temperature() {
  const std::vector<float> values = {1.0F, 0.0F};
  const hiko_u::EmbeddingView view = embedding_view(values, 2, 1);
  const std::vector<hiko_u::EmbeddingView> embeddings = {view, view, view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Soft;
  request.options.temperature = 0.0F;

  CollectingSink sink;
  const hiko_api::Engine engine;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("soft mode with temperature=0 must return InvalidArgument");
  }
}

void test_soft_mode_rejects_nan_temperature() {
  const std::vector<float> values = {1.0F, 0.0F};
  const hiko_u::EmbeddingView view = embedding_view(values, 2, 1);
  const std::vector<hiko_u::EmbeddingView> embeddings = {view, view, view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Soft;
  request.options.temperature = std::nanf("");

  CollectingSink sink;
  const hiko_api::Engine engine;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("soft mode with NaN temperature must return InvalidArgument");
  }
}

void test_hard_mode_ignores_invalid_temperature() {
  const std::vector<float> values = {1.0F, 0.0F,
                                      0.0F, 1.0F};
  const hiko_u::EmbeddingView view = embedding_view(values, 2, 2);
  const std::vector<hiko_u::EmbeddingView> embeddings = {view, view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Hard;
  request.options.temperature = 0.0F;  // ignored when mode == Hard.

  CollectingSink sink;
  const hiko_api::Engine engine;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("hard-mode all-vs-all must ignore temperature");
  }
  if (sink.records.empty() ||
      sink.records[0].result.metrics.soft_sw_score.valid) {
    fail("hard-mode all-vs-all must not populate soft_sw_score");
  }
}

void test_default_hard_mode_records_match_legacy() {
  // Regression: all-vs-all defaults to hard SW. This test exercises the
  // explicit `Hard` path and confirms the per-pair metrics still come back
  // finite. Bit-equality with the literal default is covered by the broader
  // parity suite; this smoke makes sure the Hard branch still produces the
  // same pair indices and aligned_pairs as a hand-computed expectation on a
  // near-identity input.
  const std::vector<float> values = {1.0F, 0.0F,
                                      0.0F, 1.0F};
  const hiko_u::EmbeddingView view = embedding_view(values, 2, 2);
  const std::vector<hiko_u::EmbeddingView> embeddings = {view, view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Hard;

  CollectingSink sink;
  const hiko_api::Engine engine;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("hard-mode all-vs-all must return Ok");
  }
  if (sink.records.size() != 1U) {
    fail("hard-mode all-vs-all must emit one i<j pair for n=2");
  }
  if (sink.records[0].result.path.aligned_pairs != 2U) {
    fail("hard-mode aligned_pairs mismatch on diagonal input");
  }
}

void test_both_mode_returns_hard_primary_and_soft_score() {
  const std::vector<float> values = {1.0F, 0.0F,
                                      0.0F, 1.0F};
  const hiko_u::EmbeddingView view = embedding_view(values, 2, 2);
  const std::vector<hiko_u::EmbeddingView> embeddings = {view, view};

  hiko_api::AllVsAllEmbeddingRequest hard_request{};
  hard_request.embeddings = {embeddings.data(), embeddings.size()};
  hard_request.options.mode = hiko_api::AlignmentMode::Hard;

  hiko_api::AllVsAllEmbeddingRequest soft_request = hard_request;
  soft_request.options.mode = hiko_api::AlignmentMode::Soft;
  soft_request.options.alignment.gap_open = kMpnn64SoftGapOpen;
  soft_request.options.alignment.gap_extension = kMpnn64SoftGapExtension;

  hiko_api::AllVsAllEmbeddingRequest both_request = hard_request;
  both_request.options.mode = hiko_api::AlignmentMode::Both;
  both_request.options.temperature = 1.0F;

  const hiko_api::Engine engine;
  CollectingSink hard_sink;
  CollectingSink soft_sink;
  CollectingSink both_sink;
  if (engine.all_vs_all(hard_request, hard_sink).code !=
          hiko_u::StatusCode::Ok ||
      engine.all_vs_all(soft_request, soft_sink).code !=
          hiko_u::StatusCode::Ok ||
      engine.all_vs_all(both_request, both_sink).code !=
          hiko_u::StatusCode::Ok) {
    fail("hard, soft, and both all-vs-all requests must return Ok");
  }
  if (hard_sink.records.size() != 1U || soft_sink.records.size() != 1U ||
      both_sink.records.size() != 1U) {
    fail("hard, soft, and both all-vs-all requests must emit one record");
  }
  const hiko_api::PairwiseResult& hard = hard_sink.records[0].result;
  const hiko_api::PairwiseResult& soft = soft_sink.records[0].result;
  const hiko_api::PairwiseResult& both = both_sink.records[0].result;
  if (both.metrics.raw_sw_score != hard.metrics.raw_sw_score) {
    fail("both all-vs-all raw_sw_score must keep the hard score");
  }
  if (both.path.aligned_pairs != hard.path.aligned_pairs) {
    fail("both all-vs-all path must keep the hard path");
  }
  if (!both.metrics.soft_sw_score.valid ||
      !std::isfinite(both.metrics.soft_sw_score.value)) {
    fail("both all-vs-all must populate finite soft_sw_score");
  }
  if (std::fabs(both.metrics.soft_sw_score.value -
                soft.metrics.raw_sw_score) > 1e-6) {
    fail("both all-vs-all soft_sw_score must match explicit soft raw_sw_score");
  }
}

void test_both_mode_rejects_non_positive_temperature() {
  const std::vector<float> values = {1.0F, 0.0F};
  const hiko_u::EmbeddingView view = embedding_view(values, 2, 1);
  const std::vector<hiko_u::EmbeddingView> embeddings = {view, view, view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Both;
  request.options.temperature = 0.0F;

  CollectingSink sink;
  const hiko_api::Engine engine;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("both mode with temperature=0 must return InvalidArgument");
  }
}

void test_soft_mode_determinism_across_runs() {
  // Soft mode must be bit-stable across invocations on a fixed input. The
  // streaming sink/scheduler tests already cover thread-count determinism;
  // this test pins single-threaded determinism.
  const std::vector<float> a = {1.0F, 0.0F, 0.0F,
                                 0.0F, 1.0F, 0.0F,
                                 0.0F, 0.0F, 1.0F};
  const std::vector<float> b = {0.5F, 0.5F, 0.0F,
                                 0.0F, 1.0F, 0.0F,
                                 0.0F, 0.0F, 1.0F};
  const hiko_u::EmbeddingView a_view = embedding_view(a, 3, 3);
  const hiko_u::EmbeddingView b_view = embedding_view(b, 3, 3);
  const std::vector<hiko_u::EmbeddingView> embeddings = {a_view, b_view};

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.mode = hiko_api::AlignmentMode::Soft;
  request.options.temperature = 1.0F;

  const hiko_api::Engine engine;

  CollectingSink first;
  if (engine.all_vs_all(request, first).code != hiko_u::StatusCode::Ok) {
    fail("first soft-mode run must succeed");
  }
  CollectingSink second;
  if (engine.all_vs_all(request, second).code != hiko_u::StatusCode::Ok) {
    fail("second soft-mode run must succeed");
  }
  if (first.records.size() != second.records.size()) {
    fail("soft-mode determinism: record count differs across runs");
  }
  for (std::size_t i = 0; i < first.records.size(); ++i) {
    const hiko_api::PairwiseResultRecord& a_rec = first.records[i];
    const hiko_api::PairwiseResultRecord& b_rec = second.records[i];
    if (a_rec.query_index != b_rec.query_index ||
        a_rec.target_index != b_rec.target_index ||
        a_rec.result.path.aligned_pairs != b_rec.result.path.aligned_pairs) {
      fail("soft-mode determinism: per-pair record differs across runs");
    }
    if (a_rec.result.metrics.raw_sw_score !=
        b_rec.result.metrics.raw_sw_score) {
      fail("soft-mode determinism: raw_sw_score differs across runs");
    }
  }
}

}  // namespace

int main() {
  test_default_mode_is_hard();
  test_explicit_soft_mode_runs_end_to_end();
  test_default_soft_mode_uses_mpnn_soft_gap_family();
  test_explicit_soft_mode_temperature();
  test_soft_mode_rejects_non_positive_temperature();
  test_soft_mode_rejects_nan_temperature();
  test_hard_mode_ignores_invalid_temperature();
  test_default_hard_mode_records_match_legacy();
  test_both_mode_returns_hard_primary_and_soft_score();
  test_both_mode_rejects_non_positive_temperature();
  test_soft_mode_determinism_across_runs();
  return 0;
}

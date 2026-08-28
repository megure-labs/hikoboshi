// CLI streaming summary parity tests for p46.
//
// Verifies that the new `hikoboshi::api::TsvStreamingAllVsAllSink` plus
// `hikoboshi::api::stream_all_vs_all` produces a TSV summary that is
// bit-identical to the legacy buffered path, where the legacy path collects
// every record via `Engine::collect_all_vs_all` (now `[[deprecated]]`) and
// writes the summary using the same column layout.
//
// The fixture set is intentionally tiny: this test is about parity of the
// streaming TSV writer against the buffered renderer at the API surface, not
// about scaling. Scaling evidence belongs to `bench/CLI_STREAMING_SUMMARY.md`.

#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "cli_all_vs_all_streaming_summary_tests: %s\n", message);
  std::exit(1);
}

constexpr int kSummaryDoublePrecision = 6;
constexpr int kSummaryMetricPrecision = 6;
constexpr const char* kMetricNotAvailable = "NA";

std::string format_double(double value) {
  std::ostringstream out;
  out << std::setprecision(kSummaryDoublePrecision) << value;
  return out.str();
}

std::string format_metric(hiko_u::MetricValue metric) {
  if (!metric.valid) {
    return kMetricNotAvailable;
  }
  int significant_digits = kSummaryMetricPrecision;
  if (significant_digits < 1) {
    significant_digits = 1;
  }
  if (significant_digits > std::numeric_limits<double>::max_digits10) {
    significant_digits = std::numeric_limits<double>::max_digits10;
  }
  std::ostringstream out;
  out << std::setprecision(significant_digits) << metric.value;
  return out.str();
}

hiko_u::MetricValue invalid_metric(hiko_u::MetricInvalidReason reason) noexcept {
  return {0.0, false, reason};
}

hiko_u::MetricValue valid_metric(double value) noexcept {
  return {value, true, hiko_u::MetricInvalidReason::None};
}

hiko_u::MetricValue sw_per_aligned_metric(
    const hiko::PairwiseResult& result) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return invalid_metric(hiko_u::MetricInvalidReason::ZeroDenominator);
  }
  return valid_metric(result.metrics.raw_sw_score /
                      static_cast<double>(result.path.aligned_pairs));
}

hiko_u::MetricValue sw_per_length_metric(
    const hiko::PairwiseResult& result,
    hiko_u::MetricValue coverage) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return invalid_metric(hiko_u::MetricInvalidReason::ZeroDenominator);
  }
  if (!coverage.valid) {
    return invalid_metric(coverage.reason);
  }
  return valid_metric((result.metrics.raw_sw_score * coverage.value) /
                      static_cast<double>(result.path.aligned_pairs));
}

void write_reference_header(std::ostream& out,
                            bool include_dual_score_schema) {
  out << "query_index\ttarget_index\tpair_id\traw_sw_score";
  if (include_dual_score_schema) {
    out << "\tsoft_sw_score"
        << "\tsw_per_query_len\tsw_per_target_len\tsw_per_aligned";
  }
  out << "\taligned_pairs"
      << "\tcoverage_query\tcoverage_target\tcoverage_mean\tidentity\trmsd"
      << "\ttm_score_query\ttm_score_target"
      << "\tlddt\tlddt_byA\tlddt_byB\tlddt_aln\tcoverage_byA\tcoverage_byB"
      << "\tecs\tfasta_path\tpdb_path\n";
}

// Reference renderer that mirrors `hikoboshi::cli::render_all_vs_all_summary`
// without dragging the CLI's full file IO and structure-loader stack into a
// pure API parity test. The test compares the output of this reference
// renderer against the new TsvStreamingAllVsAllSink to assert byte equality.
void reference_render(std::ostream& out,
                      const hiko::AllVsAllResult& result,
                      const std::vector<std::string>& pair_ids,
                      bool include_dual_score_schema = false) {
  write_reference_header(out, include_dual_score_schema);
  for (std::size_t index = 0; index < result.records.size(); ++index) {
    const hiko::PairwiseResultRecord& record = result.records[index];
    const hiko::PairwiseMetrics& metrics = record.result.metrics;
    const std::string& pair_id =
        index < pair_ids.size() ? pair_ids[index] : std::string{};
    out << record.query_index << '\t' << record.target_index << '\t' << pair_id
        << '\t' << format_double(metrics.raw_sw_score);
    if (include_dual_score_schema) {
      out << '\t' << format_metric(metrics.soft_sw_score) << '\t'
          << format_metric(sw_per_length_metric(record.result,
                                                metrics.coverage_query))
          << '\t'
          << format_metric(sw_per_length_metric(record.result,
                                                metrics.coverage_target))
          << '\t' << format_metric(sw_per_aligned_metric(record.result));
    }
    out << '\t' << record.result.path.aligned_pairs << '\t'
        << format_metric(metrics.coverage_query) << '\t'
        << format_metric(metrics.coverage_target) << '\t'
        << format_metric(metrics.coverage_mean) << '\t'
        << format_metric(metrics.identity) << '\t'
        << format_metric(metrics.rmsd) << '\t'
        << format_metric(metrics.tm_score_query) << '\t'
        << format_metric(metrics.tm_score_target) << '\t'
        << format_metric(metrics.lddt) << '\t'
        << format_metric(metrics.lddt_byA) << '\t'
        << format_metric(metrics.lddt_byB) << '\t'
        << format_metric(metrics.lddt_aln) << '\t'
        << format_metric(metrics.coverage_byA) << '\t'
        << format_metric(metrics.coverage_byB) << '\t'
        << format_metric(metrics.ecs) << "\t\t\n";
  }
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values) {
  return {values.size(),
          1,
          {values.data(), values.size()},
          {nullptr, 0},
          {nullptr, 0}};
}

std::vector<hiko_u::EmbeddingView> make_embedding_views(
    const std::vector<std::vector<float>>& storage) {
  std::vector<hiko_u::EmbeddingView> views;
  views.reserve(storage.size());
  for (const auto& values : storage) {
    views.push_back(embedding_view(values));
  }
  return views;
}

struct PairIdContext {
  const std::vector<std::string>* ids = nullptr;
};

std::string pair_id_callback(std::size_t query_index,
                             std::size_t target_index,
                             void* user_data) {
  if (user_data == nullptr) {
    return {};
  }
  auto* context = static_cast<PairIdContext*>(user_data);
  if (context->ids == nullptr || query_index >= context->ids->size() ||
      target_index >= context->ids->size()) {
    return {};
  }
  return (*context->ids)[query_index] + "__" + (*context->ids)[target_index];
}

void test_streaming_matches_buffered_embeddings() {
  // Embedding all-vs-all with pair_id callbacks empty (no callback) so the
  // reference renderer also emits empty pair_id strings; the rest of the
  // schema is exercised end-to-end.
  std::vector<std::vector<float>> storage;
  for (std::size_t index = 0; index < 6U; ++index) {
    storage.push_back(
        {static_cast<float>(index + 1U), static_cast<float>(index + 2U)});
  }
  // Each "embedding" is a single residue with two-dimensional data so the
  // values have a nontrivial dot product.
  std::vector<hiko_u::EmbeddingView> views;
  views.reserve(storage.size());
  for (const auto& row : storage) {
    views.push_back({1, row.size(), {row.data(), row.size()}, {nullptr, 0},
                     {nullptr, 0}});
  }

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.data(), views.size()};

  const hiko::Engine engine;

  // Buffered run via the (deprecated) collect helper.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  const hiko_u::Result<hiko::AllVsAllResult> collected =
      hiko::collect_all_vs_all(engine, request);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  if (collected.status.code != hiko_u::StatusCode::Ok) {
    fail("buffered all-vs-all must succeed for parity baseline");
  }

  std::ostringstream reference;
  reference_render(reference, collected.value, std::vector<std::string>{});

  std::ostringstream streamed;
  hiko::TsvStreamingAllVsAllSink::Callbacks callbacks{};
  hiko::TsvStreamingAllVsAllSink sink(streamed, callbacks);
  const hiko_u::Status status = hiko::stream_all_vs_all(engine, request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("streaming all-vs-all must succeed for parity comparison");
  }

  if (reference.str() != streamed.str()) {
    std::fprintf(stderr,
                 "reference vs streaming differ; reference=%zu bytes, "
                 "streaming=%zu bytes\n",
                 reference.str().size(), streamed.str().size());
    fail("streaming TSV output must be bit-identical to buffered output");
  }
}

void test_streaming_with_pair_id_callback() {
  // Adds a populated pair_id callback so both the reference renderer and the
  // streaming sink emit the same `i__j` style identifiers; the pair_id
  // column then carries non-empty content alongside the metric fields.
  std::vector<std::vector<float>> storage;
  for (std::size_t index = 0; index < 5U; ++index) {
    storage.push_back({static_cast<float>(index + 1U)});
  }
  const std::vector<hiko_u::EmbeddingView> views = make_embedding_views(storage);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.data(), views.size()};
  request.options.include_self = true;

  const hiko::Engine engine;

  std::vector<std::string> ids;
  ids.reserve(views.size());
  for (std::size_t index = 0; index < views.size(); ++index) {
    ids.push_back("input" + std::to_string(index));
  }

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  const hiko_u::Result<hiko::AllVsAllResult> collected =
      hiko::collect_all_vs_all(engine, request);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  if (collected.status.code != hiko_u::StatusCode::Ok) {
    fail("buffered include_self all-vs-all must succeed");
  }

  // Reference renderer using the same id-pair construction the CLI does.
  std::vector<std::string> pair_ids;
  pair_ids.reserve(collected.value.records.size());
  for (const hiko::PairwiseResultRecord& record : collected.value.records) {
    pair_ids.push_back(ids[record.query_index] + "__" +
                       ids[record.target_index]);
  }
  std::ostringstream reference;
  reference_render(reference, collected.value, pair_ids);

  std::ostringstream streamed;
  PairIdContext context{};
  context.ids = &ids;
  hiko::TsvStreamingAllVsAllSink::Callbacks callbacks{};
  callbacks.pair_id = &pair_id_callback;
  callbacks.user_data = &context;
  hiko::TsvStreamingAllVsAllSink sink(streamed, callbacks);
  const hiko_u::Status status = hiko::stream_all_vs_all(engine, request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("streaming include_self all-vs-all must succeed");
  }

  if (reference.str() != streamed.str()) {
    fail("streaming TSV pair_id output must match reference renderer");
  }
}

void test_streaming_writes_to_multiple_outputs() {
  // CLI writes to both stdout and the summary file; the sink handles this
  // by accepting a vector of std::ostream*. Verify both outputs receive the
  // same byte stream.
  std::vector<std::vector<float>> storage = {{1.0F}, {2.0F}, {3.0F}};
  const std::vector<hiko_u::EmbeddingView> views = make_embedding_views(storage);
  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.data(), views.size()};

  const hiko::Engine engine;
  std::ostringstream first;
  std::ostringstream second;
  hiko::TsvStreamingAllVsAllSink::Callbacks callbacks{};
  hiko::TsvStreamingAllVsAllSink sink(
      std::vector<std::ostream*>{&first, &second}, callbacks);
  const hiko_u::Status status = hiko::stream_all_vs_all(engine, request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("streaming to multiple outputs must succeed");
  }
  if (first.str() != second.str()) {
    fail("multiple-output streaming sink must duplicate bytes exactly");
  }
  if (first.str().empty()) {
    fail("multiple-output sink must emit content");
  }
}

void test_soft_streaming_matches_buffered_embeddings() {
  std::vector<std::vector<float>> storage = {
      {1.0F, 0.0F, 0.0F, 1.0F},
      {0.9F, 0.1F, 0.1F, 0.9F},
      {1.0F, 0.0F, 0.5F, 0.5F},
  };
  std::vector<hiko_u::EmbeddingView> views;
  views.reserve(storage.size());
  for (const auto& values : storage) {
    views.push_back({2, 2, {values.data(), values.size()}, {nullptr, 0},
                     {nullptr, 0}});
  }

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.data(), views.size()};
  request.options.mode = hiko::AlignmentMode::Soft;
  request.options.temperature = 1.0F;

  const hiko::Engine engine;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  const hiko_u::Result<hiko::AllVsAllResult> collected =
      hiko::collect_all_vs_all(engine, request);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  if (collected.status.code != hiko_u::StatusCode::Ok) {
    fail("buffered soft all-vs-all must succeed for parity baseline");
  }

  std::ostringstream reference;
  reference_render(reference, collected.value, std::vector<std::string>{},
                   true);

  std::ostringstream streamed;
  hiko::TsvStreamingAllVsAllSink::Callbacks callbacks{};
  callbacks.include_dual_score_schema = true;
  hiko::TsvStreamingAllVsAllSink sink(streamed, callbacks);
  const hiko_u::Status status = hiko::stream_all_vs_all(engine, request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("streaming soft all-vs-all must succeed for parity comparison");
  }

  if (reference.str() != streamed.str()) {
    fail("soft streaming TSV output must be bit-identical to buffered output");
  }
  if (streamed.str().find("soft_sw_score\tsw_per_query_len") ==
      std::string::npos) {
    fail("soft streaming TSV header must include the soft score schema");
  }
}

void test_streaming_zero_pair_count_emits_header_only() {
  // include_self=false on a single input emits zero records but the sink
  // still writes the header row up front so file-format consumers see a
  // valid TSV header even on empty inputs.
  std::vector<std::vector<float>> storage = {{1.0F}};
  const std::vector<hiko_u::EmbeddingView> views = make_embedding_views(storage);
  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.data(), views.size()};

  const hiko::Engine engine;
  std::ostringstream out;
  hiko::TsvStreamingAllVsAllSink::Callbacks callbacks{};
  hiko::TsvStreamingAllVsAllSink sink(out, callbacks);
  const hiko_u::Status status = hiko::stream_all_vs_all(engine, request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("streaming zero-pair run must succeed");
  }

  std::ostringstream header_only;
  hiko::TsvStreamingAllVsAllSink::write_header(header_only);
  if (out.str() != header_only.str()) {
    fail("zero-pair streaming output must equal header-only baseline");
  }
}

}  // namespace

int main() {
  test_streaming_matches_buffered_embeddings();
  test_streaming_with_pair_id_callback();
  test_streaming_writes_to_multiple_outputs();
  test_soft_streaming_matches_buffered_embeddings();
  test_streaming_zero_pair_count_emits_header_only();
  return 0;
}

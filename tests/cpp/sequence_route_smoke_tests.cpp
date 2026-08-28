// Sequence-route smoke tests for the hikoboshi-esm2-8m public API.
//
// Drives `Engine::encode`, `Engine::pairwise`, and `Engine::all_vs_all`
// with `EncodeSequenceRequest`, `PairwiseSequenceRequest`, and
// `AllVsAllSequenceRequest` against the embedded hikoboshi-esm2-8m package
// resolved through `hikoboshi::weights::default_esm2_8m_package`. The
// goal is to confirm the engine accepts the sequence_tokens input route,
// allocates an ESM2 workspace, runs the scalar encoder forward pass, and
// returns coherent metrics. Bit-tight parity vs the PyTorch reference is
// covered separately by `sequence_route_parity_tests.cpp`.

#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

constexpr float kEsm2HardGapOpen = -1.01982F;
constexpr float kEsm2HardGapExtension = +0.225736F;
constexpr float kEsm2SoftGapOpen = -6.72805F;
constexpr float kEsm2SoftGapExtension = -0.0159468F;

void fail(const char* tag) {
  std::fprintf(stderr, "sequence_route_smoke_tests: %s\n", tag);
  std::exit(1);
}

bool nearly_equal(float lhs, float rhs) {
  return std::fabs(lhs - rhs) <= 1.0e-6F;
}

void require_soft_metrics_match(const hiko::PairwiseResult& actual,
                                const hiko::PairwiseResult& expected,
                                const char* context) {
  if (!actual.metrics.soft_sw_score.valid ||
      !expected.metrics.soft_sw_score.valid ||
      !nearly_equal(actual.metrics.soft_sw_score.value,
                    expected.metrics.soft_sw_score.value)) {
    fail(context);
  }
}

hiko::Engine make_esm2_engine() {
  const hiko_u::Result<hiko_w::PackageHandle> package = hiko_w::default_esm2_8m_package();
  if (package.status.code != hiko_u::StatusCode::Ok) {
    fail("hikoboshi-esm2-8m package handle must be resolvable");
  }
  hiko::EngineConfig config{};
  config.package = package.value;
  return hiko::Engine{config};
}

// AAAAAA token sequence — six alanine residues. Token id 0 is A in the
// embedded ESM2-8M compacted vocab.
const std::vector<std::int32_t>& aaaaaa_tokens() {
  static const std::vector<std::int32_t> tokens(6, 0);
  return tokens;
}

const std::vector<std::int32_t>& diverse_tokens() {
  // ACDEFG residues — tokens 0,4,3,6,13,7 in the compacted vocab.
  static const std::vector<std::int32_t> tokens = {0, 4, 3, 6, 13, 7};
  return tokens;
}

const std::vector<std::int32_t>& short_tokens() {
  // ACDE residues — a shorter target so gap-family routing affects the soft
  // partition and consensus alignment.
  static const std::vector<std::int32_t> tokens = {0, 4, 3, 6};
  return tokens;
}

void test_encode_returns_nontrivial_embeddings() {
  const hiko::Engine engine = make_esm2_engine();
  hiko::EncodeSequenceRequest request{};
  request.token_ids = {aaaaaa_tokens().data(), aaaaaa_tokens().size()};
  const auto result = engine.encode(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("ESM2 sequence encode must succeed");
  }
  if (result.value.embedding.residue_count != aaaaaa_tokens().size()) {
    fail("encode result residue_count must match input token count");
  }
  if (result.value.embedding.dimension != 320U) {
    fail("encode result dimension must be 320 for hikoboshi-esm2-8m");
  }
  const std::size_t expected_size =
      aaaaaa_tokens().size() * 320U;
  if (result.value.embedding.values.size() != expected_size) {
    fail("encode embedding buffer length must equal residues*320");
  }
  // Confirm the encoder wrote real values (not all zeros).
  bool seen_nonzero = false;
  for (float v : result.value.embedding.values) {
    if (std::abs(v) > 1.0e-6F) {
      seen_nonzero = true;
      break;
    }
  }
  if (!seen_nonzero) {
    fail("ESM2 encode embeddings should not be all-zero");
  }
}

void test_pairwise_self_self_is_identity_path() {
  const hiko::Engine engine = make_esm2_engine();
  hiko::PairwiseSequenceRequest request{};
  request.query_token_ids = {diverse_tokens().data(), diverse_tokens().size()};
  request.target_token_ids = {diverse_tokens().data(), diverse_tokens().size()};
  request.mode = hiko::AlignmentMode::Hard;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("ESM2 self-self pairwise must succeed");
  }
  if (result.value.path.aligned_pairs != diverse_tokens().size()) {
    fail("ESM2 self-self pairwise must align every residue under hard SW");
  }
  if (!(result.value.metrics.raw_sw_score > 0.0)) {
    fail("ESM2 self-self pairwise raw_sw_score must be positive");
  }
}

void test_pairwise_hard_defaults_use_esm2_annealed_gap_family() {
  const hiko::Engine engine = make_esm2_engine();
  hiko::PairwiseSequenceRequest default_request{};
  default_request.query_token_ids = {diverse_tokens().data(),
                                     diverse_tokens().size()};
  default_request.target_token_ids = {short_tokens().data(),
                                      short_tokens().size()};
  default_request.mode = hiko::AlignmentMode::Hard;

  hiko::PairwiseSequenceRequest explicit_request = default_request;
  explicit_request.alignment.gap_open = kEsm2HardGapOpen;
  explicit_request.alignment.gap_extension = kEsm2HardGapExtension;

  const auto default_result = engine.pairwise(default_request);
  const auto explicit_result = engine.pairwise(explicit_request);
  if (default_result.status.code != hiko_u::StatusCode::Ok ||
      explicit_result.status.code != hiko_u::StatusCode::Ok) {
    fail("ESM2 default and explicit hard pairwise requests must succeed");
  }
  if (!nearly_equal(default_result.value.metrics.raw_sw_score,
                    explicit_result.value.metrics.raw_sw_score) ||
      default_result.value.path.aligned_pairs !=
          explicit_result.value.path.aligned_pairs) {
    fail("ESM2 default hard pairwise must use the annealed hard gaps");
  }
}

void test_pairwise_soft_defaults_use_esm2_soft_gap_family() {
  const hiko::Engine engine = make_esm2_engine();
  hiko::PairwiseSequenceRequest default_request{};
  default_request.query_token_ids = {diverse_tokens().data(),
                                     diverse_tokens().size()};
  default_request.target_token_ids = {short_tokens().data(),
                                      short_tokens().size()};
  default_request.mode = hiko::AlignmentMode::Soft;

  hiko::PairwiseSequenceRequest explicit_request = default_request;
  explicit_request.alignment.gap_open = kEsm2SoftGapOpen;
  explicit_request.alignment.gap_extension = kEsm2SoftGapExtension;

  const auto default_result = engine.pairwise(default_request);
  const auto explicit_result = engine.pairwise(explicit_request);
  if (default_result.status.code != hiko_u::StatusCode::Ok ||
      explicit_result.status.code != hiko_u::StatusCode::Ok) {
    fail("ESM2 default and explicit soft pairwise requests must succeed");
  }
  if (!nearly_equal(default_result.value.metrics.raw_sw_score,
                    explicit_result.value.metrics.raw_sw_score)) {
    fail("ESM2 default soft pairwise raw score must use the recovered soft gaps");
  }
  require_soft_metrics_match(
      default_result.value, explicit_result.value,
      "ESM2 default soft pairwise metrics must use the recovered soft gaps");
}

void test_pairwise_both_uses_hard_primary_and_esm2_soft_gap_family() {
  const hiko::Engine engine = make_esm2_engine();
  hiko::PairwiseSequenceRequest hard_request{};
  hard_request.query_token_ids = {diverse_tokens().data(),
                                  diverse_tokens().size()};
  hard_request.target_token_ids = {short_tokens().data(), short_tokens().size()};
  hard_request.mode = hiko::AlignmentMode::Hard;

  hiko::PairwiseSequenceRequest soft_request = hard_request;
  soft_request.mode = hiko::AlignmentMode::Soft;
  soft_request.alignment.gap_open = kEsm2SoftGapOpen;
  soft_request.alignment.gap_extension = kEsm2SoftGapExtension;

  hiko::PairwiseSequenceRequest both_request = hard_request;
  both_request.mode = hiko::AlignmentMode::Both;

  const auto hard_result = engine.pairwise(hard_request);
  const auto soft_result = engine.pairwise(soft_request);
  const auto both_result = engine.pairwise(both_request);
  if (hard_result.status.code != hiko_u::StatusCode::Ok ||
      soft_result.status.code != hiko_u::StatusCode::Ok ||
      both_result.status.code != hiko_u::StatusCode::Ok) {
    fail("ESM2 hard, explicit soft, and both pairwise requests must succeed");
  }
  if (!nearly_equal(both_result.value.metrics.raw_sw_score,
                    hard_result.value.metrics.raw_sw_score)) {
    fail("ESM2 both pairwise raw score must keep the hard score");
  }
  if (both_result.value.path.aligned_pairs !=
      hard_result.value.path.aligned_pairs) {
    fail("ESM2 both pairwise path must keep the hard path");
  }
  require_soft_metrics_match(
      both_result.value, soft_result.value,
      "ESM2 both pairwise soft metrics must match explicit recovered soft gaps");
}

class RecordingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    records.push_back(record);
    return {hiko_u::StatusCode::Ok, ""};
  }
  std::vector<hiko::PairwiseResultRecord> records;
};

void test_all_vs_all_emits_one_record_per_pair() {
  const hiko::Engine engine = make_esm2_engine();

  static const std::vector<std::int32_t> seq_a = {0, 0, 0, 0};
  static const std::vector<std::int32_t> seq_b = {0, 4, 3, 6};
  static const std::vector<std::int32_t> seq_c = {7, 9, 11, 13};
  static const std::vector<std::int32_t> seq_d = {15, 16, 17, 18};

  std::vector<hiko::SequenceEntry> entries(4);
  entries[0] = {"seq_a", {seq_a.data(), seq_a.size()}};
  entries[1] = {"seq_b", {seq_b.data(), seq_b.size()}};
  entries[2] = {"seq_c", {seq_c.data(), seq_c.size()}};
  entries[3] = {"seq_d", {seq_d.data(), seq_d.size()}};

  hiko::AllVsAllSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};

  RecordingSink sink{};
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("ESM2 all-vs-all sequence request must succeed");
  }
  // Six pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
  if (sink.records.size() != 6U) {
    fail("ESM2 all-vs-all sequence must emit 6 records for 4 sequences");
  }
  for (const auto& record : sink.records) {
    if (record.query_index >= 4U || record.target_index >= 4U) {
      fail("all-vs-all record indices must reference the input span");
    }
  }
}

void test_all_vs_all_soft_defaults_use_esm2_soft_gap_family() {
  const hiko::Engine engine = make_esm2_engine();
  std::vector<hiko::SequenceEntry> entries(2);
  entries[0] = {"query", {diverse_tokens().data(), diverse_tokens().size()}};
  entries[1] = {"target", {short_tokens().data(), short_tokens().size()}};

  hiko::AllVsAllSequenceRequest default_request{};
  default_request.sequences = {entries.data(), entries.size()};
  default_request.options.mode = hiko::AlignmentMode::Soft;

  hiko::AllVsAllSequenceRequest explicit_request = default_request;
  explicit_request.options.alignment.gap_open = kEsm2SoftGapOpen;
  explicit_request.options.alignment.gap_extension = kEsm2SoftGapExtension;

  RecordingSink default_sink{};
  RecordingSink explicit_sink{};
  if (engine.all_vs_all(default_request, default_sink).code !=
          hiko_u::StatusCode::Ok ||
      engine.all_vs_all(explicit_request, explicit_sink).code !=
          hiko_u::StatusCode::Ok) {
    fail("ESM2 default and explicit soft all-vs-all requests must succeed");
  }
  if (default_sink.records.size() != 1U || explicit_sink.records.size() != 1U) {
    fail("ESM2 default and explicit soft all-vs-all must emit one record");
  }
  if (!nearly_equal(default_sink.records[0].result.metrics.raw_sw_score,
                    explicit_sink.records[0].result.metrics.raw_sw_score)) {
    fail("ESM2 default soft all-vs-all raw score must use the recovered soft gaps");
  }
  require_soft_metrics_match(
      default_sink.records[0].result, explicit_sink.records[0].result,
      "ESM2 default soft all-vs-all metrics must use the recovered soft gaps");
}

void test_all_vs_all_both_uses_hard_primary_and_esm2_soft_gap_family() {
  const hiko::Engine engine = make_esm2_engine();
  std::vector<hiko::SequenceEntry> entries(2);
  entries[0] = {"query", {diverse_tokens().data(), diverse_tokens().size()}};
  entries[1] = {"target", {short_tokens().data(), short_tokens().size()}};

  hiko::AllVsAllSequenceRequest hard_request{};
  hard_request.sequences = {entries.data(), entries.size()};
  hard_request.options.mode = hiko::AlignmentMode::Hard;

  hiko::AllVsAllSequenceRequest soft_request = hard_request;
  soft_request.options.mode = hiko::AlignmentMode::Soft;
  soft_request.options.alignment.gap_open = kEsm2SoftGapOpen;
  soft_request.options.alignment.gap_extension = kEsm2SoftGapExtension;

  hiko::AllVsAllSequenceRequest both_request = hard_request;
  both_request.options.mode = hiko::AlignmentMode::Both;

  RecordingSink hard_sink{};
  RecordingSink soft_sink{};
  RecordingSink both_sink{};
  if (engine.all_vs_all(hard_request, hard_sink).code != hiko_u::StatusCode::Ok ||
      engine.all_vs_all(soft_request, soft_sink).code != hiko_u::StatusCode::Ok ||
      engine.all_vs_all(both_request, both_sink).code != hiko_u::StatusCode::Ok) {
    fail("ESM2 hard, explicit soft, and both all-vs-all requests must succeed");
  }
  if (hard_sink.records.size() != 1U || soft_sink.records.size() != 1U ||
      both_sink.records.size() != 1U) {
    fail("ESM2 hard, explicit soft, and both all-vs-all must emit one record");
  }
  const hiko::PairwiseResult& hard = hard_sink.records[0].result;
  const hiko::PairwiseResult& soft = soft_sink.records[0].result;
  const hiko::PairwiseResult& both = both_sink.records[0].result;
  if (!nearly_equal(both.metrics.raw_sw_score, hard.metrics.raw_sw_score)) {
    fail("ESM2 both all-vs-all raw score must keep the hard score");
  }
  if (both.path.aligned_pairs != hard.path.aligned_pairs) {
    fail("ESM2 both all-vs-all path must keep the hard path");
  }
  require_soft_metrics_match(
      both, soft,
      "ESM2 both all-vs-all soft metrics must match explicit recovered soft gaps");
}

}  // namespace

int main() {
  test_encode_returns_nontrivial_embeddings();
  test_pairwise_self_self_is_identity_path();
  test_pairwise_hard_defaults_use_esm2_annealed_gap_family();
  test_pairwise_soft_defaults_use_esm2_soft_gap_family();
  test_pairwise_both_uses_hard_primary_and_esm2_soft_gap_family();
  test_all_vs_all_emits_one_record_per_pair();
  test_all_vs_all_soft_defaults_use_esm2_soft_gap_family();
  test_all_vs_all_both_uses_hard_primary_and_esm2_soft_gap_family();
  std::printf("sequence_route_smoke_tests: ok\n");
  return 0;
}

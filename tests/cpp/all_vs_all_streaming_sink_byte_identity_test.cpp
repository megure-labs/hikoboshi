// ssf1 byte-identity test: drives both the legacy ring buffer
// (`StreamingRecordBuffer`) and the new per-worker-buffers + sequencer
// (`StreamingSequencerBuffer`) with the same synthetic stream of records
// across multiple producer threads (each producing a roughly-balanced shard
// of pair_ids in ascending order, with deliberately variable per-pair
// latency). The downstream sink captures every emitted record into a
// canonical-order byte stream. The test asserts that the byte stream from
// both designs is bit-identical, which is the contract documented in
// `bench/SCOPE40_5K_PERF.md`: the full TSV must be byte-equal across thread
// counts, scheduling, and re-runs.

#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/streaming_sink.hpp>
#include <hikoboshi/universal/alignment_path.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_streaming_sink_byte_identity_test: %s\n",
               message);
  std::exit(1);
}

// Capture every received record's scalar fields into a flat byte vector. The
// downstream sink runs on the legacy buffer's drain thread or the new
// sequencer thread, so it has exclusive access to the capture vector while
// `receive` is in flight.
class CaptureSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    append_size(record.query_index);
    append_size(record.target_index);
    append_double(record.result.raw_sw_score);
    append_size(record.result.path.aligned_pairs);
    append_int32(record.result.path.query_start);
    append_int32(record.result.path.query_end);
    append_int32(record.result.path.target_start);
    append_int32(record.result.path.target_end);
    append_size(record.result.path.steps.size());
    for (const hiko_u::AlignmentStep& step : record.result.path.steps) {
      append_int32(step.query_index);
      append_int32(step.target_index);
      append_float(step.residue_score);
    }
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::vector<std::uint8_t> captured;

 private:
  template <typename T>
  void append_raw(const T& value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    captured.insert(captured.end(), bytes, bytes + sizeof(T));
  }

  void append_size(std::size_t value) { append_raw(value); }
  void append_int32(std::int32_t value) { append_raw(value); }
  void append_double(double value) { append_raw(value); }
  void append_float(float value) { append_raw(value); }
};

struct SyntheticPair {
  std::size_t pair_id;
  std::size_t query_index;
  std::size_t target_index;
  double raw_sw_score;
  std::size_t aligned_pairs;
  std::int32_t query_start;
  std::int32_t query_end;
  std::int32_t target_start;
  std::int32_t target_end;
  std::vector<hiko_u::AlignmentStep> steps;
  std::uint32_t latency_microseconds;
};

std::vector<SyntheticPair> build_synthetic_pairs(std::size_t total_pair_count,
                                                 std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::uint32_t> latency_dist(0U, 200U);
  std::uniform_int_distribution<int> step_count_dist(2, 24);
  std::uniform_real_distribution<float> score_dist(-1.5F, 3.5F);

  std::vector<SyntheticPair> pairs;
  pairs.reserve(total_pair_count);
  for (std::size_t pair_id = 0; pair_id < total_pair_count; ++pair_id) {
    SyntheticPair p;
    p.pair_id = pair_id;
    p.query_index = (pair_id * 37U + 11U) % 4096U;
    p.target_index = (pair_id * 53U + 7U) % 4096U;
    p.raw_sw_score = static_cast<double>(pair_id) * 1.25 + 0.5;
    p.aligned_pairs = (pair_id * 13U) % 100U;
    p.query_start = static_cast<std::int32_t>(pair_id % 200U);
    p.query_end = p.query_start + 50;
    p.target_start = static_cast<std::int32_t>((pair_id * 7U) % 200U);
    p.target_end = p.target_start + 50;
    const int step_count = step_count_dist(rng);
    p.steps.reserve(static_cast<std::size_t>(step_count));
    for (int step_index = 0; step_index < step_count; ++step_index) {
      hiko_u::AlignmentStep step;
      step.query_index = static_cast<std::int32_t>(p.query_start + step_index);
      step.target_index = static_cast<std::int32_t>(p.target_start + step_index);
      step.residue_score = score_dist(rng);
      p.steps.push_back(step);
    }
    p.latency_microseconds = latency_dist(rng);
    pairs.push_back(std::move(p));
  }
  return pairs;
}

void populate_record(hiko::PairwiseResultRecord& record,
                     const SyntheticPair& pair) {
  record.query_index = pair.query_index;
  record.target_index = pair.target_index;
  record.result.raw_sw_score = pair.raw_sw_score;
  record.result.path.steps.clear();
  record.result.path.aligned_pairs = pair.aligned_pairs;
  record.result.path.query_start = pair.query_start;
  record.result.path.query_end = pair.query_end;
  record.result.path.target_start = pair.target_start;
  record.result.path.target_end = pair.target_end;
  for (const hiko_u::AlignmentStep& step : pair.steps) {
    record.result.path.steps.push_back(step);
  }
}

// Run the legacy ring buffer with the same synthetic stream and return the
// captured byte stream.
std::vector<std::uint8_t> run_legacy(
    const std::vector<SyntheticPair>& pairs,
    std::size_t worker_count,
    std::size_t max_step_count) {
  CaptureSink sink;
  hiko_ad::StreamingRecordBuffer buffer(
      hiko_ad::StreamingRecordBuffer::resolve_slot_count(pairs.size()),
      max_step_count, sink);

  // Shard pair_ids across workers in a round-robin pattern so each worker's
  // submission stream is monotonically increasing. The legacy ring requires
  // submissions in pair_id order to avoid rotation deadlock.
  std::atomic<std::size_t> next_pair_id{0};

  std::thread drainer([&]() {
    hiko_u::Status status = buffer.drain(pairs.size());
    (void)status;
  });

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t w = 0; w < worker_count; ++w) {
    workers.emplace_back([&]() {
      hiko::PairwiseResultRecord record;
      record.result.path.steps.reserve(max_step_count);
      for (;;) {
        const std::size_t pair_id =
            next_pair_id.fetch_add(1U, std::memory_order_relaxed);
        if (pair_id >= pairs.size()) {
          break;
        }
        const SyntheticPair& pair = pairs[pair_id];
        if (pair.latency_microseconds > 0U) {
          std::this_thread::sleep_for(
              std::chrono::microseconds(pair.latency_microseconds));
        }
        populate_record(record, pair);
        buffer.submit(pair_id, record);
      }
    });
  }

  for (auto& w : workers) w.join();
  drainer.join();

  return std::move(sink.captured);
}

// Run the new sequencer with the same synthetic stream and return the
// captured byte stream.
std::vector<std::uint8_t> run_sequencer(
    const std::vector<SyntheticPair>& pairs,
    std::size_t worker_count,
    std::size_t max_step_count) {
  CaptureSink sink;
  hiko_ad::StreamingSequencerBuffer::Config config;
  config.worker_count = worker_count;
  config.total_pair_count = pairs.size();
  config.per_worker_capacity = 16U;
  config.max_result_step_count = max_step_count;
  // Disable global cap to keep the test independent of host memory.
  config.max_in_flight_bytes = 0U;
  hiko_ad::StreamingSequencerBuffer buffer(config, sink);

  std::atomic<std::size_t> next_pair_id{0};

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t w = 0; w < worker_count; ++w) {
    workers.emplace_back([&, worker_id = w]() {
      hiko::PairwiseResultRecord record;
      record.result.path.steps.reserve(max_step_count);
      for (;;) {
        const std::size_t pair_id =
            next_pair_id.fetch_add(1U, std::memory_order_relaxed);
        if (pair_id >= pairs.size()) {
          break;
        }
        const SyntheticPair& pair = pairs[pair_id];
        if (pair.latency_microseconds > 0U) {
          std::this_thread::sleep_for(
              std::chrono::microseconds(pair.latency_microseconds));
        }
        populate_record(record, pair);
        buffer.submit(worker_id, pair_id, record);
      }
    });
  }
  for (auto& w : workers) w.join();

  hiko_u::Status drain_status = buffer.wait_drain_complete();
  if (!hiko_u::is_ok(drain_status)) {
    fail("sequencer drain returned non-Ok status");
  }
  return std::move(sink.captured);
}

void test_byte_identity_at_scale(std::size_t pair_count,
                                 std::size_t worker_count,
                                 std::uint32_t seed,
                                 const char* label) {
  const std::vector<SyntheticPair> pairs = build_synthetic_pairs(pair_count,
                                                                 seed);
  // Determine a generous max_step_count. The synthetic generator uses up to
  // 24 steps per pair, but reserve more to test that capacity is preserved
  // across the pipeline.
  const std::size_t max_step_count = 64U;

  const std::vector<std::uint8_t> legacy_bytes =
      run_legacy(pairs, worker_count, max_step_count);
  const std::vector<std::uint8_t> sequencer_bytes =
      run_sequencer(pairs, worker_count, max_step_count);

  if (legacy_bytes.size() != sequencer_bytes.size()) {
    std::fprintf(stderr,
                 "byte-identity[%s] size mismatch: legacy=%zu sequencer=%zu\n",
                 label, legacy_bytes.size(), sequencer_bytes.size());
    fail("legacy and sequencer captured different total record byte counts");
  }
  if (legacy_bytes != sequencer_bytes) {
    fail("legacy and sequencer captured different record byte streams");
  }
}

void test_byte_identity_small() {
  test_byte_identity_at_scale(/*pair_count=*/64U,
                              /*worker_count=*/4U,
                              /*seed=*/42U,
                              /*label=*/"small");
}

void test_byte_identity_medium() {
  test_byte_identity_at_scale(/*pair_count=*/1024U,
                              /*worker_count=*/8U,
                              /*seed=*/1729U,
                              /*label=*/"medium");
}

void test_byte_identity_with_serial_worker() {
  // Single-worker case: exercises the dispatch_index = pair_id path on both
  // sinks and verifies that the per-worker ring also serialises correctly.
  test_byte_identity_at_scale(/*pair_count=*/200U,
                              /*worker_count=*/1U,
                              /*seed=*/9001U,
                              /*label=*/"serial");
}

void test_byte_identity_at_scale_200_pdb_equivalent() {
  // The packet doc calls for a "200-PDB fixture" byte-identity check.
  // Wiring up actual PDB IO here would be a heavy integration test; this
  // packet-local test exercises the sink at the same shape (200 inputs ->
  // 19,900 pairs) using a synthetic pair stream. ssf2 will run the real
  // 5K/15K SCOPe40 byte-identity check end-to-end.
  constexpr std::size_t kInputCount = 200U;
  constexpr std::size_t kPairCount = (kInputCount * (kInputCount - 1U)) / 2U;
  test_byte_identity_at_scale(kPairCount,
                              /*worker_count=*/8U,
                              /*seed=*/0x5C0FE40U,
                              /*label=*/"200-pdb-equivalent");
}

}  // namespace

int main() {
  test_byte_identity_with_serial_worker();
  test_byte_identity_small();
  test_byte_identity_medium();
  test_byte_identity_at_scale_200_pdb_equivalent();
  return 0;
}

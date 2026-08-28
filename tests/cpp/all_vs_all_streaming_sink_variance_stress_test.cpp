// ssf1 variance stress test: synthetic workload with heavy-tailed
// per-pair latency distribution. Asserts that the per-worker-buffers +
// sequencer streaming sink (`StreamingSequencerBuffer`) finishes within
// 1.2x of the lower-bound wall time `(total_work / worker_count)`.
//
// Packet doc target was 1 M synthetic pairs with Pareto α=2 latency. This
// test scales down to 4000 pairs at ~50us mean per-pair latency so the
// test runs in well under a second on the unit-test lane. The per-pair
// latency is large enough that single-thread sequencer overhead (mutex,
// cv signaling, ring head scan) is amortized to << 1.2x — at smaller
// per-pair latencies (e.g. ~10us) the sequencer overhead dominates and
// the threshold needs to be relaxed. ssf2 will run the full 1 M variant
// on Datacrunch at realistic per-pair latency.

#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/streaming_sink.hpp>
#include <hikoboshi/universal/alignment_path.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr,
               "all_vs_all_streaming_sink_variance_stress_test: %s\n",
               message);
  std::exit(1);
}

class CountSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    if (record.query_index != next_expected_) {
      out_of_order_ = true;
    }
    ++next_expected_;
    ++received_count_;
    return {hiko_u::StatusCode::Ok, ""};
  }

  bool out_of_order() const noexcept { return out_of_order_; }
  std::size_t received_count() const noexcept { return received_count_; }

 private:
  std::size_t next_expected_ = 0;
  std::size_t received_count_ = 0;
  bool out_of_order_ = false;
};

// Sample one positive value from a Pareto distribution with shape alpha and
// scale x_min using the inverse-CDF method. Mean = x_min * alpha / (alpha - 1)
// for alpha > 1.
double sample_pareto(std::mt19937& rng, double alpha, double x_min) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  // Avoid u==0 which yields infinity.
  double u = 0.0;
  do {
    u = uniform(rng);
  } while (u <= 0.0);
  return x_min / std::pow(u, 1.0 / alpha);
}

void busy_wait_microseconds(double microseconds) noexcept {
  // Use steady_clock to spin for the requested number of microseconds. We
  // intentionally avoid std::this_thread::sleep_for here because sleep
  // resolution on Linux is typically ~50us, and we need finer granularity
  // for the heavy-tail test.
  if (microseconds <= 0.0) return;
  const auto start = std::chrono::steady_clock::now();
  const auto target = start + std::chrono::nanoseconds(
      static_cast<std::int64_t>(microseconds * 1000.0));
  while (std::chrono::steady_clock::now() < target) {
    // Spin.
  }
}

void test_variance_stress() {
  constexpr std::size_t kWorkerCount = 8U;
  constexpr std::size_t kTotalPairCount = 4000U;
  constexpr double kAlpha = 2.0;
  constexpr double kXMin = 25.0;  // Mean = kXMin * alpha / (alpha - 1) = 50us.

  // Pre-sample per-pair latencies so total_work is fixed across both branches
  // (helps the lower-bound calculation be accurate).
  std::mt19937 rng(0xA1FAU);
  std::vector<double> latency_us(kTotalPairCount);
  double total_work_us = 0.0;
  for (std::size_t pair_id = 0; pair_id < kTotalPairCount; ++pair_id) {
    latency_us[pair_id] = sample_pareto(rng, kAlpha, kXMin);
    total_work_us += latency_us[pair_id];
  }
  const double lower_bound_us = total_work_us / static_cast<double>(kWorkerCount);

  CountSink sink;
  hiko_ad::StreamingSequencerBuffer::Config config;
  config.worker_count = kWorkerCount;
  config.total_pair_count = kTotalPairCount;
  config.per_worker_capacity = 64U;
  config.max_result_step_count = 0U;
  config.max_in_flight_bytes = 0U;
  hiko_ad::StreamingSequencerBuffer buffer(config, sink);

  std::atomic<std::size_t> next_pair_id{0};
  const auto start = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(kWorkerCount);
  for (std::size_t worker_id = 0; worker_id < kWorkerCount; ++worker_id) {
    workers.emplace_back([&, worker_id]() {
      hiko::PairwiseResultRecord record;
      for (;;) {
        const std::size_t pair_id =
            next_pair_id.fetch_add(1U, std::memory_order_relaxed);
        if (pair_id >= kTotalPairCount) {
          break;
        }
        busy_wait_microseconds(latency_us[pair_id]);
        record.query_index = pair_id;
        record.target_index = pair_id + kTotalPairCount;
        record.result.raw_sw_score = static_cast<double>(pair_id);
        buffer.submit(worker_id, pair_id, record);
      }
    });
  }
  for (auto& w : workers) w.join();

  const hiko_u::Status drain_status = buffer.wait_drain_complete();
  const auto end = std::chrono::steady_clock::now();
  const double wall_us =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count());

  if (!hiko_u::is_ok(drain_status)) {
    fail("sequencer drain returned non-Ok");
  }
  if (sink.received_count() != kTotalPairCount) {
    fail("sink did not receive every pair");
  }
  if (sink.out_of_order()) {
    fail("sink received records out of pair_id order");
  }

  // Assert wall time is within 1.2x of the lower bound. Use a small absolute
  // floor because at very small total_work the lower bound is dominated by
  // thread-startup noise.
  constexpr double kBoundFactor = 1.2;
  constexpr double kAbsoluteFloorUs = 5000.0;  // 5 ms slack for thread setup.
  const double allowed_wall_us =
      kBoundFactor * lower_bound_us + kAbsoluteFloorUs;
  if (wall_us > allowed_wall_us) {
    std::fprintf(stderr,
                 "variance_stress_test: wall=%.0f us, lower_bound=%.0f us, "
                 "allowed=%.0f us (factor=%.2f)\n",
                 wall_us, lower_bound_us, allowed_wall_us, kBoundFactor);
    fail("wall time exceeded 1.2x lower bound");
  }
}

}  // namespace

int main() {
  test_variance_stress();
  return 0;
}

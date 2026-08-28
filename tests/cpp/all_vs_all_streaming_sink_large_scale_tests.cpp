#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_ud = hikoboshi::universal::detail;

namespace {

// Smallest deadlock-triggering N at thread_count = 180. With kInputCount = 400
// the pair_count is 79,800, which exceeds the streaming-sink slot ceiling of
// 65,536. The cost-descending dispatch order is enabled because
// pair_count > kCostAwarePairSchedulingThreshold = 16. The pre-p50 streaming
// driver deadlocks here; the fix ships the streaming branch onto natural
// pair_id dispatch order so the run completes in milliseconds.
constexpr std::size_t kInputCount = 400;
constexpr std::size_t kThreadCount = 180;
// Reports a clearer message before meson's default 30s test timeout fires.
// The fixed run completes in milliseconds even on a low-core host because
// each pair operates on 1- or 4-residue inputs.
constexpr int kTimeoutSeconds = 20;

void fail(const char* message) {
  std::fprintf(stderr,
               "all_vs_all_streaming_sink_large_scale_tests: %s\n",
               message);
  // Use std::quick_exit so the watchdog path can terminate the process
  // without waiting on a still-blocked thread pool destructor.
  std::quick_exit(1);
}

class OrderedCountingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    const hiko_ad::PairIndex expected = hiko_ad::pair_index_to_ij(
        count_, expected_item_count_, /*include_self=*/false);
    if (record.query_index != expected.query_index ||
        record.target_index != expected.target_index) {
      out_of_order_ = true;
    }
    ++count_;
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::size_t count() const noexcept { return count_; }
  bool out_of_order() const noexcept { return out_of_order_; }
  void set_expected_item_count(std::size_t value) noexcept {
    expected_item_count_ = value;
  }

 private:
  std::size_t expected_item_count_ = 0;
  std::size_t count_ = 0;
  bool out_of_order_ = false;
};

hiko_u::EmbeddingView make_embedding_view(const std::vector<float>& values) {
  return {values.size(),
          1U,
          {values.data(), values.size()},
          {nullptr, 0},
          {nullptr, 0}};
}

void test_streaming_does_not_deadlock_with_cost_descending_pair_dispatch() {
  // Build a fixture where cost-descending dispatch order places pair_id 0 at
  // the very last dispatch slot. The predicted per-pair cost is the product
  // of saturated residue counts (`saturated_pair_cost` in
  // cpp/algorithms/all_vs_all/all_vs_all.cpp), so giving structures 0 and 1
  // a single residue while every other structure carries four residues makes
  // pair_id 0 = (structure 0, structure 1) the unique-low-cost pair and
  // sorts it last in cost-descending order. Without the p50 fix the
  // streaming driver assigns the first batch of high-cost dispatch_indexes
  // to second-generation slots whose drain depends on pair_id 0, so every
  // worker blocks on the streaming-sink condition variable while pair_id 0
  // sits at an unreached dispatch_index.
  std::vector<std::vector<float>> storage;
  storage.reserve(kInputCount);
  storage.push_back({1.0F});
  storage.push_back({2.0F});
  for (std::size_t index = 2; index < kInputCount; ++index) {
    storage.push_back({static_cast<float>(index + 1U),
                       static_cast<float>(index + 2U),
                       static_cast<float>(index + 3U),
                       static_cast<float>(index + 4U)});
  }

  std::vector<hiko_u::EmbeddingView> embeddings;
  embeddings.reserve(storage.size());
  for (const auto& values : storage) {
    embeddings.push_back(make_embedding_view(values));
  }

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_ud::ThreadPool pool(kThreadCount);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  OrderedCountingSink sink;
  sink.set_expected_item_count(embeddings.size());

  std::promise<hiko_u::Status> promise;
  auto future = promise.get_future();
  std::thread runner([&]() {
    hiko_u::Status status = hiko::run_all_vs_all_embeddings(
        request, sink, &pool, pool.thread_count(),
        {workers.data(), workers.size()});
    promise.set_value(status);
  });

  if (future.wait_for(std::chrono::seconds(kTimeoutSeconds)) !=
      std::future_status::ready) {
    // Detach the runner so std::quick_exit can terminate the process
    // without waiting on a thread blocked in the streaming sink.
    runner.detach();
    fail("streaming all-vs-all run did not complete within timeout — likely deadlock");
  }
  runner.join();
  const hiko_u::Status status = future.get();
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("streaming all-vs-all run did not return Ok");
  }
  const std::size_t expected_count = hiko_ad::symmetric_pair_count(
      embeddings.size(), /*include_self=*/false);
  if (sink.count() != expected_count) {
    fail("streaming sink record count mismatch");
  }
  if (sink.out_of_order()) {
    fail("streaming sink received out-of-order records");
  }
}

}  // namespace

int main() {
  test_streaming_does_not_deadlock_with_cost_descending_pair_dispatch();
  return 0;
}

#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>
#include <hikoboshi/algorithms/detail/streaming_sink.hpp>
#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_api = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;
namespace hiko_ud = hikoboshi::universal::detail;

namespace {

std::atomic<bool> g_track_allocations{false};
std::atomic<std::size_t> g_max_allocation{0};

void track_allocation(std::size_t size) noexcept {
  if (!g_track_allocations.load(std::memory_order_relaxed)) {
    return;
  }
  std::size_t current = g_max_allocation.load(std::memory_order_relaxed);
  while (size > current &&
         !g_max_allocation.compare_exchange_weak(
             current, size, std::memory_order_relaxed)) {
  }
}

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_streaming_sink_tests: %s\n", message);
  std::exit(1);
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values) {
  return {values.size(), 1, {values.data(), values.size()}, {nullptr, 0}, {nullptr, 0}};
}

std::vector<hiko_u::EmbeddingView> make_embedding_views(
    const std::vector<std::vector<float>>& storage) {
  std::vector<hiko_u::EmbeddingView> embeddings;
  embeddings.reserve(storage.size());
  for (const auto& values : storage) {
    embeddings.push_back(embedding_view(values));
  }
  return embeddings;
}

class CountingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    const hiko_ad::PairIndex expected =
        hiko_ad::pair_index_to_ij(count, expected_item_count, include_self);
    if (record.query_index != expected.query_index ||
        record.target_index != expected.target_index) {
      fail("streaming sink received an out-of-order record");
    }
    ++count;
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::size_t expected_item_count = 0;
  bool include_self = false;
  std::size_t count = 0;
};

class StopOnSecondSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord&) override {
    ++count;
    if (count == 2U) {
      return {hiko_u::StatusCode::Unavailable, "intentional sink stop"};
    }
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::size_t count = 0;
};

class ApiStopOnSecondSink final : public hiko_api::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko_api::PairwiseResultRecord&) override {
    ++count;
    if (count == 2U) {
      return {hiko_u::StatusCode::Unavailable, "intentional API sink stop"};
    }
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::size_t count = 0;
};

void test_sink_failure_stops_enumeration() {
  const std::vector<std::vector<float>> storage = {{1.0F}, {2.0F}, {3.0F}};
  const std::vector<hiko_u::EmbeddingView> embeddings = make_embedding_views(storage);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  StopOnSecondSink sink;
  const hiko_u::Status status = hiko::run_all_vs_all_embeddings(request, sink);
  if (status.code != hiko_u::StatusCode::Unavailable || sink.count != 2U) {
    fail("enumerator must stop and return sink failure immediately");
  }
}

void test_public_engine_sink_failure_stops_enumeration() {
  const std::vector<std::vector<float>> storage = {{1.0F}, {2.0F}, {3.0F}};
  const std::vector<hiko_u::EmbeddingView> embeddings = make_embedding_views(storage);

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  const hiko_api::Engine engine;
  ApiStopOnSecondSink sink;
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::Unavailable || sink.count != 2U) {
    fail("public Engine all-vs-all must stop and return sink failure");
  }
}

void test_streaming_does_not_allocate_record_matrix() {
  std::vector<std::vector<float>> storage;
  constexpr std::size_t kInputCount = 16;
  storage.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    storage.push_back({static_cast<float>(index + 1U)});
  }
  const std::vector<hiko_u::EmbeddingView> embeddings = make_embedding_views(storage);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  CountingSink sink;
  sink.expected_item_count = embeddings.size();

  g_max_allocation.store(0, std::memory_order_relaxed);
  g_track_allocations.store(true, std::memory_order_relaxed);
  const hiko_u::Status status = hiko::run_all_vs_all_embeddings(request, sink);
  g_track_allocations.store(false, std::memory_order_relaxed);

  if (status.code != hiko_u::StatusCode::Ok) {
    fail("streaming all-vs-all run must return Ok");
  }
  const std::size_t expected_count =
      hiko_ad::symmetric_pair_count(embeddings.size(), false);
  if (sink.count != expected_count) {
    fail("streaming sink record count mismatch");
  }

  const std::size_t forbidden_record_batch =
      expected_count * sizeof(hiko::PairwiseResultRecord);
  if (g_max_allocation.load(std::memory_order_relaxed) >=
      forbidden_record_batch) {
    fail("streaming enumerator allocated a pair-record batch");
  }
}

void test_parallel_streaming_sink_receives_ordered_records() {
  std::vector<std::vector<float>> storage;
  constexpr std::size_t kInputCount = 12;
  storage.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    storage.push_back({static_cast<float>(index + 1U)});
  }
  const std::vector<hiko_u::EmbeddingView> embeddings =
      make_embedding_views(storage);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_ud::ThreadPool pool(4);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  CountingSink sink;
  sink.expected_item_count = embeddings.size();

  const hiko_u::Status status =
      hiko::run_all_vs_all_embeddings(request,
                                     sink,
                                     &pool,
                                     pool.thread_count(),
                                     {workers.data(), workers.size()});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("parallel streaming all-vs-all run must return Ok");
  }
  const std::size_t expected_count =
      hiko_ad::symmetric_pair_count(embeddings.size(), false);
  if (sink.count != expected_count) {
    fail("parallel streaming sink record count mismatch");
  }
}

void test_parallel_sink_failure_stops_ordered_drain() {
  std::vector<std::vector<float>> storage;
  constexpr std::size_t kInputCount = 10;
  storage.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    storage.push_back({static_cast<float>(index + 1U)});
  }
  const std::vector<hiko_u::EmbeddingView> embeddings =
      make_embedding_views(storage);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_ud::ThreadPool pool(3);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  StopOnSecondSink sink;
  const hiko_u::Status status =
      hiko::run_all_vs_all_embeddings(request,
                                     sink,
                                     &pool,
                                     pool.thread_count(),
                                     {workers.data(), workers.size()});
  if (status.code != hiko_u::StatusCode::Unavailable || sink.count != 2U) {
    fail("parallel enumerator must drain in order and return sink failure");
  }
}

class SlowSink final : public hiko::PairwiseResultSink {
 public:
  SlowSink(std::size_t expected_item_count,
           bool include_self,
           std::chrono::microseconds delay)
      : expected_item_count_(expected_item_count),
        include_self_(include_self),
        delay_(delay) {}

  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    const hiko_ad::PairIndex expected =
        hiko_ad::pair_index_to_ij(count_, expected_item_count_, include_self_);
    if (record.query_index != expected.query_index ||
        record.target_index != expected.target_index) {
      out_of_order_ = true;
    }
    ++count_;
    if (delay_.count() > 0) {
      std::this_thread::sleep_for(delay_);
    }
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::size_t count() const noexcept { return count_; }
  bool out_of_order() const noexcept { return out_of_order_; }

 private:
  std::size_t expected_item_count_;
  bool include_self_;
  std::chrono::microseconds delay_;
  std::size_t count_ = 0;
  bool out_of_order_ = false;
};

void test_streaming_backpressure_stress() {
  // Slow sink + many fast workers tests that workers correctly block on
  // backpressure when the ring buffer fills, that no deadlock occurs, and
  // that records arrive in pair_id order despite ring-buffer rotation.
  std::vector<std::vector<float>> storage;
  constexpr std::size_t kInputCount = 64;
  storage.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    storage.push_back({static_cast<float>(index + 1U)});
  }
  const std::vector<hiko_u::EmbeddingView> embeddings = make_embedding_views(storage);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_ud::ThreadPool pool(8);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  SlowSink sink(embeddings.size(), false, std::chrono::microseconds(50));

  const hiko_u::Status status =
      hiko::run_all_vs_all_embeddings(request, sink, &pool, pool.thread_count(),
                                     {workers.data(), workers.size()});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("backpressure-stress run must return Ok");
  }
  const std::size_t expected_count =
      hiko_ad::symmetric_pair_count(embeddings.size(), false);
  if (sink.count() != expected_count) {
    fail("backpressure-stress sink record count mismatch");
  }
  if (sink.out_of_order()) {
    fail("backpressure-stress sink received an out-of-order record");
  }
}

void test_streaming_thread_count_sweep_preserves_pair_order() {
  // Run the same pair set across thread_count {1, 2, 4, 8, 16, 32, 64} and
  // verify each run delivers records in pair_id order with the right count.
  std::vector<std::vector<float>> storage;
  constexpr std::size_t kInputCount = 24;
  storage.reserve(kInputCount);
  for (std::size_t index = 0; index < kInputCount; ++index) {
    storage.push_back({static_cast<float>(index + 1U)});
  }
  const std::vector<hiko_u::EmbeddingView> embeddings = make_embedding_views(storage);

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  const std::size_t expected_count =
      hiko_ad::symmetric_pair_count(embeddings.size(), false);
  constexpr std::size_t kThreadCounts[] = {1U, 2U, 4U, 8U, 16U, 32U, 64U};
  for (std::size_t thread_count : kThreadCounts) {
    hiko_ud::ThreadPool pool(thread_count);
    std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
    CountingSink sink;
    sink.expected_item_count = embeddings.size();
    const hiko_u::Status status =
        hiko::run_all_vs_all_embeddings(request, sink, &pool,
                                       pool.thread_count(),
                                       {workers.data(), workers.size()});
    if (status.code != hiko_u::StatusCode::Ok) {
      fail("thread-sweep run must return Ok");
    }
    if (sink.count != expected_count) {
      fail("thread-sweep sink record count mismatch");
    }
  }
}

void test_streaming_record_buffer_resolves_slot_count() {
  using hiko_ad::StreamingRecordBuffer;
  if (StreamingRecordBuffer::resolve_slot_count(0U) != 1U) {
    fail("resolve_slot_count(0) must return 1");
  }
  if (StreamingRecordBuffer::resolve_slot_count(1U) != 1U) {
    fail("resolve_slot_count(1) must return 1");
  }
  if (StreamingRecordBuffer::resolve_slot_count(2U) != 2U) {
    fail("resolve_slot_count(2) must return 2");
  }
  if (StreamingRecordBuffer::resolve_slot_count(3U) != 4U) {
    fail("resolve_slot_count(3) must round up to 4");
  }
  if (StreamingRecordBuffer::resolve_slot_count(7U) != 8U) {
    fail("resolve_slot_count(7) must round up to 8");
  }
  if (StreamingRecordBuffer::resolve_slot_count(127U) != 128U) {
    fail("resolve_slot_count(127) must round up to 128");
  }
  if (StreamingRecordBuffer::resolve_slot_count(1U << 20U) !=
      hiko_ad::kDefaultStreamingSinkSlots) {
    fail("resolve_slot_count must clamp at default cap");
  }
}

class CountingDownstreamSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    if (record.query_index != next_id_) {
      out_of_order_ = true;
    }
    ++next_id_;
    return {hiko_u::StatusCode::Ok, ""};
  }
  std::size_t next_id() const noexcept { return next_id_; }
  bool out_of_order() const noexcept { return out_of_order_; }

 private:
  std::size_t next_id_ = 0;
  bool out_of_order_ = false;
};

void test_streaming_buffer_backpressure_with_small_k() {
  // Force pair_count >> K to exercise the ring-buffer rotation and verify the
  // submit/drain protocol delivers in pair_id order without deadlock.
  using hiko_ad::StreamingRecordBuffer;
  CountingDownstreamSink downstream;
  constexpr std::size_t kSlots = 8U;
  constexpr std::size_t kPairCount = 1024U;
  StreamingRecordBuffer buffer(kSlots, /*max_result_step_count=*/0U,
                               downstream);
  if (buffer.slot_count() != kSlots) {
    fail("backpressure-small-K test: slot_count must be 8");
  }

  std::atomic<bool> producer_failed{false};
  std::thread producer([&]() {
    hiko::PairwiseResultRecord scratch{};
    for (std::size_t pair_id = 0; pair_id < kPairCount; ++pair_id) {
      scratch.query_index = pair_id;
      scratch.target_index = pair_id + kPairCount;
      scratch.result.raw_sw_score =
          static_cast<double>(pair_id);
      buffer.submit(pair_id, scratch);
      if (buffer.aborted()) {
        producer_failed.store(true);
        return;
      }
    }
  });

  const hiko_u::Status drain_status = buffer.drain(kPairCount);
  producer.join();

  if (drain_status.code != hiko_u::StatusCode::Ok) {
    fail("backpressure-small-K drain must return Ok");
  }
  if (producer_failed.load()) {
    fail("backpressure-small-K producer observed unexpected abort");
  }
  if (downstream.next_id() != kPairCount) {
    fail("backpressure-small-K downstream sink count mismatch");
  }
  if (downstream.out_of_order()) {
    fail("backpressure-small-K downstream received an out-of-order record");
  }
}

void test_streaming_buffer_abort_unblocks_blocked_submit() {
  // Small K + a producer that out-runs the (absent) drainer + abort signal
  // unblocks the producer cleanly without delivering pending records.
  using hiko_ad::StreamingRecordBuffer;
  CountingDownstreamSink downstream;
  StreamingRecordBuffer buffer(/*slot_count_hint=*/4U,
                               /*max_result_step_count=*/0U,
                               downstream);

  std::atomic<bool> producer_done{false};
  std::thread producer([&]() {
    hiko::PairwiseResultRecord scratch{};
    for (std::size_t pair_id = 0; pair_id < 64U; ++pair_id) {
      buffer.submit(pair_id, scratch);
      if (buffer.aborted()) {
        break;
      }
    }
    producer_done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  buffer.request_abort();
  producer.join();
  if (!producer_done.load()) {
    fail("abort must unblock the producer");
  }
}

void test_swap_pairwise_result_record_preserves_capacity() {
  hiko::PairwiseResultRecord a{};
  hiko::PairwiseResultRecord b{};
  a.result.path.steps.reserve(64U);
  b.result.path.steps.reserve(128U);
  const std::size_t a_cap_before = a.result.path.steps.capacity();
  const std::size_t b_cap_before = b.result.path.steps.capacity();
  hiko_u::AlignmentStep step{};
  step.query_index = 7;
  step.target_index = 11;
  step.residue_score = 0.5F;
  a.result.path.steps.push_back(step);
  a.query_index = 3U;
  a.target_index = 5U;
  a.result.raw_sw_score = 1.25;

  hiko_ad::swap_pairwise_result_record(a, b);

  if (b.query_index != 3U || b.target_index != 5U) {
    fail("swap must move scalar fields");
  }
  if (b.result.path.steps.size() != 1U) {
    fail("swap must move steps vector content");
  }
  if (b.result.path.steps[0].query_index != 7 ||
      b.result.path.steps[0].target_index != 11) {
    fail("swap must preserve step contents");
  }
  if (a.result.path.steps.size() != 0U) {
    fail("swap must leave source steps empty");
  }
  // After swap, a holds b's old buffer (capacity 128) and b holds a's (64).
  if (a.result.path.steps.capacity() != b_cap_before) {
    fail("swap must transfer reserved capacity from b to a");
  }
  if (b.result.path.steps.capacity() != a_cap_before) {
    fail("swap must transfer reserved capacity from a to b");
  }
}

void test_public_collect_uses_chartered_records() {
  const std::vector<std::vector<float>> storage = {{1.0F}, {2.0F}, {3.0F}};
  const std::vector<hiko_u::EmbeddingView> embeddings = make_embedding_views(storage);

  hiko_api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.include_self = true;

  const hiko_api::Engine engine;
  const auto result = hiko_api::collect_all_vs_all(engine, request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("public collect_all_vs_all embedding request must return Ok");
  }
  const std::size_t expected_count =
      hiko_ad::symmetric_pair_count(embeddings.size(), true);
  if (result.value.records.size() != expected_count) {
    fail("public collect_all_vs_all record count mismatch");
  }
  for (std::size_t index = 0; index < expected_count; ++index) {
    const hiko_ad::PairIndex expected =
        hiko_ad::pair_index_to_ij(index, embeddings.size(), true);
    if (result.value.records[index].query_index != expected.query_index ||
        result.value.records[index].target_index != expected.target_index) {
      fail("public collect_all_vs_all record order mismatch");
    }
  }
}

}  // namespace

void* operator new(std::size_t size) {
  track_allocation(size);
  if (void* pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
  track_allocation(size);
  if (void* pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

int main() {
  test_swap_pairwise_result_record_preserves_capacity();
  test_streaming_record_buffer_resolves_slot_count();
  test_streaming_buffer_backpressure_with_small_k();
  test_streaming_buffer_abort_unblocks_blocked_submit();
  test_sink_failure_stops_enumeration();
  test_public_engine_sink_failure_stops_enumeration();
  test_streaming_does_not_allocate_record_matrix();
  test_parallel_streaming_sink_receives_ordered_records();
  test_parallel_sink_failure_stops_ordered_drain();
  test_streaming_backpressure_stress();
  test_streaming_thread_count_sweep_preserves_pair_order();
  test_public_collect_uses_chartered_records();
  return 0;
}

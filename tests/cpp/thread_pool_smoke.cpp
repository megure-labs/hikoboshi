#include <hikoboshi/universal/detail/thread_pool.hpp>
#include <hikoboshi/universal/execution_options.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace hiko_ud = hikoboshi::universal::detail;
namespace hiko_u = hikoboshi::universal;

static_assert(std::is_same<decltype(std::declval<hiko_u::ExecutionOptions>().thread_count),
                           std::uint32_t>::value,
              "ExecutionOptions.thread_count must be the public thread selector");

template <typename T, typename = void>
struct HasMaxThreadsField : std::false_type {};

template <typename T>
struct HasMaxThreadsField<
    T,
    std::void_t<decltype(std::declval<T>().max_threads)>> : std::true_type {};

static_assert(!HasMaxThreadsField<hiko_u::ExecutionOptions>::value,
              "pre-release ExecutionOptions.max_threads must not be public");

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "thread_pool_smoke: %s\n", message);
  std::exit(1);
}

void require(bool condition, const char* message) {
  if (!condition) {
    fail(message);
  }
}

void test_empty_range_does_not_call_body() {
  hiko_ud::ThreadPool pool(4);
  std::atomic<int> calls{0};
  pool.parallel_for(5, 5, [&](std::size_t, std::size_t, std::size_t) {
    calls.fetch_add(1, std::memory_order_relaxed);
  });
  require(calls.load(std::memory_order_relaxed) == 0,
          "empty range must not invoke the body");
}

void test_small_static_partition() {
  hiko_ud::ThreadPool pool(3);
  std::array<std::size_t, 3> begins{};
  std::array<std::size_t, 3> ends{};
  std::array<bool, 3> seen{};

  pool.parallel_for(2, 8, [&](std::size_t worker_id,
                              std::size_t begin,
                              std::size_t end) {
    require(worker_id < seen.size(), "unexpected worker id for small range");
    begins[worker_id] = begin;
    ends[worker_id] = end;
    seen[worker_id] = true;
  });

  require(seen[0] && seen[1] && seen[2],
          "small range must use three static blocks");
  require(begins[0] == 2 && ends[0] == 4, "worker 0 range mismatch");
  require(begins[1] == 4 && ends[1] == 6, "worker 1 range mismatch");
  require(begins[2] == 6 && ends[2] == 8, "worker 2 range mismatch");
}

void test_large_range_covers_each_index_once() {
  constexpr std::size_t kCount = 10000;
  hiko_ud::ThreadPool pool(4);
  std::vector<std::atomic<int>> visits(kCount);
  for (std::atomic<int>& visit : visits) {
    visit.store(0, std::memory_order_relaxed);
  }

  pool.parallel_for(0, kCount, [&](std::size_t,
                                   std::size_t begin,
                                   std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      visits[index].fetch_add(1, std::memory_order_relaxed);
    }
  });

  for (std::size_t index = 0; index < visits.size(); ++index) {
    if (visits[index].load(std::memory_order_relaxed) != 1) {
      fail("large range must visit every index exactly once");
    }
  }
}

void test_single_thread_runs_on_caller() {
  hiko_ud::ThreadPool pool(1);
  const std::thread::id caller = std::this_thread::get_id();
  bool called = false;

  pool.parallel_for(3, 9, [&](std::size_t worker_id,
                              std::size_t begin,
                              std::size_t end) {
    require(std::this_thread::get_id() == caller,
            "single-thread pool must run on caller thread");
    require(worker_id == 0, "single-thread pool must use worker id zero");
    require(begin == 3 && end == 9,
            "single-thread pool must receive the whole range");
    called = true;
  });

  require(called, "single-thread pool must invoke the body once");
}

void test_multiple_sequential_calls() {
  hiko_ud::ThreadPool pool(4);
  std::array<std::atomic<int>, 32> visits{};
  for (std::atomic<int>& visit : visits) {
    visit.store(0, std::memory_order_relaxed);
  }

  for (int pass = 0; pass < 5; ++pass) {
    pool.parallel_for(0, visits.size(), [&](std::size_t,
                                            std::size_t begin,
                                            std::size_t end) {
      for (std::size_t index = begin; index < end; ++index) {
        visits[index].fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (const std::atomic<int>& visit : visits) {
    require(visit.load(std::memory_order_relaxed) == 5,
            "sequential calls must not lose or repeat work");
  }
}

void test_overprovisioned_workers_skip_empty_blocks() {
  hiko_ud::ThreadPool pool(8);
  std::array<std::atomic<int>, 3> visits{};
  std::atomic<int> calls{0};
  for (std::atomic<int>& visit : visits) {
    visit.store(0, std::memory_order_relaxed);
  }

  pool.parallel_for(0, visits.size(), [&](std::size_t worker_id,
                                          std::size_t begin,
                                          std::size_t end) {
    require(worker_id < visits.size(),
            "overprovisioned range must not schedule empty workers");
    require(end == begin + 1,
            "overprovisioned range should give one item to each active worker");
    visits[begin].fetch_add(1, std::memory_order_relaxed);
    calls.fetch_add(1, std::memory_order_relaxed);
  });

  require(calls.load(std::memory_order_relaxed) == 3,
          "overprovisioned range must schedule only non-empty blocks");
  for (const std::atomic<int>& visit : visits) {
    require(visit.load(std::memory_order_relaxed) == 1,
            "overprovisioned range must visit each item once");
  }
}

void test_zero_thread_count_resolves_to_at_least_one() {
  hiko_ud::ThreadPool pool(0);
  require(pool.thread_count() >= 1,
          "zero thread count must resolve to hardware or fallback one");
}

void test_worker_exception_is_rethrown() {
  hiko_ud::ThreadPool pool(4);
  bool rethrown = false;

  try {
    pool.parallel_for(0, 8, [&](std::size_t worker_id,
                                std::size_t,
                                std::size_t) {
      if (worker_id == 1) {
        throw std::runtime_error("worker failure");
      }
    });
  } catch (const std::runtime_error&) {
    rethrown = true;
  }

  require(rethrown, "worker exceptions must propagate to the caller");

  std::atomic<int> visits{0};
  pool.parallel_for(0, 4, [&](std::size_t, std::size_t begin, std::size_t end) {
    visits.fetch_add(static_cast<int>(end - begin), std::memory_order_relaxed);
  });
  require(visits.load(std::memory_order_relaxed) == 4,
          "pool must remain reusable after an exception");
}

}  // namespace

int main() {
  test_empty_range_does_not_call_body();
  test_small_static_partition();
  test_large_range_covers_each_index_once();
  test_single_thread_runs_on_caller();
  test_multiple_sequential_calls();
  test_overprovisioned_workers_skip_empty_blocks();
  test_zero_thread_count_resolves_to_at_least_one();
  test_worker_exception_is_rethrown();
  return 0;
}

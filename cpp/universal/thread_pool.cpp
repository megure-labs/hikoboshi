#include <hikoboshi/universal/detail/thread_pool.hpp>

#include <algorithm>
#include <thread>

namespace hikoboshi::universal::detail {
namespace {

struct Range {
  std::size_t begin;
  std::size_t end;
};

std::size_t resolve_thread_count(std::size_t requested) noexcept {
  if (requested != 0) {
    return requested;
  }
  const unsigned hardware = std::thread::hardware_concurrency();
  return hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
}

Range partition_range(std::size_t begin,
                      std::size_t total,
                      std::size_t worker_id,
                      std::size_t active_workers) noexcept {
  const std::size_t base = total / active_workers;
  const std::size_t extra = total % active_workers;
  const std::size_t offset =
      worker_id * base + std::min(worker_id, extra);
  const std::size_t size = base + (worker_id < extra ? 1U : 0U);
  return {begin + offset, begin + offset + size};
}

}  // namespace

ThreadPool::ThreadPool(std::size_t thread_count) {
  impl_.thread_count = resolve_thread_count(thread_count);
  impl_.worker_states.resize(impl_.thread_count);
  if (impl_.thread_count <= 1U) {
    return;
  }

  try {
    impl_.threads.reserve(impl_.thread_count - 1U);
    for (std::size_t worker_id = 1; worker_id < impl_.thread_count;
         ++worker_id) {
      impl_.threads.emplace_back([this, worker_id]() {
        worker_loop(worker_id);
      });
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(impl_.mutex);
      impl_.stop = true;
    }
    impl_.work_available.notify_all();
    for (std::thread& worker : impl_.threads) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    throw;
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(impl_.mutex);
    impl_.stop = true;
  }
  impl_.work_available.notify_all();
  for (std::thread& worker : impl_.threads) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

std::size_t ThreadPool::thread_count() const noexcept {
  return impl_.thread_count;
}

void ThreadPool::parallel_for_impl(std::size_t begin,
                                   std::size_t end,
                                   void* body,
                                   TaskFunction task) {
  if (begin >= end) {
    return;
  }

  const std::size_t total = end - begin;
  const std::size_t active_workers = std::min(impl_.thread_count, total);
  const Range caller_range = partition_range(begin, total, 0, active_workers);

  if (active_workers == 1U) {
    task(body, 0, caller_range.begin, caller_range.end);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(impl_.mutex);
    impl_.exception = nullptr;
    impl_.task_body = body;
    impl_.task = task;
    ++impl_.generation;
    impl_.pending_workers = active_workers - 1U;

    for (std::size_t worker_id = 1; worker_id < active_workers; ++worker_id) {
      const Range range = partition_range(begin, total, worker_id, active_workers);
      WorkerState& state = impl_.worker_states[worker_id];
      state.range_begin = range.begin;
      state.range_end = range.end;
      state.generation = impl_.generation;
    }
  }

  impl_.work_available.notify_all();

  try {
    task(body, 0, caller_range.begin, caller_range.end);
  } catch (...) {
    capture_current_exception();
  }

  std::exception_ptr exception;
  {
    std::unique_lock<std::mutex> lock(impl_.mutex);
    impl_.work_finished.wait(lock, [this]() {
      return impl_.pending_workers == 0U;
    });
    exception = impl_.exception;
    impl_.task = nullptr;
    impl_.task_body = nullptr;
    impl_.exception = nullptr;
  }

  if (exception) {
    std::rethrow_exception(exception);
  }
}

void ThreadPool::worker_loop(std::size_t worker_id) noexcept {
  std::uint64_t observed_generation = 0;

  for (;;) {
    void* body = nullptr;
    TaskFunction task = nullptr;
    Range range{0, 0};
    std::uint64_t work_generation = 0;

    {
      std::unique_lock<std::mutex> lock(impl_.mutex);
      impl_.work_available.wait(lock, [this, worker_id, observed_generation]() {
        return impl_.stop ||
               impl_.worker_states[worker_id].generation != observed_generation;
      });

      if (impl_.stop) {
        return;
      }

      const WorkerState& state = impl_.worker_states[worker_id];
      range = {state.range_begin, state.range_end};
      work_generation = state.generation;
      body = impl_.task_body;
      task = impl_.task;
    }

    try {
      task(body, worker_id, range.begin, range.end);
    } catch (...) {
      capture_current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(impl_.mutex);
      observed_generation = work_generation;
      --impl_.pending_workers;
      if (impl_.pending_workers == 0U) {
        impl_.work_finished.notify_one();
      }
    }
  }
}

void ThreadPool::capture_current_exception() noexcept {
  std::lock_guard<std::mutex> lock(impl_.mutex);
  if (!impl_.exception) {
    impl_.exception = std::current_exception();
  }
}

}  // namespace hikoboshi::universal::detail

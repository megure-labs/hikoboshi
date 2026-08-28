#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_THREAD_POOL_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_THREAD_POOL_HPP

/// @file
/// Private fixed-size thread pool for coarse Hikoboshi work partitioning.

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace hikoboshi::universal::detail {

/// Fixed worker pool for blocking static range partitioning.
///
/// `thread_count == 1` is an explicit serial mode. In that mode the pool does
/// not create background worker threads and `parallel_for` runs the whole range
/// on the caller thread with worker id zero.
///
/// Calls to `parallel_for` are blocking and must not be made concurrently on
/// the same pool instance.
class ThreadPool {
 public:
  explicit ThreadPool(std::size_t thread_count = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  [[nodiscard]] std::size_t thread_count() const noexcept;

  template <typename Body>
  void parallel_for(std::size_t begin, std::size_t end, Body&& body) {
    parallel_for_impl(begin,
                      end,
                      std::addressof(body),
                      &ThreadPool::invoke_body<Body>);
  }

 private:
  using TaskFunction = void (*)(void* body,
                                std::size_t worker_id,
                                std::size_t begin,
                                std::size_t end);

  struct alignas(64) WorkerState {
    std::size_t range_begin = 0;
    std::size_t range_end = 0;
    std::uint64_t generation = 0;
  };
  static_assert(alignof(WorkerState) == 64,
                "ThreadPool worker state must be 64-byte aligned");
  static_assert(sizeof(WorkerState) % 64 == 0,
                "ThreadPool worker state must occupy whole cache-line slots");

  template <typename Body>
  static void invoke_body(void* body,
                          std::size_t worker_id,
                          std::size_t begin,
                          std::size_t end) {
    using BodyType = std::remove_reference_t<Body>;
    (*static_cast<BodyType*>(body))(worker_id, begin, end);
  }

  void parallel_for_impl(std::size_t begin,
                         std::size_t end,
                         void* body,
                         TaskFunction task);
  void worker_loop(std::size_t worker_id) noexcept;
  void capture_current_exception() noexcept;

  struct Impl {
    std::size_t thread_count = 1;
    std::vector<WorkerState> worker_states;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable work_available;
    std::condition_variable work_finished;
    std::uint64_t generation = 0;
    std::size_t pending_workers = 0;
    bool stop = false;
    void* task_body = nullptr;
    TaskFunction task = nullptr;
    std::exception_ptr exception;
  };

  Impl impl_;
};

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_THREAD_POOL_HPP

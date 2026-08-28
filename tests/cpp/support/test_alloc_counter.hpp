#ifndef HIKOBOSHI_TESTS_CPP_SUPPORT_TEST_ALLOC_COUNTER_HPP
#define HIKOBOSHI_TESTS_CPP_SUPPORT_TEST_ALLOC_COUNTER_HPP

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace hikoboshi::tests {

class AllocationCounter {
 public:
  static void reset() noexcept {
    allocation_count_.store(0, std::memory_order_relaxed);
  }

  static void set_enabled(bool enabled) noexcept {
    enabled_.store(enabled, std::memory_order_relaxed);
  }

  static std::size_t allocations() noexcept {
    return allocation_count_.load(std::memory_order_relaxed);
  }

  static void record_allocation() noexcept {
    if (enabled_.load(std::memory_order_relaxed)) {
      allocation_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }

 private:
  static inline std::atomic<bool> enabled_{false};
  static inline std::atomic<std::size_t> allocation_count_{0};
};

}  // namespace hikoboshi::tests

#ifdef HIKOBOSHI_TEST_ALLOC_COUNTER_IMPLEMENTATION
void* operator new(std::size_t size) {
  hikoboshi::tests::AllocationCounter::record_allocation();
  if (void* pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
  hikoboshi::tests::AllocationCounter::record_allocation();
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
#endif

#endif  // HIKOBOSHI_TESTS_CPP_SUPPORT_TEST_ALLOC_COUNTER_HPP

#include <hikoboshi/universal/detail/thread_pool.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace hiko_ud = hikoboshi::universal::detail;

namespace {

std::atomic<bool> g_count_allocations{false};
std::atomic<std::size_t> g_allocation_count{0};

void note_allocation() noexcept {
  if (g_count_allocations.load(std::memory_order_relaxed)) {
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void* allocate_bytes(std::size_t size) {
  note_allocation();
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* allocate_aligned_bytes(std::size_t size, std::align_val_t alignment) {
  note_allocation();
  void* pointer = nullptr;
  const std::size_t alignment_value = static_cast<std::size_t>(alignment);
  const int rc = posix_memalign(&pointer,
                                alignment_value,
                                size == 0 ? alignment_value : size);
  if (rc == 0 && pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}

void fail(const char* message) {
  std::fprintf(stderr, "thread_pool_no_alloc: %s\n", message);
  std::exit(1);
}

}  // namespace

void* operator new(std::size_t size) {
  return allocate_bytes(size);
}

void* operator new[](std::size_t size) {
  return allocate_bytes(size);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  return allocate_aligned_bytes(size, alignment);
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate_aligned_bytes(size, alignment);
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

void operator delete(void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}

int main() {
  constexpr std::size_t kCount = 128;
  constexpr int kPasses = 8;

  hiko_ud::ThreadPool pool(4);
  std::array<std::atomic<int>, kCount> visits{};
  for (std::atomic<int>& visit : visits) {
    visit.store(0, std::memory_order_relaxed);
  }

  auto body = [&](std::size_t, std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      visits[index].fetch_add(1, std::memory_order_relaxed);
    }
  };

  pool.parallel_for(0, visits.size(), body);

  for (std::atomic<int>& visit : visits) {
    visit.store(0, std::memory_order_relaxed);
  }
  g_allocation_count.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_relaxed);

  for (int pass = 0; pass < kPasses; ++pass) {
    pool.parallel_for(0, visits.size(), body);
  }

  g_count_allocations.store(false, std::memory_order_relaxed);

  if (g_allocation_count.load(std::memory_order_relaxed) != 0) {
    fail("parallel_for must not allocate after construction and warmup");
  }

  for (const std::atomic<int>& visit : visits) {
    if (visit.load(std::memory_order_relaxed) != kPasses) {
      fail("parallel_for no-allocation run must still cover every index");
    }
  }

  return 0;
}

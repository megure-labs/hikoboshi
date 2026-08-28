// Confirms the staged package validator allocates no heap memory after a
// one-time warmup. The test installs a tracing global allocator, runs the
// validator on the default compiled Hikoboshi-MPNN-64 package descriptor
// inside the tracing window, and fails when any allocation is observed.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string_view>

#include <hikoboshi/dispatch/registry/validation_core.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/package_validation_codes.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

namespace hiko_u = hikoboshi::universal;
namespace hiko_dr = hikoboshi::dispatch::registry;

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
  const int rc = posix_memalign(&pointer, alignment_value,
                                size == 0 ? alignment_value : size);
  if (rc == 0 && pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}

void fail(const char* message) {
  std::fprintf(stderr, "package_validation_noalloc_test: %s\n", message);
  std::exit(1);
}

}  // namespace

void* operator new(std::size_t size) { return allocate_bytes(size); }
void* operator new[](std::size_t size) { return allocate_bytes(size); }
void* operator new(std::size_t size, std::align_val_t alignment) {
  return allocate_aligned_bytes(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate_aligned_bytes(size, alignment);
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
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
  // Warmup: build the default package handle and validate once outside the
  // tracing window. Any one-time static-initialization or registry-resolution
  // allocations land here, not inside the measured loop.
  const hiko_u::Result<hikoboshi::weights::PackageHandle> warmup_result =
      hikoboshi::weights::default_mpnn_d64_package();
  if (warmup_result.status.code != hiko_u::StatusCode::Ok ||
      warmup_result.value.descriptor == nullptr) {
    fail("default_mpnn64_package must succeed during warmup");
  }
  const hikoboshi::weights::WeightManifestView& manifest =
      hikoboshi::weights::default_mpnn_d64_manifest();
  const hiko_dr::TensorChecksumView tensor_checksums[1] = {
      {hikoboshi::weights::kDefaultMpnnD64ModelName, manifest.checksum}};
  const hiko_dr::PackageProvenance provenance{manifest.checksum,
                                           manifest.checksum_algorithm,
                                           manifest.source_checkpoint,
                                           {tensor_checksums, 1}};
  std::uint8_t payload_bytes[16]{};
  const hiko_u::Span<const std::uint8_t> payload{payload_bytes, 16};

  hiko_dr::ValidationDiagnosticsBuffer warmup_buffer{};
  const hiko_u::PackageValidationReport warmup_report = hiko_dr::validate_package(
      *warmup_result.value.descriptor, provenance, payload, warmup_buffer);
  if (!warmup_report.ok || warmup_report.diagnostics.size != 0 ||
      warmup_report.failed_stage_flags != 0) {
    fail("warmup validation must accept the default MPNN-64 package cleanly");
  }

  // Measured loop: every validation under the tracing window must not
  // allocate. Each iteration uses a stack-local buffer so report storage is
  // included in the no-allocation guarantee.
  constexpr int kPasses = 64;
  g_allocation_count.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_relaxed);

  for (int pass = 0; pass < kPasses; ++pass) {
    hiko_dr::ValidationDiagnosticsBuffer measured_buffer{};
    const hiko_u::PackageValidationReport measured_report =
        hiko_dr::validate_package(*warmup_result.value.descriptor, provenance,
                               payload, measured_buffer);
    if (!measured_report.ok || measured_report.diagnostics.size != 0 ||
        measured_report.failed_stage_flags != 0) {
      g_count_allocations.store(false, std::memory_order_relaxed);
      fail("validate_package must remain accepting on stable input");
    }
  }

  g_count_allocations.store(false, std::memory_order_relaxed);

  const std::size_t allocations =
      g_allocation_count.load(std::memory_order_relaxed);
  if (allocations != 0) {
    std::fprintf(stderr,
                 "package_validation_noalloc_test: %zu allocations during "
                 "validate_package loop (expected 0)\n",
                 allocations);
    return 1;
  }

  return 0;
}

#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_BUFFER_DESCRIPTOR_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_BUFFER_DESCRIPTOR_HPP

/// @file
/// Low-level borrowed buffer descriptors for Hikoboshi workspace reporting.

#include <cstddef>
#include <cstdint>

#include <hikoboshi/universal/backend.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::universal::detail {

/// Memory space tag for low-level buffer descriptors.
///
/// Hikoboshi 0.1.0 uses host buffers for the scalar backend. Other spaces are
/// reserved descriptor vocabulary for backend-capable builds.
enum class MemorySpace : std::uint8_t {
  Host = 0,
  Device = 1,
  Staging = 2,
  Unified = 3,
};

/// Opaque backend-owned handle carried without exposing backend SDK types.
struct OpaqueBackendHandle {
  hikoboshi::universal::Backend backend = hikoboshi::universal::Backend::Scalar;
  const void* handle = nullptr;
  std::uintptr_t payload = 0;
};

/// Borrowed lifetime marker for buffers owned outside the descriptor.
struct LifetimeToken {
  const void* owner = nullptr;
  std::uint64_t generation = 0;
};

/// Low-level borrowed buffer description used by workspace descriptors.
///
/// Shapes and strides are expressed in elements. The descriptor describes
/// storage and lifetime; it does not allocate, retain, or free memory.
struct BufferDescriptor {
  MemorySpace memory_space = MemorySpace::Host;
  hikoboshi::universal::DataType dtype = hikoboshi::universal::DataType::Float32;
  hikoboshi::universal::Span<const std::size_t> shape{};
  hikoboshi::universal::Span<const std::size_t> strides{};
  std::size_t alignment = 0;
  OpaqueBackendHandle backend_handle{};
  LifetimeToken lifetime_token{};
};

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_BUFFER_DESCRIPTOR_HPP

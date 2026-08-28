#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_DEVICE_WORKSPACE_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_DEVICE_WORKSPACE_HPP

/// @file
/// Backend workspace inventory descriptors.

#include <cstddef>

#include <hikoboshi/universal/detail/buffer_descriptor.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::universal::detail {

/// Borrowed workspace inventory for backend-specific buffers.
///
/// This descriptor is intentionally plain data: it reports persistent package
/// storage, scratch buffers, capacity, and observed use without owning those
/// allocations.
struct DeviceWorkspace {
  hikoboshi::universal::Span<BufferDescriptor> buffers{};
  hikoboshi::universal::Span<BufferDescriptor> persistent_weights{};
  hikoboshi::universal::Span<BufferDescriptor> scratch_buffers{};
  std::size_t capacity_bytes = 0;
  std::size_t peak_bytes = 0;
  std::size_t allocation_count = 0;
  LifetimeToken lifetime_token{};
};

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_DEVICE_WORKSPACE_HPP

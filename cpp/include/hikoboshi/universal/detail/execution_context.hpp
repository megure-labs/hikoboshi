#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_EXECUTION_CONTEXT_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_EXECUTION_CONTEXT_HPP

/// @file
/// Opaque execution-context descriptors for backend planning boundaries.

#include <cstdint>

#include <hikoboshi/universal/backend.hpp>
#include <hikoboshi/universal/detail/buffer_descriptor.hpp>

namespace hikoboshi::universal::detail {

/// Opaque execution context passed across backend-planning boundaries.
///
/// Scalar Hikoboshi does not require a device queue. Future backend-aware builds
/// can populate the opaque handles while keeping public headers independent of
/// CUDA, HIP, Metal, or other SDK types.
struct ExecutionContext {
  hikoboshi::universal::Backend backend = hikoboshi::universal::Backend::Scalar;
  std::int32_t device_id = -1;
  OpaqueBackendHandle stream_or_queue{};
  OpaqueBackendHandle event_or_fence_pool{};
  OpaqueBackendHandle timing_hooks{};
  LifetimeToken lifetime_token{};
  bool timing_enabled = false;
};

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_EXECUTION_CONTEXT_HPP

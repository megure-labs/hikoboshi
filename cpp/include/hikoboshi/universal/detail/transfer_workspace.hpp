#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_TRANSFER_WORKSPACE_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_TRANSFER_WORKSPACE_HPP

/// @file
/// Transfer workspace descriptors for backend-capable builds.

#include <cstddef>

#include <hikoboshi/universal/detail/buffer_descriptor.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::universal::detail {

/// Borrowed staging-buffer inventory for host/device transfer planning.
///
/// Hikoboshi 0.1.0 scalar execution does not require device transfers, but this
/// plain-data shape lets capability reports and future backends describe
/// staging memory without exposing backend-specific SDK types.
struct TransferWorkspace {
  hikoboshi::universal::Span<BufferDescriptor> staging_buffers{};
  hikoboshi::universal::Span<LifetimeToken> outstanding_transfers{};
  std::size_t capacity_bytes = 0;
  std::size_t staged_bytes = 0;
  std::size_t transfer_bytes = 0;
  LifetimeToken lifetime_token{};
};

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_TRANSFER_WORKSPACE_HPP

#include <hikoboshi/universal/detail/buffer_descriptor.hpp>
#include <hikoboshi/universal/detail/device_workspace.hpp>
#include <hikoboshi/universal/detail/execution_context.hpp>
#include <hikoboshi/universal/detail/transfer_workspace.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hiko_d = hikoboshi::universal::detail;
namespace hiko_u = hikoboshi::universal;

static_assert(std::is_standard_layout<hiko_d::BufferDescriptor>::value,
              "BufferDescriptor must stay a plain descriptor record");
static_assert(std::is_trivially_copyable<hiko_d::BufferDescriptor>::value,
              "BufferDescriptor must remain trivially copyable");
static_assert(std::is_standard_layout<hiko_d::DeviceWorkspace>::value,
              "DeviceWorkspace must stay a plain private workspace record");
static_assert(std::is_standard_layout<hiko_d::TransferWorkspace>::value,
              "TransferWorkspace must stay a plain private workspace record");
static_assert(std::is_standard_layout<hiko_d::ExecutionContext>::value,
              "ExecutionContext must stay a plain private context record");
static_assert(static_cast<std::uint8_t>(hiko_d::MemorySpace::Host) == 0,
              "MemorySpace::Host is the host-resident descriptor tag");
static_assert(static_cast<std::uint8_t>(hiko_d::MemorySpace::Device) == 1,
              "MemorySpace::Device is the future device-resident descriptor tag");
static_assert(static_cast<std::uint8_t>(hiko_d::MemorySpace::Staging) == 2,
              "MemorySpace::Staging is the future transfer-staging tag");
static_assert(static_cast<std::uint8_t>(hiko_d::MemorySpace::Unified) == 3,
              "MemorySpace::Unified is the future unified-memory tag");

int main() {
  const std::size_t shape[2] = {4, 64};
  const std::size_t strides[2] = {64, 1};
  const hiko_d::LifetimeToken lifetime{nullptr, 1};
  const hiko_d::OpaqueBackendHandle handle{hiko_u::Backend::Scalar, nullptr, 0};

  hiko_d::BufferDescriptor descriptor{};
  descriptor.memory_space = hiko_d::MemorySpace::Host;
  descriptor.dtype = hiko_u::DataType::Float32;
  descriptor.shape = {shape, 2};
  descriptor.strides = {strides, 2};
  descriptor.alignment = alignof(float);
  descriptor.backend_handle = handle;
  descriptor.lifetime_token = lifetime;

  hiko_d::BufferDescriptor buffers[1] = {descriptor};
  hiko_d::LifetimeToken transfers[1] = {lifetime};

  hiko_d::DeviceWorkspace device_workspace{};
  device_workspace.buffers = {buffers, 1};
  device_workspace.scratch_buffers = {buffers, 1};
  device_workspace.capacity_bytes = 256;
  device_workspace.lifetime_token = lifetime;

  hiko_d::TransferWorkspace transfer_workspace{};
  transfer_workspace.staging_buffers = {buffers, 1};
  transfer_workspace.outstanding_transfers = {transfers, 1};
  transfer_workspace.capacity_bytes = 256;
  transfer_workspace.lifetime_token = lifetime;

  hiko_d::ExecutionContext execution_context{};
  execution_context.backend = hiko_u::Backend::Scalar;
  execution_context.device_id = -1;
  execution_context.lifetime_token = lifetime;

  if (descriptor.shape.size != 2 || descriptor.strides.size != 2 ||
      device_workspace.buffers.size != 1 ||
      transfer_workspace.outstanding_transfers.size != 1 ||
      execution_context.backend != hiko_u::Backend::Scalar) {
    return 1;
  }

  return 0;
}

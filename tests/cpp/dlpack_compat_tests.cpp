#include <hikoboshi/bindings/dlpack/dlpack_compat.hpp>

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace hiko_d = hikoboshi::bindings::dlpack;

static_assert(hiko_d::kDLPackVersion == 80, "Hikoboshi exports DLPack v0.8 ABI");
static_assert(hiko_d::kDLPackAbiVersion == 1, "Unexpected DLPack ABI version");
static_assert(hiko_d::kDLCPU == 1, "DLPack kDLCPU enum value mismatch");
static_assert(hiko_d::kDLFloat == 2U, "DLPack kDLFloat enum value mismatch");
static_assert(hiko_d::kDLInt == 0U, "DLPack kDLInt enum value mismatch");
static_assert(hiko_d::kDLUInt == 1U, "DLPack kDLUInt enum value mismatch");

static_assert(std::is_standard_layout<hiko_d::DLDevice>::value,
              "DLDevice must be ABI-compatible standard-layout");
static_assert(std::is_standard_layout<hiko_d::DLDataType>::value,
              "DLDataType must be ABI-compatible standard-layout");
static_assert(std::is_standard_layout<hiko_d::DLTensor>::value,
              "DLTensor must be ABI-compatible standard-layout");
static_assert(std::is_standard_layout<hiko_d::DLManagedTensor>::value,
              "DLManagedTensor must be ABI-compatible standard-layout");

static_assert(hiko_d::kDLDeviceDeviceTypeOffset == 0,
              "DLDevice.device_type offset mismatch");
static_assert(hiko_d::kDLDataTypeCodeOffset == 0,
              "DLDataType.code offset mismatch");
static_assert(hiko_d::kDLTensorDataOffset == 0,
              "DLTensor.data offset mismatch");
static_assert(hiko_d::kDLManagedTensorDLTensorOffset == 0,
              "DLManagedTensor.dl_tensor offset mismatch");

static_assert(hiko_d::kDLTensorDataOffset < hiko_d::kDLTensorDeviceOffset,
              "DLTensor field order mismatch");
static_assert(hiko_d::kDLTensorDeviceOffset < hiko_d::kDLTensorNDimOffset,
              "DLTensor field order mismatch");
static_assert(hiko_d::kDLTensorNDimOffset < hiko_d::kDLTensorDTypeOffset,
              "DLTensor field order mismatch");
static_assert(hiko_d::kDLTensorDTypeOffset < hiko_d::kDLTensorShapeOffset,
              "DLTensor field order mismatch");
static_assert(hiko_d::kDLTensorShapeOffset < hiko_d::kDLTensorStridesOffset,
              "DLTensor field order mismatch");
static_assert(hiko_d::kDLTensorStridesOffset < hiko_d::kDLTensorByteOffsetOffset,
              "DLTensor field order mismatch");
static_assert(hiko_d::kDLManagedTensorDLTensorOffset <
                  hiko_d::kDLManagedTensorManagerCtxOffset,
              "DLManagedTensor field order mismatch");
static_assert(hiko_d::kDLManagedTensorManagerCtxOffset <
                  hiko_d::kDLManagedTensorDeleterOffset,
              "DLManagedTensor field order mismatch");

int main() {
  if (std::strcmp(hiko_d::kDLPackCapsuleName, "dltensor") != 0 ||
      std::strcmp(hiko_d::kDLPackUsedCapsuleName, "used_dltensor") != 0) {
    return 1;
  }
  if (!hiko_d::kCapsuleDestructorSkipsUsedDltensor ||
      !hiko_d::kManagedTensorDeleterOwnsSelf ||
      !hiko_d::kProducerKeepsMemoryAliveUntilDeleter) {
    return 2;
  }
  return 0;
}

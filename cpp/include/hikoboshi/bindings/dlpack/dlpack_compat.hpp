#ifndef HIKOBOSHI_BINDINGS_DLPACK_DLPACK_COMPAT_HPP
#define HIKOBOSHI_BINDINGS_DLPACK_DLPACK_COMPAT_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hikoboshi::bindings::dlpack {

inline constexpr int kDLPackVersion = 80;
inline constexpr int kDLPackAbiVersion = 1;

enum DLDeviceType : std::int32_t {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16,
};

struct DLDevice {
  DLDeviceType device_type;
  std::int32_t device_id;
};

enum DLDataTypeCode : std::uint8_t {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
};

struct DLDataType {
  std::uint8_t code;
  std::uint8_t bits;
  std::uint16_t lanes;
};

struct DLTensor {
  void* data;
  DLDevice device;
  std::int32_t ndim;
  DLDataType dtype;
  std::int64_t* shape;
  std::int64_t* strides;
  std::uint64_t byte_offset;
};

struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensor* self);
};

inline constexpr const char* kDLPackCapsuleName = "dltensor";
inline constexpr const char* kDLPackUsedCapsuleName = "used_dltensor";

inline constexpr bool kCapsuleDestructorSkipsUsedDltensor = true;
inline constexpr bool kManagedTensorDeleterOwnsSelf = true;
inline constexpr bool kProducerKeepsMemoryAliveUntilDeleter = true;

inline constexpr std::size_t kDLDeviceDeviceTypeOffset =
    offsetof(DLDevice, device_type);
inline constexpr std::size_t kDLDeviceDeviceIdOffset =
    offsetof(DLDevice, device_id);
inline constexpr std::size_t kDLDataTypeCodeOffset =
    offsetof(DLDataType, code);
inline constexpr std::size_t kDLDataTypeBitsOffset =
    offsetof(DLDataType, bits);
inline constexpr std::size_t kDLDataTypeLanesOffset =
    offsetof(DLDataType, lanes);
inline constexpr std::size_t kDLTensorDataOffset = offsetof(DLTensor, data);
inline constexpr std::size_t kDLTensorDeviceOffset =
    offsetof(DLTensor, device);
inline constexpr std::size_t kDLTensorNDimOffset = offsetof(DLTensor, ndim);
inline constexpr std::size_t kDLTensorDTypeOffset = offsetof(DLTensor, dtype);
inline constexpr std::size_t kDLTensorShapeOffset = offsetof(DLTensor, shape);
inline constexpr std::size_t kDLTensorStridesOffset =
    offsetof(DLTensor, strides);
inline constexpr std::size_t kDLTensorByteOffsetOffset =
    offsetof(DLTensor, byte_offset);
inline constexpr std::size_t kDLManagedTensorDLTensorOffset =
    offsetof(DLManagedTensor, dl_tensor);
inline constexpr std::size_t kDLManagedTensorManagerCtxOffset =
    offsetof(DLManagedTensor, manager_ctx);
inline constexpr std::size_t kDLManagedTensorDeleterOffset =
    offsetof(DLManagedTensor, deleter);

static_assert(std::is_standard_layout<DLDevice>::value,
              "DLPack ABI structs must be standard-layout");
static_assert(std::is_standard_layout<DLDataType>::value,
              "DLPack ABI structs must be standard-layout");
static_assert(std::is_standard_layout<DLTensor>::value,
              "DLPack ABI structs must be standard-layout");
static_assert(std::is_standard_layout<DLManagedTensor>::value,
              "DLPack ABI structs must be standard-layout");

static_assert(kDLDeviceDeviceTypeOffset == 0,
              "DLDevice.device_type must be the first field");
static_assert(kDLDeviceDeviceIdOffset > kDLDeviceDeviceTypeOffset,
              "DLDevice.device_id must follow device_type");
static_assert(kDLDataTypeCodeOffset == 0,
              "DLDataType.code must be the first field");
static_assert(kDLDataTypeBitsOffset > kDLDataTypeCodeOffset,
              "DLDataType.bits must follow code");
static_assert(kDLDataTypeLanesOffset > kDLDataTypeBitsOffset,
              "DLDataType.lanes must follow bits");
static_assert(kDLTensorDataOffset == 0,
              "DLTensor.data must be the first field");
static_assert(kDLTensorDeviceOffset > kDLTensorDataOffset,
              "DLTensor.device must follow data");
static_assert(kDLTensorNDimOffset > kDLTensorDeviceOffset,
              "DLTensor.ndim must follow device");
static_assert(kDLTensorDTypeOffset > kDLTensorNDimOffset,
              "DLTensor.dtype must follow ndim");
static_assert(kDLTensorShapeOffset > kDLTensorDTypeOffset,
              "DLTensor.shape must follow dtype");
static_assert(kDLTensorStridesOffset > kDLTensorShapeOffset,
              "DLTensor.strides must follow shape");
static_assert(kDLTensorByteOffsetOffset > kDLTensorStridesOffset,
              "DLTensor.byte_offset must follow strides");
static_assert(kDLManagedTensorDLTensorOffset == 0,
              "DLManagedTensor.dl_tensor must be the first field");
static_assert(kDLManagedTensorManagerCtxOffset >
                  kDLManagedTensorDLTensorOffset,
              "DLManagedTensor.manager_ctx must follow dl_tensor");
static_assert(kDLManagedTensorDeleterOffset >
                  kDLManagedTensorManagerCtxOffset,
              "DLManagedTensor.deleter must follow manager_ctx");

}  // namespace hikoboshi::bindings::dlpack

#endif  // HIKOBOSHI_BINDINGS_DLPACK_DLPACK_COMPAT_HPP

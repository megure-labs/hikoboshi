#ifndef HIKOBOSHI_UNIVERSAL_TENSOR_HPP
#define HIKOBOSHI_UNIVERSAL_TENSOR_HPP

/// @file
/// Borrowed typed tensor views used by packages and weights.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::universal {

/// Element types used by public tensor and package descriptors.
enum class DataType : std::uint8_t {
  Float32 = 0,
  Float64 = 1,
  Int32 = 2,
  UInt8 = 3,
};

/// Borrowed strided tensor view.
///
/// `shape` and `strides` are expressed in elements, not bytes. `data` is
/// interpreted according to `dtype`; the caller owns the backing storage.
struct TensorView {
  const void* data;
  Span<const std::size_t> shape;
  Span<const std::size_t> strides;
  DataType dtype;
  std::string_view name;
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_TENSOR_HPP

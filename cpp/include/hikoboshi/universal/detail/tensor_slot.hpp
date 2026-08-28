#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_TENSOR_SLOT_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_TENSOR_SLOT_HPP

/// @file
/// Compile-time tensor slot descriptors for generated package payloads.

#include <array>
#include <cstddef>

namespace hikoboshi::universal::detail {

/// Byte width of float32 tensor-slot elements.
inline constexpr std::size_t kTensorSlotFloat32ByteLength = sizeof(float);

/// Compile-time product of static tensor extents.
template <std::size_t... Extents>
constexpr std::size_t tensor_slot_element_count() noexcept {
  return (std::size_t{1} * ... * Extents);
}

/// Static descriptor for a generated float32 tensor slice.
///
/// `DataOffset` is the byte offset in the compiled package payload. `Extents`
/// are the tensor shape in row-major order.
template <std::size_t DataOffset, std::size_t... Extents>
struct TensorSlot {
  using element_type = float;

  static constexpr std::size_t data_offset = DataOffset;
  static constexpr std::size_t rank = sizeof...(Extents);
  inline static constexpr std::array<std::size_t, rank> shape{Extents...};
  static constexpr std::size_t element_count =
      tensor_slot_element_count<Extents...>();
  static constexpr std::size_t element_byte_length =
      kTensorSlotFloat32ByteLength;
  static constexpr std::size_t byte_length =
      element_count * element_byte_length;

  static_assert(rank > 0, "TensorSlot requires at least one extent");
  static_assert(((Extents > 0) && ...),
                "TensorSlot extents must be non-zero");
};

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_TENSOR_SLOT_HPP

#ifndef HIKOBOSHI_CORE_TYPES_SPAN_HPP
#define HIKOBOSHI_CORE_TYPES_SPAN_HPP

#include <cstddef>

namespace hikoboshi::core {

template <typename T>
struct Span {
  T* data;
  std::size_t size;

  constexpr T* begin() const noexcept { return data; }
  constexpr T* end() const noexcept { return size == 0 ? data : data + size; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
  [[nodiscard]] constexpr T& operator[](std::size_t index) const noexcept {
    return data[index];
  }
};

}  // namespace hikoboshi::core

#endif  // HIKOBOSHI_CORE_TYPES_SPAN_HPP

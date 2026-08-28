#ifndef HIKOBOSHI_UNIVERSAL_SPAN_HPP
#define HIKOBOSHI_UNIVERSAL_SPAN_HPP

/// @file
/// Minimal borrowed contiguous span for C++17 public headers.

#include <cstddef>

namespace hikoboshi::universal {

/// Minimal borrowed contiguous span used in C++17 public headers.
///
/// The span does not own storage. `data` may be null only when `size == 0`.
/// Callers must keep the referenced storage alive for the duration of the
/// Hikoboshi call that consumes the span.
template <typename T>
struct Span {
  T* data;
  std::size_t size;

  constexpr T* begin() const noexcept { return data; }
  constexpr T* end() const noexcept { return size == 0 ? data : data + size; }
  constexpr T* cbegin() const noexcept { return data; }
  constexpr T* cend() const noexcept { return end(); }
  [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
  [[nodiscard]] constexpr T& operator[](std::size_t index) const noexcept {
    return data[index];
  }
  [[nodiscard]] constexpr T& front() const noexcept { return data[0]; }
  [[nodiscard]] constexpr T& back() const noexcept { return data[size - 1]; }
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_SPAN_HPP

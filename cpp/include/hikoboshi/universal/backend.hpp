#ifndef HIKOBOSHI_UNIVERSAL_BACKEND_HPP
#define HIKOBOSHI_UNIVERSAL_BACKEND_HPP

/// @file
/// Public backend selector and capability-report tags.

#include <cstdint>

namespace hikoboshi::universal {

/// Public backend selector and capability-report tag.
///
/// `Auto` and `Scalar` are the selectable Hikoboshi 0.1.0 execution backends.
/// The remaining values are stable names for diagnostics and future builds;
/// callers should expect them to be reported as unavailable unless a build
/// explicitly compiles and exposes the matching backend.
enum class Backend : std::uint8_t {
  Auto = 0,
  Scalar = 1,
  Sse4 = 2,
  Avx2 = 3,
  Avx512 = 4,
  Neon = 5,
  Sve = 6,
  Cuda = 7,
  Hip = 8,
  Metal = 9,
  Vulkan = 10,
  OpenCl = 11,
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_BACKEND_HPP

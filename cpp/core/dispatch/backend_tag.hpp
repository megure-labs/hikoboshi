#ifndef HIKOBOSHI_CORE_DISPATCH_BACKEND_TAG_HPP
#define HIKOBOSHI_CORE_DISPATCH_BACKEND_TAG_HPP

#include <cstdint>

namespace hikoboshi::core::dispatch {

enum class BackendTag : std::uint8_t {
  Auto = 0,
  Scalar = 1,
};

constexpr bool selects_scalar(const BackendTag backend) noexcept {
  return backend == BackendTag::Auto || backend == BackendTag::Scalar;
}

}  // namespace hikoboshi::core::dispatch

#endif  // HIKOBOSHI_CORE_DISPATCH_BACKEND_TAG_HPP

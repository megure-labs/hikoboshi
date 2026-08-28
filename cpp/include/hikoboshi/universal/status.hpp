#ifndef HIKOBOSHI_UNIVERSAL_STATUS_HPP
#define HIKOBOSHI_UNIVERSAL_STATUS_HPP

/// @file
/// Allocation-free public status and result records.

#include <cstdint>

namespace hikoboshi::universal {

/// Stable status categories returned by the public C++ API.
enum class StatusCode : std::uint8_t {
  Ok = 0,
  InvalidArgument = 1,
  FailedPrecondition = 2,
  Unavailable = 3,
  Unimplemented = 4,
  InternalError = 5,
};

/// Lightweight status value with a static or caller-owned detail string.
///
/// A status is successful only when `code == StatusCode::Ok`. The `detail`
/// pointer is never owned by Status.
struct [[nodiscard]] Status {
  StatusCode code;
  const char* detail;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == StatusCode::Ok;
  }
};

/// Result wrapper used when an operation can return either a value or status.
template <typename T>
struct [[nodiscard]] Result {
  Status status;
  T value;

  [[nodiscard]] constexpr bool ok() const noexcept { return status.ok(); }
};

/// Build a status value without allocation.
[[nodiscard]] constexpr Status make_status(StatusCode code,
                                           const char* detail = "") noexcept {
  return {code, detail};
}

/// Convenience constructor for successful status returns.
[[nodiscard]] constexpr Status ok_status() noexcept {
  return make_status(StatusCode::Ok);
}

[[nodiscard]] constexpr Status invalid_argument_status(
    const char* detail) noexcept {
  return make_status(StatusCode::InvalidArgument, detail);
}

[[nodiscard]] constexpr Status failed_precondition_status(
    const char* detail) noexcept {
  return make_status(StatusCode::FailedPrecondition, detail);
}

[[nodiscard]] constexpr Status unavailable_status(const char* detail) noexcept {
  return make_status(StatusCode::Unavailable, detail);
}

[[nodiscard]] constexpr Status unimplemented_status(
    const char* detail) noexcept {
  return make_status(StatusCode::Unimplemented, detail);
}

[[nodiscard]] constexpr Status internal_error_status(
    const char* detail) noexcept {
  return make_status(StatusCode::InternalError, detail);
}

[[nodiscard]] constexpr bool is_ok(Status status) noexcept {
  return status.ok();
}

template <typename T>
[[nodiscard]] Result<T> result_from_status(Status status) {
  return {status, T{}};
}

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_STATUS_HPP

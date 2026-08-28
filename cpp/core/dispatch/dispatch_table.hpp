#ifndef HIKOBOSHI_CORE_DISPATCH_DISPATCH_TABLE_HPP
#define HIKOBOSHI_CORE_DISPATCH_DISPATCH_TABLE_HPP

#include <string_view>

#include "backend_tag.hpp"
#include "op_registry.hpp"
#include "scalar_forward.hpp"

namespace hikoboshi::core::dispatch {

struct DispatchTable {
  using Resolve = OpForward (*)(std::string_view op_id) noexcept;
  using Call = bool (*)(std::string_view op_id,
                       void* request,
                       void* output) noexcept;

  BackendTag backend;
  Resolve resolve;
  Call call;
};

inline DispatchTable scalar_dispatch_table() noexcept {
  return {BackendTag::Scalar, &resolve_scalar_forward, &scalar_forward};
}

inline DispatchTable selected_dispatch_table(
    const BackendTag requested = BackendTag::Auto) noexcept {
  (void)requested;
  return scalar_dispatch_table();
}

}  // namespace hikoboshi::core::dispatch

#endif  // HIKOBOSHI_CORE_DISPATCH_DISPATCH_TABLE_HPP

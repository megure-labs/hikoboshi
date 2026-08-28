#ifndef HIKOBOSHI_CORE_DISPATCH_SCALAR_FORWARD_HPP
#define HIKOBOSHI_CORE_DISPATCH_SCALAR_FORWARD_HPP

#include <string_view>

#include "backend_tag.hpp"
#include "op_registry.hpp"

namespace hikoboshi::core::dispatch {

inline bool register_scalar_op(std::string_view op_id, OpForward forward) {
  return register_op({op_id, BackendTag::Scalar, forward});
}

inline OpForward resolve_scalar_forward(std::string_view op_id) noexcept {
  return resolve_op(op_id, BackendTag::Scalar);
}

inline bool scalar_forward(std::string_view op_id,
                           void* request,
                           void* output) noexcept {
  const OpForward forward = resolve_scalar_forward(op_id);
  if (forward == nullptr) {
    return false;
  }
  forward(request, output);
  return true;
}

}  // namespace hikoboshi::core::dispatch

#endif  // HIKOBOSHI_CORE_DISPATCH_SCALAR_FORWARD_HPP

#ifndef HIKOBOSHI_CORE_DISPATCH_OP_REGISTRY_HPP
#define HIKOBOSHI_CORE_DISPATCH_OP_REGISTRY_HPP

#include <cstddef>
#include <string_view>

#include "backend_tag.hpp"

namespace hikoboshi::core::dispatch {

using OpForward = void (*)(void* request, void* output) noexcept;

struct OpRegistration {
  std::string_view op_id;
  BackendTag backend;
  OpForward forward;
};

bool register_op(const OpRegistration& registration);
OpForward resolve_op(std::string_view op_id, BackendTag backend) noexcept;
std::size_t registered_op_count() noexcept;

}  // namespace hikoboshi::core::dispatch

#endif  // HIKOBOSHI_CORE_DISPATCH_OP_REGISTRY_HPP

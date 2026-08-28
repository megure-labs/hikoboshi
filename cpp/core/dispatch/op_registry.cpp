#include "op_registry.hpp"

#include <mutex>
#include <vector>

namespace hikoboshi::core::dispatch {
namespace {

std::mutex& registry_mutex() noexcept {
  static std::mutex mutex;
  return mutex;
}

std::vector<OpRegistration>& registry_entries() {
  static std::vector<OpRegistration> entries;
  return entries;
}

}  // namespace

bool register_op(const OpRegistration& registration) {
  if (registration.op_id.empty() || registration.forward == nullptr ||
      !selects_scalar(registration.backend)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(registry_mutex());
  std::vector<OpRegistration>& entries = registry_entries();
  for (OpRegistration& entry : entries) {
    if (entry.op_id == registration.op_id &&
        entry.backend == registration.backend) {
      entry.forward = registration.forward;
      return true;
    }
  }
  entries.push_back(registration);
  return true;
}

OpForward resolve_op(const std::string_view op_id,
                     const BackendTag backend) noexcept {
  std::lock_guard<std::mutex> lock(registry_mutex());
  const std::vector<OpRegistration>& entries = registry_entries();
  for (const OpRegistration& entry : entries) {
    if (entry.op_id == op_id && entry.backend == backend) {
      return entry.forward;
    }
  }
  return nullptr;
}

std::size_t registered_op_count() noexcept {
  std::lock_guard<std::mutex> lock(registry_mutex());
  return registry_entries().size();
}

}  // namespace hikoboshi::core::dispatch

#include <hikoboshi/io/all_vs_all_layout.hpp>

#include <string>
#include <string_view>

namespace hikoboshi::io {
namespace {

bool is_allowed_identifier_char(char c) noexcept {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
}

}  // namespace

std::string sanitize_identifier(std::string_view value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  bool previous_underscore = false;
  for (const char c : value) {
    char out = is_allowed_identifier_char(c) ? c : '_';
    if (out == '_') {
      if (previous_underscore) {
        continue;
      }
      previous_underscore = true;
    } else {
      previous_underscore = false;
    }
    sanitized.push_back(out);
  }
  return sanitized;
}

}  // namespace hikoboshi::io

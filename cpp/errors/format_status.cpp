#include <hikoboshi/errors/format.hpp>

#include <string>

namespace hikoboshi::errors {

std::string format_status_code(universal::StatusCode code) {
  switch (code) {
    case universal::StatusCode::Ok:
      return "ok";
    case universal::StatusCode::InvalidArgument:
      return "invalid_argument";
    case universal::StatusCode::FailedPrecondition:
      return "failed_precondition";
    case universal::StatusCode::Unavailable:
      return "unavailable";
    case universal::StatusCode::Unimplemented:
      return "unimplemented";
    case universal::StatusCode::InternalError:
      return "internal_error";
  }
  return "unknown_status";
}

std::string format_status(universal::Status status) {
  std::string rendered = format_status_code(status.code);
  if (status.detail != nullptr && status.detail[0] != '\0') {
    rendered += ": ";
    rendered += status.detail;
  }
  return rendered;
}

}  // namespace hikoboshi::errors

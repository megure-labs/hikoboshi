#include <hikoboshi/errors/format.hpp>

#include <iostream>

namespace hikoboshi::cli {

int exit_code_success() noexcept {
  return 0;
}

int exit_code_invalid_arguments() noexcept {
  return 2;
}

int exit_code_input_or_output_unavailable() noexcept {
  return 3;
}

int exit_code_unimplemented() noexcept {
  return 4;
}

int exit_code_internal_failure() noexcept {
  return 5;
}

int exit_code_for_status(hikoboshi::universal::Status status) noexcept {
  switch (status.code) {
    case hikoboshi::universal::StatusCode::Ok:
      return exit_code_success();
    case hikoboshi::universal::StatusCode::InvalidArgument:
    case hikoboshi::universal::StatusCode::FailedPrecondition:
      return exit_code_invalid_arguments();
    case hikoboshi::universal::StatusCode::Unavailable:
      return exit_code_input_or_output_unavailable();
    case hikoboshi::universal::StatusCode::Unimplemented:
      return exit_code_unimplemented();
    case hikoboshi::universal::StatusCode::InternalError:
      return exit_code_internal_failure();
  }
  return exit_code_internal_failure();
}

int report_status(hikoboshi::universal::Status status) {
  if (status.code == hikoboshi::universal::StatusCode::Ok) {
    return exit_code_success();
  }
  std::cerr << "hikoboshi: " << hikoboshi::errors::format_status(status) << '\n';
  return exit_code_for_status(status);
}

}  // namespace hikoboshi::cli

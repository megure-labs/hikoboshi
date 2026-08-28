#include <hikoboshi/api/version.hpp>
#include <hikoboshi/errors/format.hpp>

#include <iostream>
#include <string_view>

namespace hikoboshi::cli {

bool is_help_flag(std::string_view arg) noexcept;
int report_status(hikoboshi::universal::Status status);

namespace {

void print_version_usage(std::ostream& out) {
  out << "usage: hikoboshi version\n";
}

}  // namespace

int run_version(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_version_usage(std::cout);
    return 0;
  }
  if (argc != 0) {
    print_version_usage(std::cerr);
    return report_status({hikoboshi::universal::StatusCode::InvalidArgument,
                          "version does not accept arguments"});
  }

  const hikoboshi::api::VersionInfo info = hikoboshi::api::version_info();
  std::cout << info.product_name << ' ' << info.version.major << '.'
            << info.version.minor << '.' << info.version.patch;
  if (!info.version.label.empty()) {
    std::cout << '-' << info.version.label;
  }
  std::cout << '\n';
  return 0;
}

}  // namespace hikoboshi::cli

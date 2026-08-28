#include <hikoboshi/api/engine.hpp>

#include <cerrno>
#include <cstdlib>
#include <string>
#include <string_view>

namespace hikoboshi::cli {
namespace {

struct ReservedBackendDiagnostic {
  std::string_view name;
  const char* detail;
};

// See also: docs/architecture/CPP_TREE_OVERVIEW.md#reserved-axes. The CLI
// names future backend families for diagnostics, but only auto/scalar are
// accepted by Hikoboshi 0.1.0.
constexpr ReservedBackendDiagnostic kReservedBackendDiagnostics[] = {
    {"sse4",
     "backend 'sse4' is reserved for future x86 SSE4 SIMD builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"avx2",
     "backend 'avx2' is reserved for future x86 AVX2 SIMD builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"avx512",
     "backend 'avx512' is reserved for future x86 AVX-512 SIMD builds; "
     "Hikoboshi 0.1.0 accepts only auto or scalar"},
    {"neon",
     "backend 'neon' is reserved for future ARM NEON SIMD builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"sve",
     "backend 'sve' is reserved for future ARM SVE SIMD builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"cuda",
     "backend 'cuda' is reserved for future NVIDIA CUDA GPU builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"hip",
     "backend 'hip' is reserved for future AMD HIP GPU builds; Hikoboshi 0.1.0 "
     "accepts only auto or scalar"},
    {"metal",
     "backend 'metal' is reserved for future Apple Metal GPU builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"vulkan",
     "backend 'vulkan' is reserved for future Vulkan GPU builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"opencl",
     "backend 'opencl' is reserved for future OpenCL GPU builds; Hikoboshi "
     "0.1.0 accepts only auto or scalar"},
    {"fat",
     "backend 'fat' is reserved for future multi-backend build bundles; "
     "Hikoboshi 0.1.0 accepts only auto or scalar"},
};

}  // namespace

bool is_help_flag(std::string_view arg) noexcept {
  return arg == "-h" || arg == "--help";
}

bool is_option(std::string_view arg, std::string_view name) noexcept {
  return arg == name;
}

bool parse_option_assignment(std::string_view arg,
                             std::string_view name,
                             std::string& value) {
  if (arg.size() <= name.size() + 1 || arg.substr(0, name.size()) != name ||
      arg[name.size()] != '=') {
    return false;
  }
  value.assign(arg.substr(name.size() + 1));
  return true;
}

bool is_rejected_historical_flag(std::string_view arg) noexcept {
  // See also: docs/architecture/ROADMAP.md#reserved-but-not-01-surface. These
  // flags are historical or future surfaces, not hidden aliases for 0.1.0.
  return arg == "--soft" || arg == "--soft-sw" || arg == "--temperature" ||
         arg == "--anneal" || arg == "--save-frames" || arg == "--tree" ||
         arg == "--msa" || arg == "--database" || arg == "--gpu" ||
         arg == "--simd";
}

bool parse_float(std::string_view text, float& value) noexcept {
  std::string copy{text};
  char* end = nullptr;
  errno = 0;
  const float parsed = std::strtof(copy.c_str(), &end);
  if (errno != 0 || end == copy.c_str() || *end != '\0') {
    return false;
  }
  value = parsed;
  return true;
}

hikoboshi::universal::Result<hikoboshi::universal::Backend> parse_backend(
    std::string_view text) noexcept {
  if (text == "auto") {
    return {{hikoboshi::universal::StatusCode::Ok, ""},
            hikoboshi::universal::Backend::Auto};
  }
  if (text == "scalar") {
    return {{hikoboshi::universal::StatusCode::Ok, ""},
            hikoboshi::universal::Backend::Scalar};
  }
  for (const ReservedBackendDiagnostic& diagnostic :
       kReservedBackendDiagnostics) {
    if (text == diagnostic.name) {
      return {{hikoboshi::universal::StatusCode::InvalidArgument,
               diagnostic.detail},
              hikoboshi::universal::Backend::Auto};
    }
  }
  return {{hikoboshi::universal::StatusCode::InvalidArgument,
           "backend must be auto or scalar"},
          hikoboshi::universal::Backend::Auto};
}

}  // namespace hikoboshi::cli

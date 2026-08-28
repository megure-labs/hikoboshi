#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace hikoboshi::cli {
namespace {

constexpr char kExternalPackagePathDiagnostic[] =
    "external package paths are not supported in Hikoboshi 0.1.0; use a compiled "
    "package ID or alias: hikoboshi-mpnn-d64, mpnn64, mpnn-64, "
    "Hikoboshi-MPNN-64, hikoboshi-esm2-8m, esm2-8m, esm2_8m, "
    "Hikoboshi-ESM2-8M, proteinmpnn-v48-eps020, v_48_020, "
    "Hikoboshi-ProteinMPNN-v48-020, proteinmpnn-v48-020, proteinmpnn";

// See also: docs/architecture/ROADMAP.md#reserved-but-not-01-surface. Package
// path and file-like inputs are rejected before provider lookup because 0.1.0
// ships only compiled registered packages.
bool ascii_case_equal(const std::string_view lhs,
                      const std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    char left = lhs[index];
    char right = rhs[index];
    if (left >= 'A' && left <= 'Z') {
      left = static_cast<char>(left - 'A' + 'a');
    }
    if (right >= 'A' && right <= 'Z') {
      right = static_cast<char>(right - 'A' + 'a');
    }
    if (left != right) {
      return false;
    }
  }
  return true;
}

bool ends_with_ascii_case(const std::string_view value,
                          const std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         ascii_case_equal(value.substr(value.size() - suffix.size()), suffix);
}

bool has_file_like_suffix(const std::string_view value) noexcept {
  constexpr std::string_view kRejectedSuffixes[] = {
      ".bin",         ".ckpt", ".dat",  ".gz",   ".h5",
      ".hdf5",        ".json", ".model", ".npy",  ".npz",
      ".onnx",        ".pb",   ".pickle", ".pkg", ".pkl",
      ".pt",          ".pth",  ".safetensors", ".tar", ".tar.gz",
      ".tgz",         ".toml", ".txt",  ".weights", ".yaml",
      ".yml",         ".zip",
  };
  for (const std::string_view suffix : kRejectedSuffixes) {
    if (ends_with_ascii_case(value, suffix)) {
      return true;
    }
  }
  return false;
}

bool is_external_package_path_attempt(const std::string_view value) noexcept {
  return value.find('/') != std::string_view::npos ||
         value.find('\\') != std::string_view::npos ||
         has_file_like_suffix(value);
}

const char* deprecated_alias_target(const std::string_view value) noexcept {
  if (ascii_case_equal(value, "mpnn64") ||
      ascii_case_equal(value, "mpnn-64") ||
      ascii_case_equal(value, "Hikoboshi-MPNN-64")) {
    return "hikoboshi-mpnn-d64";
  }
  if (ascii_case_equal(value, "esm2-8m") ||
      ascii_case_equal(value, "esm2_8m") ||
      ascii_case_equal(value, "Hikoboshi-ESM2-8M")) {
    return "hikoboshi-esm2-8m";
  }
  if (ascii_case_equal(value, "v_48_020") ||
      ascii_case_equal(value, "Hikoboshi-ProteinMPNN-v48-020") ||
      ascii_case_equal(value, "proteinmpnn-v48-020") ||
      ascii_case_equal(value, "proteinmpnn")) {
    return "proteinmpnn-v48-eps020";
  }
  return nullptr;
}

void warn_deprecated_alias_once(const std::string_view value) noexcept {
  const char* target = deprecated_alias_target(value);
  if (target == nullptr) {
    return;
  }
  if (value == std::string_view{target}) {
    return;
  }

  static bool warned_mpnn_d64 = false;
  static bool warned_esm2_8m = false;
  static bool warned_proteinmpnn = false;
  bool* warned = &warned_proteinmpnn;
  if (std::string_view{target} == "hikoboshi-mpnn-d64") {
    warned = &warned_mpnn_d64;
  } else if (std::string_view{target} == "hikoboshi-esm2-8m") {
    warned = &warned_esm2_8m;
  }
  if (*warned) {
    return;
  }
  *warned = true;
  std::fprintf(stderr,
               "warning: package alias '%.*s' is deprecated; use '%s'\n",
               static_cast<int>(value.size()), value.data(), target);
}

}  // namespace

hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_default_cli_package() noexcept {
  return hikoboshi::weights::default_package(
      hikoboshi::weights::kDefaultMpnnD64ModelName);
}

hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_default_design_package() noexcept {
  return hikoboshi::weights::default_package(
      hikoboshi::weights::kDefaultProteinMpnnV48Eps020ModelName);
}

hikoboshi::universal::Result<hikoboshi::universal::PackageHandle>
resolve_cli_package(const std::string_view package_id) noexcept {
  if (is_external_package_path_attempt(package_id)) {
    return {{hikoboshi::universal::StatusCode::InvalidArgument,
             kExternalPackagePathDiagnostic},
            {nullptr, nullptr}};
  }
  warn_deprecated_alias_once(package_id);
  return hikoboshi::weights::default_package(package_id);
}

hikoboshi::universal::Result<hikoboshi::universal::WeightsHandle>
resolve_default_weights() noexcept {
  const hikoboshi::universal::Result<hikoboshi::universal::PackageHandle> package =
      resolve_default_cli_package();
  if (package.status.code != hikoboshi::universal::StatusCode::Ok) {
    return {package.status, {nullptr, nullptr}};
  }
  if (package.value.descriptor == nullptr) {
    return {{hikoboshi::universal::StatusCode::FailedPrecondition,
             "default hikoboshi-mpnn-d64 package descriptor is missing"},
            {nullptr, nullptr}};
  }
  return {package.status,
          package.value.descriptor->compatibility_views.weights};
}

hikoboshi::universal::Result<hikoboshi::api::Engine> make_engine_with_package(
    const hikoboshi::universal::PackageHandle package,
    const hikoboshi::universal::Backend backend,
    const std::uint32_t thread_count) {
  if (package.descriptor == nullptr) {
    return {{hikoboshi::universal::StatusCode::FailedPrecondition,
             "selected Hikoboshi package descriptor is missing"},
            hikoboshi::api::Engine{}};
  }

  hikoboshi::api::EngineConfig config{};
  config.weights = package.descriptor->compatibility_views.weights;
  config.execution.backend = backend;
  config.execution.thread_count = thread_count;
  config.package = package;
  return {{hikoboshi::universal::StatusCode::Ok, ""},
          hikoboshi::api::Engine{config}};
}

hikoboshi::universal::Result<hikoboshi::api::Engine> make_engine_with_package(
    const hikoboshi::universal::PackageHandle package,
    const hikoboshi::universal::Backend backend) {
  return make_engine_with_package(package, backend, 0);
}

hikoboshi::universal::Result<hikoboshi::api::Engine> make_engine_with_default_weights(
    hikoboshi::universal::Backend backend) {
  const hikoboshi::universal::Result<hikoboshi::universal::PackageHandle> package =
      resolve_default_cli_package();
  if (package.status.code != hikoboshi::universal::StatusCode::Ok) {
    return {package.status, hikoboshi::api::Engine{}};
  }
  return make_engine_with_package(package.value, backend);
}

}  // namespace hikoboshi::cli

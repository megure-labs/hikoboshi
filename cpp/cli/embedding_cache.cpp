#include "embedding_cache.hpp"
#include <cstdlib>
#include <iostream>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

namespace hikoboshi::cli {
universal::Result<api::Engine> make_engine_with_package(
    universal::PackageHandle package, universal::Backend backend,
    std::uint32_t thread_count);

universal::Result<api::Engine> make_cached_structure_engine(
    universal::PackageHandle package, universal::Backend backend,
    std::uint32_t thread_count, const std::string& directory,
    io::DiskStructureEmbeddingCache& cache) {
  auto engine = make_engine_with_package(package, backend, thread_count);
  if (!universal::is_ok(engine.status) || directory.empty()) return engine;
  const auto executable = io::current_executable_sha256();
  if (!universal::is_ok(executable.status)) return {executable.status, api::Engine{}};
  const auto* weights = package.descriptor->compatibility_views.weights.view;
  if (weights == nullptr || weights->metadata.checksum.empty())
    return {universal::failed_precondition_status("embedding cache requires model checksum"), api::Engine{}};
  std::string identity = "structure-mpnn-cache-v1\n" + executable.value + "\n" +
      std::string(package.descriptor->identity.package_id) + "\n" +
      std::string(weights->metadata.checksum) + "\n";
  // The exact executable binds the build default. Retain the environment
  // selector verbatim: unrecognized selectors only cause conservative misses.
  const char* parity = std::getenv("HIKOBOSHI_GEMM_PARITY_MODE");
  identity += parity == nullptr ? "build-default" : parity;
#if defined(__unix__) || defined(__APPLE__)
  struct utsname machine{};
  if (uname(&machine) != 0)
    return {universal::unavailable_status("embedding cache machine identity lookup failed"), api::Engine{}};
  identity += "\n" + std::string(machine.sysname) + "\n" + machine.release +
      "\n" + machine.machine + "\n" + machine.nodename;
#endif
  const auto status = cache.prepare(directory, identity);
  if (!universal::is_ok(status)) return {status, api::Engine{}};
  auto config = engine.value.config();
  config.structure_embedding_cache = &cache;
  return {universal::ok_status(), api::Engine{config}};
}

void report_embedding_cache_statistics(const io::DiskStructureEmbeddingCache& cache) {
  const auto counts = cache.statistics();
  std::cerr << "embedding-cache: hits=" << counts.hits << " misses=" << counts.misses
            << " writes=" << counts.writes << " rejected=" << counts.rejected << '\n';
}
}  // namespace hikoboshi::cli

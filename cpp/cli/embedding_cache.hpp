#ifndef HIKOBOSHI_CLI_EMBEDDING_CACHE_HPP
#define HIKOBOSHI_CLI_EMBEDDING_CACHE_HPP
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/io/structure_embedding_cache.hpp>

namespace hikoboshi::cli {
universal::Result<api::Engine> make_cached_structure_engine(
    universal::PackageHandle package, universal::Backend backend,
    std::uint32_t thread_count, const std::string& directory,
    io::DiskStructureEmbeddingCache& cache);
void report_embedding_cache_statistics(const io::DiskStructureEmbeddingCache& cache);
}
#endif

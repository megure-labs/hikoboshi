#ifndef HIKOBOSHI_IO_STRUCTURE_EMBEDDING_CACHE_HPP
#define HIKOBOSHI_IO_STRUCTURE_EMBEDDING_CACHE_HPP

#include <atomic>
#include <filesystem>
#include <string_view>
#include <hikoboshi/universal/detail/sha256.hpp>
#include <hikoboshi/universal/structure_embedding_cache.hpp>

namespace hikoboshi::io {

/// Local, content-addressed cache. The caller supplies a complete encoder
/// identity (code/build, package/weights and effective numerical mode).
/// Concurrent loads/stores are supported after prepare, including separate
/// processes sharing a directory. No eviction is performed.
class DiskStructureEmbeddingCache final : public universal::StructureEmbeddingCache {
 public:
  [[nodiscard]] universal::Status prepare(const std::filesystem::path& directory,
                                         std::string_view encoder_identity);
  [[nodiscard]] universal::Status load(const universal::StructureView& structure,
                                       universal::Span<float> output, bool& hit) override;
  [[nodiscard]] universal::Status store(const universal::StructureView& structure,
                                        universal::Span<const float> values) override;
  struct Statistics {
    std::size_t hits, misses, writes, rejected;
  };
  [[nodiscard]] Statistics statistics() const noexcept;

 private:
  using Digest = universal::detail::Sha256::Digest;
  [[nodiscard]] Digest key(const universal::StructureView& structure,
                           std::size_t value_count) const noexcept;
  [[nodiscard]] std::filesystem::path entry_path(const Digest& key) const;
  std::filesystem::path directory_;
  Digest identity_{};
  bool prepared_ = false;
  std::atomic<std::size_t> hits_{0}, misses_{0}, writes_{0}, rejected_{0};
};

/// Hash the running executable, not argv[0], on supported native CLI platforms.
[[nodiscard]] universal::Result<std::string> current_executable_sha256();

}  // namespace hikoboshi::io
#endif

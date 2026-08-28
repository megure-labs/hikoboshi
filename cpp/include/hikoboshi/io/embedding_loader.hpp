#ifndef HIKOBOSHI_IO_EMBEDDING_LOADER_HPP
#define HIKOBOSHI_IO_EMBEDDING_LOADER_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

struct EmbeddingLoadOptions {
  std::optional<std::size_t> expected_dimension{};
  std::optional<std::size_t> expected_residue_count{};
};

class LoadedEmbedding {
 public:
  struct Impl {
    std::vector<float> values;
    std::vector<char> residue_codes;
    std::vector<universal::ResidueMetadataView> residues;
    std::deque<std::string> string_pool;
    std::string source_filename_storage;
    std::string input_id_storage;
    std::size_t residue_count = 0;
    std::size_t dimension = 0;
    bool has_residue_metadata = false;
  };

  LoadedEmbedding();
  ~LoadedEmbedding();
  LoadedEmbedding(const LoadedEmbedding&) = delete;
  LoadedEmbedding& operator=(const LoadedEmbedding&) = delete;
  LoadedEmbedding(LoadedEmbedding&&) noexcept;
  LoadedEmbedding& operator=(LoadedEmbedding&&) noexcept;

  universal::EmbeddingView view() const noexcept;

  std::size_t residue_count() const noexcept;
  std::size_t dimension() const noexcept;
  std::string_view input_id() const noexcept;
  std::string_view source_filename() const noexcept;

  bool has_residue_metadata() const noexcept;

  Impl& impl() noexcept { return *impl_; }
  const Impl& impl() const noexcept { return *impl_; }

 private:
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] universal::Result<LoadedEmbedding> load_embedding_from_npy_bytes(
    universal::Span<const std::uint8_t> bytes,
    std::string_view origin_label,
    const EmbeddingLoadOptions& options = {});

[[nodiscard]] universal::Result<LoadedEmbedding> load_embedding_from_file(
    std::string_view path,
    const EmbeddingLoadOptions& options = {});

void attach_residue_metadata(LoadedEmbedding& embedding,
                             universal::Span<const char> residue_codes);

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_EMBEDDING_LOADER_HPP

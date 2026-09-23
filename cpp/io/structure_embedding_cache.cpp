#include <hikoboshi/io/structure_embedding_cache.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace hikoboshi::io {
namespace {
using universal::detail::Sha256;
using universal::Status;
constexpr std::array<unsigned char, 8> kMagic{'H','I','K','O','E','M','B',1};
using Header = std::array<unsigned char, 80>;

void hash_count(Sha256& hash, std::size_t value) noexcept {
  std::array<unsigned char, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<unsigned char>(static_cast<std::uint64_t>(value) >> (8 * i));
  hash.update(bytes.data(), bytes.size());
}

Header header(const Sha256::Digest& key, std::size_t count,
              const Sha256::Digest& payload) noexcept {
  Header result{};
  std::copy(kMagic.begin(), kMagic.end(), result.begin());
  std::copy(key.begin(), key.end(), result.begin() + 8);
  for (std::size_t i = 0; i < 8; ++i)
    result[40 + i] = static_cast<unsigned char>(static_cast<std::uint64_t>(count) >> (8 * i));
  std::copy(payload.begin(), payload.end(), result.begin() + 48);
  return result;
}

bool valid_shape(const universal::StructureView& structure,
                 std::size_t count, const void* values) noexcept {
  return values != nullptr && structure.residue_count != 0 && count != 0 &&
      count % structure.residue_count == 0 &&
      count <= (std::numeric_limits<std::size_t>::max() - Header{}.size()) / sizeof(float) &&
      structure.residue_count <= std::numeric_limits<std::size_t>::max() / (15 * sizeof(float)) &&
      structure.coordinates.data != nullptr &&
      structure.coordinates.size >= structure.residue_count * 15 &&
      structure.atom_sources.data != nullptr &&
      structure.atom_sources.size >= structure.residue_count * 5;
}

struct TemporaryEntry {
  std::string path;
  std::FILE* file = nullptr;
  ~TemporaryEntry() {
    if (file != nullptr) std::fclose(file);
    if (!path.empty()) std::remove(path.c_str());
  }
};
}  // namespace

universal::Status DiskStructureEmbeddingCache::prepare(
    const std::filesystem::path& directory, std::string_view encoder_identity) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  prepared_ = false;
  if (directory.empty() || encoder_identity.empty())
    return universal::invalid_argument_status("embedding cache requires directory and encoder identity");
  Sha256 hash;
  hash.update(kMagic.data(), kMagic.size());
  hash.update(encoder_identity.data(), encoder_identity.size());
  const std::uint32_t endian_marker = 1;
  const unsigned char little = *reinterpret_cast<const unsigned char*>(&endian_marker);
  hash.update(&little, 1);
  identity_ = hash.finish();
  directory_ = directory / Sha256::hex(identity_);
  std::error_code error;
  std::filesystem::create_directories(directory_, error);
  if (error || !std::filesystem::is_directory(directory_, error) || error)
    return universal::unavailable_status("embedding cache directory is not writable");
  hits_ = misses_ = writes_ = rejected_ = 0;
  prepared_ = true;
  return universal::ok_status();
}

DiskStructureEmbeddingCache::Digest DiskStructureEmbeddingCache::key(
    const universal::StructureView& structure, std::size_t value_count) const noexcept {
  Sha256 hash;
  hash.update(identity_.data(), identity_.size());
  hash_count(hash, structure.residue_count);
  hash_count(hash, value_count);
  hash.update(structure.coordinates.data, structure.residue_count * 15 * sizeof(float));
  hash.update(structure.atom_sources.data, structure.residue_count * 5 * sizeof(universal::AtomSource));
  // Metadata remains live for geometry/artifacts. The encoder consumes only
  // normalized coordinates and atom provenance, not names or source filenames.
  return hash.finish();
}

std::filesystem::path DiskStructureEmbeddingCache::entry_path(const Digest& digest) const {
  const auto name = Sha256::hex(digest);
  return directory_ / name.substr(0, 2) / (name + ".hke");
}

universal::Status DiskStructureEmbeddingCache::load(
    const universal::StructureView& structure, universal::Span<float> output, bool& hit) {
  hit = false;
  if (!prepared_ || !valid_shape(structure, output.size, output.data))
    return universal::invalid_argument_status("embedding cache load shape or initialization is invalid");
  const auto digest = key(structure, output.size);
  const auto path = entry_path(digest);
  errno = 0;
  std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
      std::fopen(path.c_str(), "rb"), &std::fclose);
  if (!file) {
    if (errno != ENOENT)
      return universal::unavailable_status("embedding cache entry could not be read");
    ++misses_;
    return universal::ok_status();
  }
  Header stored{};
  bool valid = std::fread(stored.data(), 1, stored.size(), file.get()) == stored.size();
  const auto expected = header(digest, output.size, {});
  valid = valid && std::equal(stored.begin(), stored.begin() + 48, expected.begin());
  if (valid) {
    valid = std::fread(output.data, sizeof(float), output.size, file.get()) == output.size &&
        std::fgetc(file.get()) == EOF;
  }
  if (std::ferror(file.get()))
    return universal::unavailable_status("embedding cache entry read failed");
  if (valid) {
    Sha256 hash;
    hash.update(output.data, output.size * sizeof(float));
    const auto payload = hash.finish();
    valid = std::equal(payload.begin(), payload.end(), stored.begin() + 48);
    for (std::size_t i = 0; valid && i < output.size; ++i)
      valid = std::isfinite(output.data[i]);
  }
  if (!valid) {
    ++rejected_;
    ++misses_;
    return universal::ok_status();
  }
  hit = true;
  ++hits_;
  return universal::ok_status();
}

universal::Status DiskStructureEmbeddingCache::store(
    const universal::StructureView& structure, universal::Span<const float> values) {
  if (!prepared_ || !valid_shape(structure, values.size, values.data))
    return universal::invalid_argument_status("embedding cache store shape or initialization is invalid");
  for (std::size_t i = 0; i < values.size; ++i)
    if (!std::isfinite(values.data[i]))
      return universal::invalid_argument_status("embedding cache cannot store non-finite embeddings");
  const auto digest = key(structure, values.size);
  const auto path = entry_path(digest);
  Sha256 hash;
  hash.update(values.data, values.size * sizeof(float));
  const auto bytes = header(digest, values.size, hash.finish());
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) return universal::unavailable_status("embedding cache entry directory creation failed");
  TemporaryEntry temporary;
#if defined(__unix__) || defined(__APPLE__)
  auto pattern = (path.parent_path() / ".pending-XXXXXX").string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  const int descriptor = mkstemp(name.data());
  if (descriptor < 0)
    return universal::unavailable_status("embedding cache temporary file creation failed");
  temporary.path = name.data();
  temporary.file = fdopen(descriptor, "wb");
  if (!temporary.file) {
    close(descriptor);
    return universal::unavailable_status("embedding cache temporary file open failed");
  }
#else
  return universal::unavailable_status("embedding cache atomic writes require a POSIX platform");
#endif
  if (std::fwrite(bytes.data(), 1, bytes.size(), temporary.file) != bytes.size() ||
      std::fwrite(values.data, sizeof(float), values.size, temporary.file) != values.size)
    return universal::unavailable_status("embedding cache write failed");
  const int closed = std::fclose(temporary.file);
  temporary.file = nullptr;
  if (closed != 0) return universal::unavailable_status("embedding cache flush failed");
  std::filesystem::rename(temporary.path, path, error);
  if (error) return universal::unavailable_status("embedding cache atomic publication failed");
  temporary.path.clear();
  ++writes_;
  return universal::ok_status();
}

DiskStructureEmbeddingCache::Statistics DiskStructureEmbeddingCache::statistics() const noexcept {
  return {hits_.load(), misses_.load(), writes_.load(), rejected_.load()};
}

universal::Result<std::string> current_executable_sha256() {
  std::filesystem::path path;
#if defined(__linux__)
  path = "/proc/self/exe";
#elif defined(__APPLE__)
  std::uint32_t size = 0;
  (void)_NSGetExecutablePath(nullptr, &size);
  std::vector<char> buffer(size);
  if (_NSGetExecutablePath(buffer.data(), &size) != 0)
    return {universal::unavailable_status("embedding cache executable path lookup failed"), {}};
  path = buffer.data();
#else
  return {universal::unavailable_status("embedding cache executable identity is unavailable on this platform"), {}};
#endif
  std::ifstream input(path, std::ios::binary);
  if (!input) return {universal::unavailable_status("embedding cache executable could not be read"), {}};
  Sha256 hash;
  std::array<char, 65536> block{};
  while (input) {
    input.read(block.data(), block.size());
    hash.update(block.data(), static_cast<std::size_t>(input.gcount()));
  }
  if (!input.eof()) return {universal::unavailable_status("embedding cache executable hashing failed"), {}};
  return {universal::ok_status(), Sha256::hex(hash.finish())};
}
}  // namespace hikoboshi::io

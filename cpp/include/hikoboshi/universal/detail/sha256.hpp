#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_SHA256_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_SHA256_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace hikoboshi::universal::detail {

// Incremental SHA-256 for cache content identity. Independently implemented
// from FIPS 180-4, sections 4.1.2, 4.2.2, 5 and 6.2; no external source used.
// https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
class Sha256 {
 public:
  using Digest = std::array<std::uint8_t, 32>;
  void update(const void* bytes, std::size_t count) noexcept;
  [[nodiscard]] Digest finish() const noexcept;
  [[nodiscard]] static std::string hex(const Digest& digest);

 private:
  void compress(const std::uint8_t* block) noexcept;
  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> pending_{};
  std::size_t pending_size_ = 0;
  std::uint64_t byte_count_ = 0;
};

}  // namespace hikoboshi::universal::detail
#endif

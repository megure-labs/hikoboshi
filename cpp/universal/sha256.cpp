#include <hikoboshi/universal/detail/sha256.hpp>

#include <algorithm>
#include <cstring>

namespace hikoboshi::universal::detail {
namespace {
constexpr std::uint32_t rotate_right(std::uint32_t x, unsigned bits) noexcept {
  return (x >> bits) | (x << (32U - bits));
}
// FIPS 180-4 section 4.2.2 constants (mathematical values).
constexpr std::array<std::uint32_t, 64> kRound{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
}

void Sha256::compress(const std::uint8_t* block) noexcept {
  std::array<std::uint32_t, 64> schedule{};
  for (std::size_t i = 0; i < 16; ++i) {
    for (std::size_t byte = 0; byte < 4; ++byte)
      schedule[i] = (schedule[i] << 8U) | block[4 * i + byte];
  }
  for (std::size_t i = 16; i < schedule.size(); ++i) {
    const auto x = schedule[i - 15], y = schedule[i - 2];
    schedule[i] = schedule[i - 16] + schedule[i - 7] +
        (rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3U)) +
        (rotate_right(y, 17) ^ rotate_right(y, 19) ^ (y >> 10U));
  }
  auto words = state_;
  for (std::size_t i = 0; i < schedule.size(); ++i) {
    const auto a = words[0], b = words[1], c = words[2], e = words[4];
    const auto sum = words[7] +
        (rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25)) +
        ((e & words[5]) ^ (~e & words[6])) + kRound[i] + schedule[i];
    const auto next = sum + (rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22)) +
        ((a & b) ^ (a & c) ^ (b & c));
    for (std::size_t j = 7; j > 0; --j) words[j] = words[j - 1];
    words[4] += sum;
    words[0] = next;
  }
  for (std::size_t i = 0; i < state_.size(); ++i) state_[i] += words[i];
}

void Sha256::update(const void* bytes, std::size_t count) noexcept {
  const auto* input = static_cast<const std::uint8_t*>(bytes);
  byte_count_ += count;
  while (count != 0) {
    if (pending_size_ == 0 && count >= pending_.size()) {
      compress(input);
      input += pending_.size();
      count -= pending_.size();
      continue;
    }
    const auto n = std::min(count, pending_.size() - pending_size_);
    std::memcpy(pending_.data() + pending_size_, input, n);
    input += n;
    count -= n;
    pending_size_ += n;
    if (pending_size_ == pending_.size()) {
      compress(pending_.data());
      pending_size_ = 0;
    }
  }
}

Sha256::Digest Sha256::finish() const noexcept {
  Sha256 final = *this;
  const auto bits = byte_count_ * 8U;
  const std::uint8_t marker = 0x80, zero = 0;
  final.update(&marker, 1);
  while (final.pending_size_ != 56) final.update(&zero, 1);
  std::array<std::uint8_t, 8> length{};
  for (std::size_t i = 0; i < length.size(); ++i)
    length[i] = static_cast<std::uint8_t>(bits >> (8 * (7 - i)));
  final.update(length.data(), length.size());
  Digest digest{};
  for (std::size_t i = 0; i < digest.size(); ++i)
    digest[i] = static_cast<std::uint8_t>(final.state_[i / 4] >> (8 * (3 - i % 4)));
  return digest;
}

std::string Sha256::hex(const Digest& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = digits[digest[i] >> 4U];
    result[2 * i + 1] = digits[digest[i] & 15U];
  }
  return result;
}
}  // namespace hikoboshi::universal::detail

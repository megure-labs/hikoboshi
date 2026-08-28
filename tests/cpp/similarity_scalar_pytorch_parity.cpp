// fe2 diagnostic: Hikoboshi similarity_scalar versus the PyTorch reference
// `sim_matrix` on the three fe1 lead-diagnostic pairs.
//
// fe1 proved Hikoboshi's `soft_smith_waterman_scalar` kernel matches
// `orihime.soft_sw_affine` to <= 3e-6 max abs delta when both are fed the
// same `sim_matrix` (see `tests/cpp/softsw_posterior_pytorch_parity.cpp`).
// The only remaining input the kernel sees in the production pairwise
// soft-SW path is the `sim_matrix` that `cpp/modules/similarity/similarity_scalar.cpp`
// builds, so fe2 tests whether that buffer matches PyTorch's
// `einsum("id,jd->ij", h1, h2)` element-by-element when Hikoboshi is fed
// the same CLS/EOS-wrapped tokens PyTorch saw.
//
// Pipeline driven by this test:
//   1. Hard-coded aatype tokens for the three lead pairs (loaded once
//      from the private validation corpus and inlined
//      here so the test does not need an HDF5 reader or a Python
//      fixture-generator). Token semantics match PyTorch's training
//      pipeline: each value is a 0..19 index into the compacted 20-AA
//      vocabulary; CLS_IDX=26 / EOS_IDX=27 are added in-test before the
//      encoder call.
//   2. Wrap as `[CLS, aa..., EOS]` and run
//      `hikoboshi::algorithms::detail::encode_esm2_sequence` against the
//      embedded Hikoboshi-ESM2-8M weights.
//   3. Slice CLS/EOS off the resulting `[L+2, 320]` embeddings — fe1's
//      goldens are `[L, L]` over residue positions only, mirroring
//      PyTorch's `embeddings[1:-1]` in `scripts/softsw_pytorch_goldens.py`.
//   4. Hand the residue-only embeddings to `score_embeddings_and_run_resolved`
//      indirectly via `run_pairwise_embeddings`, configured for
//      `soft_mode=true` with the trained gap params and `temperature=1.0`
//      stored in the fe1 goldens. After the call,
//      `workspace.similarity_data()` holds the score matrix the soft-SW
//      kernel was fed.
//   5. Element-by-element compare to fe1's `pair_<idx>_sim_matrix` golden.
//
// Bisection outcome table:
//   - delta `<= 1e-3` (well below the worst-case encoder-drift bound of
//     `sqrt(320) * 1e-4 ~ 1.8e-3`): similarity_scalar matches PyTorch's
//     einsum semantics — the upstream divergence is encoder drift, not
//     a similarity bug (fe3 case B).
//   - delta `>> 1.8e-3`: similarity_scalar is doing something other than
//     `einsum("id,jd->ij", h1, h2)` (transposing axes, applying a scale,
//     using a different accumulation order beyond fp32 tolerance) — fe3
//     case A.
//
// The test prints per-pair diagnostics in all cases. Like the fe1
// kernel-level test, this test does NOT relax tolerance to make a real
// divergence pass; the soft tolerance threshold `1.0e-3` matches the
// encoder-drift accumulation bound and the test reports its own
// per-pair PASS/FAIL line plus an overall exit code.

#include <hikoboshi/algorithms/detail/pairwise_workspace.hpp>
#include <hikoboshi/algorithms/pairwise.hpp>
#include <hikoboshi/modules/esm2.hpp>
#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>
#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_m = hikoboshi::modules;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

[[noreturn]] void fail(const std::string& detail) {
  std::fprintf(stderr, "similarity_scalar_pytorch_parity: %s\n",
               detail.c_str());
  std::exit(1);
}

// === ESM2-8M constants — mirror engine.cpp's `esm2_8m_descriptor`. =========
constexpr std::size_t kEsm2HiddenDim = 320;
constexpr std::size_t kEsm2LayerCount = 6;
constexpr std::size_t kEsm2HeadCount = 20;
constexpr std::size_t kEsm2HeadDim = kEsm2HiddenDim / kEsm2HeadCount;
constexpr std::size_t kEsm2FfnHiddenDim = 1280;
constexpr std::size_t kEsm2VocabSize = 29;
constexpr std::size_t kEsm2MaxSequenceLength = 1024;

// Special-token ids mirror `scripts/esm2_pytorch_goldens.py`. Hikoboshi's
// public Python binding tokenizes raw AA strings without CLS/EOS, but the
// PyTorch reference (and Casey's training pipeline) prepends/appends
// these tokens before running the encoder; matching that wrapping here
// keeps the test honest against fe1's `sim_matrix` goldens.
constexpr std::int32_t kEsm2CLS = 26;
constexpr std::int32_t kEsm2EOS = 27;

// Hard-coded aatype tokens for the three fe1 lead-diagnostic pairs.
// Extracted once via `proteins['<pdb_id>']['aatype'][:]` against
// the private validation corpus under the reference environment
// conda env; values are int8 in the file, all in the 0..19 canonical AA
// range. Inlined here so the test does not require an HDF5 dependency
// or a Python sidecar fixture.
constexpr std::int32_t kPair0Query[] = {
    10, 8, 14, 3, 15, 6, 8, 6, 0, 3, 13, 0, 14, 14, 11, 14,
    9, 0, 17, 0, 9, 6, 3, 9, 0, 15, 9, 7, 12, 0, 3, 18,
    8, 13, 13, 11, 17, 15, 0, 0, 12, 15, 8, 0, 16, 16, 17, 3,
    0, 0, 1, 14, 19, 7, 14, 6, 9, 13, 13, 11, 5, 15, 16,
};
constexpr std::int32_t kPair0Target[] = {
    15, 14, 15, 16, 6, 11, 3, 10, 3, 8, 11, 14, 14, 0, 6, 9,
    14, 9, 15, 9, 3, 8, 9, 8, 5, 9, 17, 12, 9, 5, 12, 2,
    15, 15, 14, 6, 16, 16, 9, 15, 9, 9, 16, 8, 0, 8, 9, 6,
    7, 8, 8, 9, 3, 2, 15, 2, 14, 8, 0, 17, 6, 13, 7, 2,
    13, 9, 13, 14, 3, 13, 14, 6, 9, 8, 14, 13, 9, 3, 8,
};
constexpr std::int32_t kPair1Query[] = {
    14, 15, 6, 15, 9, 6, 19, 9, 4, 10, 5, 0, 15, 3, 13, 2,
    9, 5, 9, 15, 9, 4, 3, 0, 9, 5, 19, 17, 2, 2, 13, 9,
    4, 17, 4, 19, 2, 6, 3, 15, 14, 14, 17, 3, 12, 14, 16, 12,
    18, 17, 15, 15, 14, 7, 15, 15, 13, 10, 18, 9, 13, 9, 15, 13,
    15, 9, 8, 5, 18, 2, 6, 10, 4, 16, 17, 2, 4, 18, 16, 7,
    10, 3, 11, 6, 11, 6, 15, 8, 3, 15, 6, 16, 9, 13, 17, 7,
    9, 5, 1, 3, 10, 13, 3, 2, 11, 15, 16, 3, 5, 19, 18, 8,
    19, 5, 19, 2, 5, 13, 2, 6, 9, 3, 4, 1, 12, 2, 16, 9,
    2, 18, 14, 0, 0, 3, 12, 14, 0, 18, 12, 16, 8, 9, 3, 18,
    3, 14, 6, 8, 7, 14, 0, 14, 13, 11, 14, 0, 19, 9, 3, 14,
    2, 1, 12, 0, 13, 9, 13, 13, 9, 9, 3, 9, 5, 14, 5, 17,
    9, 2,
};
constexpr std::int32_t kPair1Target[] = {
    16, 6, 16, 9, 14, 19, 11, 17, 14, 0, 6, 15, 9, 3, 5, 15,
    3, 8, 16, 13, 9, 9, 17, 9, 7, 19, 17, 2, 3, 3, 9, 4,
    9, 8, 19, 11, 5, 2, 15, 14, 3, 16, 3, 12, 9, 5, 1, 18,
    7, 8, 5, 6, 5, 5, 11, 3, 16, 1, 0, 14, 3, 16, 11, 11,
    9, 9, 8, 17, 3, 3, 8, 9, 14, 5, 10, 10, 0, 3, 17, 7,
    11, 13, 8, 15, 13, 3, 3, 5, 9, 6, 16, 9, 13, 0, 16, 9,
    5, 1, 3, 9, 9, 15, 11, 5, 15, 16, 14, 5, 4, 18, 6, 9,
    5, 19, 2, 5, 13, 11, 4, 9, 16, 4, 2, 13, 8, 16, 9, 16,
    18, 16, 17, 2, 5, 12, 15, 16, 13, 13, 11, 8, 10, 4, 18, 8,
    16, 6, 0, 12, 14, 0, 2, 9, 17, 8, 16, 4, 9, 2, 2, 7,
    1, 12, 0, 6, 9, 13, 14, 19, 9, 0, 15, 9, 14, 11, 5,
};
constexpr std::int32_t kPair2Query[] = {
    13, 2, 9, 0, 14, 7, 3, 13, 4, 9, 2, 0, 9, 18, 9, 3,
    8, 11, 9, 0, 3, 11, 16, 9, 11, 0, 19, 14, 14, 2, 9, 15,
    10, 10, 17, 3, 18, 9, 6, 6, 14, 5, 9, 16, 9, 0, 16, 0,
    13, 15, 2, 2, 9, 13, 0, 9, 9, 0, 3, 14, 9, 15, 15, 0,
    14, 9, 9, 15, 0, 17, 14, 14, 9, 4, 13, 19, 9, 19, 14, 3,
    8, 4, 14, 3, 2, 2, 12, 15, 0, 6, 9,
};
constexpr std::int32_t kPair2Target[] = {
    3, 0, 3, 4, 7, 0, 5, 16, 0, 9, 16, 10, 17, 5, 10, 16,
    9, 17, 5, 9, 0, 7, 5, 4, 17, 9, 9, 14, 17, 3, 15, 9,
    17, 3,
};

struct PairTokens {
  std::string pair_id;
  const std::int32_t* query;
  std::size_t query_len;
  const std::int32_t* target;
  std::size_t target_len;
};

const PairTokens kPairs[] = {
    {"d1a0aa_/d1nlwa_", kPair0Query, sizeof(kPair0Query) / sizeof(*kPair0Query),
     kPair0Target, sizeof(kPair0Target) / sizeof(*kPair0Target)},
    {"d1a6za2/d6a97a1", kPair1Query, sizeof(kPair1Query) / sizeof(*kPair1Query),
     kPair1Target, sizeof(kPair1Target) / sizeof(*kPair1Target)},
    {"d1a0pa1/d1q90m_", kPair2Query, sizeof(kPair2Query) / sizeof(*kPair2Query),
     kPair2Target, sizeof(kPair2Target) / sizeof(*kPair2Target)},
};

// === Minimal ZIP_STORED NPZ reader ========================================
// Identical to the reader in `softsw_posterior_pytorch_parity.cpp`. We
// could hoist this into a shared helper later, but fe2 keeps it inline
// to avoid pulling additional surface into `parity_fixtures.{hpp,cpp}`.

struct NpyEntry {
  std::string dtype;
  std::vector<std::size_t> shape;
  std::vector<unsigned char> data;
};

std::uint16_t read_u16_le(const unsigned char* bytes) noexcept {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::uint32_t read_u32_le(const unsigned char* bytes) noexcept {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint64_t read_u64_le(const unsigned char* bytes) noexcept {
  std::uint64_t lo = read_u32_le(bytes);
  std::uint64_t hi = read_u32_le(bytes + 4);
  return lo | (hi << 32);
}

std::string strip_quotes(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
    value.remove_suffix(1);
  }
  if (value.size() >= 2 && (value.front() == '\'' || value.front() == '"') &&
      value.back() == value.front()) {
    return std::string{value.substr(1, value.size() - 2)};
  }
  return std::string{value};
}

std::string find_string_field(const std::string& header,
                              const std::string& key) {
  const std::string needle = "'" + key + "':";
  const std::size_t pos = header.find(needle);
  if (pos == std::string::npos) {
    return std::string{};
  }
  const std::size_t value_start = pos + needle.size();
  std::size_t value_end = header.find(',', value_start);
  if (value_end == std::string::npos) {
    value_end = header.find('}', value_start);
  }
  if (value_end == std::string::npos) {
    return std::string{};
  }
  return strip_quotes(std::string_view{header}.substr(
      value_start, value_end - value_start));
}

std::vector<std::size_t> parse_shape_field(const std::string& header) {
  const std::string needle = "'shape':";
  const std::size_t pos = header.find(needle);
  if (pos == std::string::npos) {
    return {};
  }
  const std::size_t paren_open = header.find('(', pos);
  const std::size_t paren_close = header.find(')', paren_open);
  if (paren_open == std::string::npos || paren_close == std::string::npos ||
      paren_close <= paren_open) {
    return {};
  }
  std::string body =
      header.substr(paren_open + 1, paren_close - paren_open - 1);
  std::vector<std::size_t> dims;
  std::string token;
  for (char c : body) {
    if (c == ',' || c == ' ' || c == '\t') {
      if (!token.empty()) {
        dims.push_back(static_cast<std::size_t>(std::stoull(token)));
        token.clear();
      }
    } else {
      token.push_back(c);
    }
  }
  if (!token.empty()) {
    dims.push_back(static_cast<std::size_t>(std::stoull(token)));
  }
  return dims;
}

NpyEntry parse_npy(const unsigned char* bytes, std::size_t size) {
  if (size < 10) {
    fail("npy entry truncated below magic+header-len");
  }
  static const unsigned char kMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0) {
    fail("npy entry has wrong magic");
  }
  const std::uint8_t major = bytes[6];
  std::size_t header_offset = 0;
  std::size_t header_len = 0;
  if (major == 1) {
    header_len = read_u16_le(bytes + 8);
    header_offset = 10;
  } else if (major == 2 || major == 3) {
    if (size < 12) {
      fail("npy v2+ entry truncated");
    }
    header_len = read_u32_le(bytes + 8);
    header_offset = 12;
  } else {
    fail("npy entry has unsupported major version");
  }
  if (header_offset + header_len > size) {
    fail("npy header runs past entry boundary");
  }
  std::string header(reinterpret_cast<const char*>(bytes + header_offset),
                     header_len);
  NpyEntry entry{};
  entry.dtype = find_string_field(header, "descr");
  if (entry.dtype.empty()) {
    fail("npy header missing descr field");
  }
  const std::string fortran = find_string_field(header, "fortran_order");
  if (!fortran.empty() && fortran != "False") {
    fail("npy entry must be C-order");
  }
  entry.shape = parse_shape_field(header);
  const std::size_t data_offset = header_offset + header_len;
  const std::size_t data_bytes = size - data_offset;
  entry.data.assign(bytes + data_offset, bytes + data_offset + data_bytes);
  return entry;
}

void read_npz(const std::filesystem::path& path,
              std::vector<std::pair<std::string, NpyEntry>>& out_entries) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    fail("could not open npz fixture: " + path.string());
  }
  const std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
  if (!in.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()))) {
    fail("failed reading npz fixture: " + path.string());
  }
  std::size_t cursor = 0;
  while (cursor + 4 <= bytes.size()) {
    if (read_u32_le(bytes.data() + cursor) != 0x04034b50u) {
      break;
    }
    if (cursor + 30 > bytes.size()) {
      fail("npz local file header truncated");
    }
    const std::uint16_t method = read_u16_le(bytes.data() + cursor + 8);
    if (method != 0) {
      fail("npz fixture must use ZIP_STORED (method=0)");
    }
    std::uint64_t compressed_size =
        read_u32_le(bytes.data() + cursor + 18);
    std::uint64_t uncompressed_size =
        read_u32_le(bytes.data() + cursor + 22);
    const std::uint16_t name_len = read_u16_le(bytes.data() + cursor + 26);
    const std::uint16_t extra_len = read_u16_le(bytes.data() + cursor + 28);
    const std::size_t header_size =
        30 + static_cast<std::size_t>(name_len) + extra_len;
    if (cursor + header_size > bytes.size()) {
      fail("npz local file header + name + extra exceeds archive size");
    }
    if (compressed_size == 0xFFFFFFFFull ||
        uncompressed_size == 0xFFFFFFFFull) {
      const unsigned char* extra = bytes.data() + cursor + 30 + name_len;
      std::size_t extra_pos = 0;
      bool found_zip64 = false;
      while (extra_pos + 4 <= static_cast<std::size_t>(extra_len)) {
        const std::uint16_t tag = read_u16_le(extra + extra_pos);
        const std::uint16_t data_size = read_u16_le(extra + extra_pos + 2);
        if (extra_pos + 4 + data_size >
            static_cast<std::size_t>(extra_len)) {
          break;
        }
        if (tag == 0x0001u) {
          std::size_t inner = 0;
          if (uncompressed_size == 0xFFFFFFFFull && inner + 8 <= data_size) {
            uncompressed_size = read_u64_le(extra + extra_pos + 4 + inner);
            inner += 8;
          }
          if (compressed_size == 0xFFFFFFFFull && inner + 8 <= data_size) {
            compressed_size = read_u64_le(extra + extra_pos + 4 + inner);
            inner += 8;
          }
          found_zip64 = true;
          break;
        }
        extra_pos += 4 + data_size;
      }
      if (!found_zip64) {
        fail("npz local file header marked ZIP64 but lacks a 0x0001 extra");
      }
    }
    if (compressed_size != uncompressed_size) {
      fail("npz fixture has compressed entry where stored is required");
    }
    if (cursor + header_size + compressed_size > bytes.size()) {
      fail("npz entry runs past archive end");
    }
    std::string name(
        reinterpret_cast<const char*>(bytes.data() + cursor + 30), name_len);
    if (name.size() >= 4 &&
        std::string_view{name}.substr(name.size() - 4) ==
            std::string_view{".npy"}) {
      name = name.substr(0, name.size() - 4);
    }
    NpyEntry entry =
        parse_npy(bytes.data() + cursor + header_size, compressed_size);
    out_entries.emplace_back(std::move(name), std::move(entry));
    cursor += header_size + compressed_size;
  }
  if (out_entries.empty()) {
    fail("npz fixture contained no entries: " + path.string());
  }
}

const NpyEntry* find_entry(
    const std::vector<std::pair<std::string, NpyEntry>>& entries,
    const std::string_view name) noexcept {
  for (const auto& entry : entries) {
    if (entry.first == name) {
      return &entry.second;
    }
  }
  return nullptr;
}

const NpyEntry& require_entry(
    const std::vector<std::pair<std::string, NpyEntry>>& entries,
    const std::string& name) {
  const NpyEntry* entry = find_entry(entries, name);
  if (entry == nullptr) {
    fail("npz fixture missing required entry: " + name);
  }
  return *entry;
}

std::int32_t scalar_i32(const NpyEntry& entry, const std::string& name) {
  if (entry.dtype != "<i4" && entry.dtype != "|i4" && entry.dtype != "i4") {
    fail(name + " must be int32 (dtype <i4)");
  }
  if (entry.data.size() != sizeof(std::int32_t)) {
    fail(name + " must be a scalar (4 bytes)");
  }
  std::int32_t value = 0;
  std::memcpy(&value, entry.data.data(), sizeof(std::int32_t));
  return value;
}

float scalar_f32(const NpyEntry& entry, const std::string& name) {
  if (entry.dtype != "<f4" && entry.dtype != "|f4" && entry.dtype != "f4") {
    fail(name + " must be float32 (dtype <f4)");
  }
  if (entry.data.size() != sizeof(float)) {
    fail(name + " must be a scalar (4 bytes)");
  }
  float value = 0.0F;
  std::memcpy(&value, entry.data.data(), sizeof(float));
  return value;
}

std::vector<float> matrix_f32(const NpyEntry& entry,
                              const std::string& name,
                              std::size_t expected_rows,
                              std::size_t expected_cols) {
  if (entry.dtype != "<f4" && entry.dtype != "|f4" && entry.dtype != "f4") {
    fail(name + " must be float32 (dtype <f4)");
  }
  if (entry.shape.size() != 2 || entry.shape[0] != expected_rows ||
      entry.shape[1] != expected_cols) {
    fail(name + " has unexpected shape; expected (" +
         std::to_string(expected_rows) + ", " +
         std::to_string(expected_cols) + ")");
  }
  const std::size_t count = expected_rows * expected_cols;
  if (entry.data.size() != count * sizeof(float)) {
    fail(name + " byte length does not match shape");
  }
  std::vector<float> values(count);
  std::memcpy(values.data(), entry.data.data(), entry.data.size());
  return values;
}

std::filesystem::path fixture_path() {
  const char* explicit_path = std::getenv("HIKOBOSHI_SOFTSW_PYTORCH_GOLDENS");
  if (explicit_path != nullptr && *explicit_path != '\0') {
    return std::filesystem::path{explicit_path};
  }
  const char* source_root = std::getenv("HIKOBOSHI_SOURCE_ROOT");
  if (source_root != nullptr && *source_root != '\0') {
    return std::filesystem::path{source_root} / "tests" / "cpp" / "data" /
           "softsw_posterior_goldens.npz";
  }
  return std::filesystem::path{"tests/cpp/data/softsw_posterior_goldens.npz"};
}

// === Descriptor + workspace setup =========================================

hiko_m::Esm2Descriptor esm2_8m_descriptor(std::size_t seq_len) noexcept {
  hiko_m::Esm2Descriptor descriptor{};
  descriptor.vocab_size = kEsm2VocabSize;
  descriptor.hidden_dimension = kEsm2HiddenDim;
  descriptor.layer_count = kEsm2LayerCount;
  descriptor.head_count = kEsm2HeadCount;
  descriptor.head_dim = kEsm2HeadDim;
  descriptor.ffn_hidden_dimension = kEsm2FfnHiddenDim;
  descriptor.max_sequence_length = std::max(seq_len, kEsm2MaxSequenceLength);
  return descriptor;
}

// === Comparison reporting =================================================

struct DeltaReport {
  float max_abs = 0.0F;
  std::size_t worst_i = 0;
  std::size_t worst_j = 0;
  float worst_hikoboshi = 0.0F;
  float worst_pytorch = 0.0F;
  double l2 = 0.0;
};

DeltaReport compute_delta(const float* hiko_values, const float* golden,
                          std::size_t lq, std::size_t lt) {
  DeltaReport report{};
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < lq; ++i) {
    for (std::size_t j = 0; j < lt; ++j) {
      const std::size_t cell = i * lt + j;
      const float diff = std::fabs(hiko_values[cell] - golden[cell]);
      sum_sq += static_cast<double>(diff) * static_cast<double>(diff);
      if (diff > report.max_abs) {
        report.max_abs = diff;
        report.worst_i = i;
        report.worst_j = j;
        report.worst_hikoboshi = hiko_values[cell];
        report.worst_pytorch = golden[cell];
      }
    }
  }
  report.l2 = std::sqrt(sum_sq);
  return report;
}

struct WorstCell {
  std::size_t i;
  std::size_t j;
  float hikoboshi;
  float pytorch;
  float abs_delta;
};

std::vector<WorstCell> collect_top10_worst(const float* hiko_values,
                                           const float* golden,
                                           std::size_t lq, std::size_t lt) {
  std::vector<WorstCell> cells;
  cells.reserve(lq * lt);
  for (std::size_t i = 0; i < lq; ++i) {
    for (std::size_t j = 0; j < lt; ++j) {
      const std::size_t cell = i * lt + j;
      WorstCell entry{};
      entry.i = i;
      entry.j = j;
      entry.hikoboshi = hiko_values[cell];
      entry.pytorch = golden[cell];
      entry.abs_delta = std::fabs(entry.hikoboshi - entry.pytorch);
      cells.push_back(entry);
    }
  }
  std::sort(cells.begin(), cells.end(),
            [](const WorstCell& a, const WorstCell& b) {
              return a.abs_delta > b.abs_delta;
            });
  if (cells.size() > 10) {
    cells.resize(10);
  }
  return cells;
}

struct RowColSummary {
  float hiko_max = 0.0F;
  float pt_max = 0.0F;
  double hiko_mean = 0.0;
  double pt_mean = 0.0;
};

RowColSummary summarize_marginals(const float* hiko_values, const float* golden,
                                  std::size_t outer, std::size_t inner,
                                  bool row_major_outer) {
  RowColSummary out{};
  const std::size_t lq = row_major_outer ? outer : inner;
  const std::size_t lt = row_major_outer ? inner : outer;
  for (std::size_t o = 0; o < outer; ++o) {
    double hiko_sum = 0.0;
    double pt_sum = 0.0;
    for (std::size_t i = 0; i < inner; ++i) {
      const std::size_t cell =
          row_major_outer ? o * lt + i : i * lt + o;
      (void)lq;
      hiko_sum += static_cast<double>(hiko_values[cell]);
      pt_sum += static_cast<double>(golden[cell]);
    }
    out.hiko_mean += hiko_sum;
    out.pt_mean += pt_sum;
    out.hiko_max = std::max(out.hiko_max, static_cast<float>(hiko_sum));
    out.pt_max = std::max(out.pt_max, static_cast<float>(pt_sum));
  }
  if (outer > 0) {
    out.hiko_mean /= static_cast<double>(outer);
    out.pt_mean /= static_cast<double>(outer);
  }
  return out;
}

}  // namespace

int main() {
  const std::filesystem::path fixture = fixture_path();
  std::vector<std::pair<std::string, NpyEntry>> entries;
  read_npz(fixture, entries);

  const std::int32_t pair_count =
      scalar_i32(require_entry(entries, "pair_count"), "pair_count");
  const float gap_open =
      scalar_f32(require_entry(entries, "gap_open"), "gap_open");
  const float gap_ext =
      scalar_f32(require_entry(entries, "gap_ext"), "gap_ext");
  const float temperature =
      scalar_f32(require_entry(entries, "temperature"), "temperature");
  if (pair_count != static_cast<std::int32_t>(sizeof(kPairs) / sizeof(*kPairs))) {
    fail("npz pair count does not match inlined token-pair count");
  }

  std::printf(
      "similarity_scalar_pytorch_parity: pairs=%d gap_open=%.6f gap_ext=%.6f "
      "temperature=%.6f\n",
      pair_count, static_cast<double>(gap_open), static_cast<double>(gap_ext),
      static_cast<double>(temperature));

  // fe3 lock: tightened to the public ESM2-8M fast-mode encoder parity
  // contract (`<= 1e-4` per cell). The conservative compounded
  // encoder-drift bound on a 320-dim dot product is `sqrt(320) * 1e-4
  // ~ 1.8e-3`, but fe2's observed worst-case sim_matrix delta across
  // the three lead pairs is `1.264e-05` (pair 1, 178x175 sim_matrix)
  // — an order of magnitude below the contract bound and three orders
  // below the legacy `2e-3` admit-encoder-drift threshold. Holding the
  // bar at `1e-4` rejects any future structural similarity_scalar bug
  // while still admitting the empirically-small fp32 accumulation
  // residual seen on the largest pair.
  const float similarity_tolerance = 1.0e-4F;
  bool any_failed = false;

  const hiko_u::Result<hiko_u::WeightsHandle> weights_handle = hiko_w::default_esm2_8m();
  if (weights_handle.status.code != hiko_u::StatusCode::Ok) {
    fail("default_esm2_8m() must resolve to a weights handle");
  }
  if (weights_handle.value.view == nullptr) {
    fail("default_esm2_8m() returned a null weights view");
  }
  const hiko_u::WeightsView& weights_view = *weights_handle.value.view;

  for (std::int32_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    const std::string prefix = "pair_" + std::to_string(pair_index) + "_";
    const std::string lq_key = prefix + "lq";
    const std::string lt_key = prefix + "lt";
    const std::string sim_key = prefix + "sim_matrix";
    const NpyEntry& lq_entry = require_entry(entries, lq_key);
    const NpyEntry& lt_entry = require_entry(entries, lt_key);
    const NpyEntry& sim_entry = require_entry(entries, sim_key);
    const std::size_t lq =
        static_cast<std::size_t>(scalar_i32(lq_entry, lq_key));
    const std::size_t lt =
        static_cast<std::size_t>(scalar_i32(lt_entry, lt_key));
    const std::vector<float> golden_sim_matrix =
        matrix_f32(sim_entry, sim_key, lq, lt);

    const PairTokens& tokens = kPairs[pair_index];
    if (tokens.query_len != lq || tokens.target_len != lt) {
      fail("inlined token-pair length disagrees with fe1 npz lq/lt");
    }

    // Wrap the raw aatype with CLS + EOS so Hikoboshi's encoder sees the
    // same input shape PyTorch saw in `scripts/softsw_pytorch_goldens.py`.
    // The residue positions of the Hikoboshi output map to indices [1..L+1)
    // (a.k.a. PyTorch's `embeddings[1:-1]`).
    std::vector<std::int32_t> query_wrapped;
    query_wrapped.reserve(lq + 2);
    query_wrapped.push_back(kEsm2CLS);
    query_wrapped.insert(query_wrapped.end(), tokens.query,
                         tokens.query + lq);
    query_wrapped.push_back(kEsm2EOS);

    std::vector<std::int32_t> target_wrapped;
    target_wrapped.reserve(lt + 2);
    target_wrapped.push_back(kEsm2CLS);
    target_wrapped.insert(target_wrapped.end(), tokens.target,
                          tokens.target + lt);
    target_wrapped.push_back(kEsm2EOS);

    const hiko_m::Esm2Descriptor descriptor =
        esm2_8m_descriptor(std::max(query_wrapped.size(),
                                    target_wrapped.size()));

    std::vector<float> query_wrapped_embeddings(
        query_wrapped.size() * kEsm2HiddenDim, 0.0F);
    hiko_u::Status status = hiko::detail::encode_esm2_sequence(
        weights_view, descriptor,
        hiko_u::Span<const std::int32_t>{query_wrapped.data(),
                                      query_wrapped.size()},
        query_wrapped_embeddings.data());
    if (!hiko_u::is_ok(status)) {
      std::string detail = "query encode failed";
      if (status.detail != nullptr && *status.detail != '\0') {
        detail += std::string(": ") + status.detail;
      }
      fail(detail);
    }

    std::vector<float> target_wrapped_embeddings(
        target_wrapped.size() * kEsm2HiddenDim, 0.0F);
    status = hiko::detail::encode_esm2_sequence(
        weights_view, descriptor,
        hiko_u::Span<const std::int32_t>{target_wrapped.data(),
                                      target_wrapped.size()},
        target_wrapped_embeddings.data());
    if (!hiko_u::is_ok(status)) {
      std::string detail = "target encode failed";
      if (status.detail != nullptr && *status.detail != '\0') {
        detail += std::string(": ") + status.detail;
      }
      fail(detail);
    }

    // Slice off CLS (position 0) and EOS (last position) to recover the
    // L residue rows that fe1's `sim_matrix` golden is keyed on.
    std::vector<float> query_residue_embeddings(lq * kEsm2HiddenDim, 0.0F);
    std::memcpy(query_residue_embeddings.data(),
                query_wrapped_embeddings.data() + kEsm2HiddenDim,
                lq * kEsm2HiddenDim * sizeof(float));
    std::vector<float> target_residue_embeddings(lt * kEsm2HiddenDim, 0.0F);
    std::memcpy(target_residue_embeddings.data(),
                target_wrapped_embeddings.data() + kEsm2HiddenDim,
                lt * kEsm2HiddenDim * sizeof(float));

    hiko::detail::PairwiseWorkspacePlan plan{};
    plan.max_query_length = lq;
    plan.max_target_length = lt;
    plan.embedding_dimension = kEsm2HiddenDim;
    plan.allocate_mpnn = false;
    plan.allocate_soft_sw = true;

    hiko::detail::PairwiseWorkspace workspace{};
    status = workspace.prepare(plan);
    if (!hiko_u::is_ok(status)) {
      std::string detail = "pairwise workspace prepare failed";
      if (status.detail != nullptr && *status.detail != '\0') {
        detail += std::string(": ") + status.detail;
      }
      fail(detail);
    }

    hiko_u::EmbeddingView query_view{};
    query_view.residue_count = lq;
    query_view.dimension = kEsm2HiddenDim;
    query_view.values = {query_residue_embeddings.data(),
                         query_residue_embeddings.size()};
    query_view.residue_codes = {};
    query_view.residues = {};

    hiko_u::EmbeddingView target_view{};
    target_view.residue_count = lt;
    target_view.dimension = kEsm2HiddenDim;
    target_view.values = {target_residue_embeddings.data(),
                          target_residue_embeddings.size()};
    target_view.residue_codes = {};
    target_view.residues = {};

    hiko::PairwiseEmbeddingRequest request{};
    request.query_embedding = query_view;
    request.target_embedding = target_view;
    request.options.gap_open = gap_open;
    request.options.gap_extension = gap_ext;
    request.soft_mode = true;
    request.temperature = temperature;

    hiko::PairwiseResult result{};
    status = hiko::run_pairwise_embeddings(request, workspace, result);
    if (!hiko_u::is_ok(status)) {
      std::string detail = "run_pairwise_embeddings failed";
      if (status.detail != nullptr && *status.detail != '\0') {
        detail += std::string(": ") + status.detail;
      }
      fail(detail);
    }

    // After `score_embeddings_and_run_resolved` returns, the workspace
    // similarity buffer still holds the score matrix that
    // `similarity_scalar` wrote and that the soft-SW kernel consumed:
    // the kernel only reads it, never overwrites it. So this is the
    // matrix fe1's golden generator compares against.
    const float* hiko_sim = workspace.similarity_data();

    const DeltaReport delta =
        compute_delta(hiko_sim, golden_sim_matrix.data(), lq, lt);
    const std::vector<WorstCell> top10 =
        collect_top10_worst(hiko_sim, golden_sim_matrix.data(), lq, lt);
    const RowColSummary row_summary = summarize_marginals(
        hiko_sim, golden_sim_matrix.data(), lq, lt, true);
    const RowColSummary col_summary = summarize_marginals(
        hiko_sim, golden_sim_matrix.data(), lt, lq, false);

    const bool passed = delta.max_abs <= similarity_tolerance;
    const char* verdict = passed ? "PASS" : "FAIL";
    if (!passed) {
      any_failed = true;
    }
    std::printf(
        "--- pair %d: %s lq=%zu lt=%zu [%s] ---\n"
        "  sim_matrix max_abs=%.6e at (%zu, %zu) hikoboshi=%.6f pytorch=%.6f\n"
        "  l2=%.6e (against fe1 golden sim_matrix)\n"
        "  row_sums: hikoboshi max=%.4f mean=%.4f | pytorch max=%.4f mean=%.4f\n"
        "  col_sums: hikoboshi max=%.4f mean=%.4f | pytorch max=%.4f mean=%.4f\n"
        "  top-10 worst cells (i, j, hikoboshi, pytorch, |delta|):\n",
        pair_index, tokens.pair_id.c_str(), lq, lt, verdict,
        static_cast<double>(delta.max_abs), delta.worst_i, delta.worst_j,
        static_cast<double>(delta.worst_hikoboshi),
        static_cast<double>(delta.worst_pytorch), delta.l2,
        static_cast<double>(row_summary.hiko_max), row_summary.hiko_mean,
        static_cast<double>(row_summary.pt_max), row_summary.pt_mean,
        static_cast<double>(col_summary.hiko_max), col_summary.hiko_mean,
        static_cast<double>(col_summary.pt_max), col_summary.pt_mean);
    for (const WorstCell& cell : top10) {
      std::printf(
          "    (%zu, %zu) hikoboshi=%.6f pytorch=%.6f |delta|=%.6e\n",
          cell.i, cell.j, static_cast<double>(cell.hikoboshi),
          static_cast<double>(cell.pytorch),
          static_cast<double>(cell.abs_delta));
    }
  }

  if (any_failed) {
    std::fprintf(
        stderr,
        "similarity_scalar_pytorch_parity: Hikoboshi similarity_scalar "
        "diverges from PyTorch's einsum(\"id,jd->ij\", h1, h2) by more than "
        "the conservative encoder-drift bound %.6e on at least one of the "
        "three fe1 lead pairs. The fe2 finding favors fe3 case A "
        "(similarity_scalar bug).\n",
        static_cast<double>(similarity_tolerance));
    return 1;
  }
  std::printf(
      "similarity_scalar_pytorch_parity: similarity_scalar agrees with the "
      "fe1 PyTorch sim_matrix on all pairs within the conservative "
      "encoder-drift bound (%.3e). The fe2 finding favors fe3 case B "
      "(encoder drift through dot product) or case C (plumbing); the Python "
      "encoder-drift script discriminates between B and C.\n",
      static_cast<double>(similarity_tolerance));
  return 0;
}

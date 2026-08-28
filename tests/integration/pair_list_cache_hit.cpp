// Encode-once invariant check for the hikoboshi-esm2-8m pair-list route.
//
// The pair-list contract is that each unique protein across the caller's
// pair list is encoded exactly once, regardless of how many input pairs
// reference it. npc1b adds a debug-only encode counter
// (`hikoboshi::algorithms::pair_list_debug_encode_count`, packet section E);
// this test runs an 8-pair list over 5 unique proteins — every protein
// referenced by several pairs — and asserts the counter equals the unique
// protein count.
//
// The counter is compiled only into debug (`!NDEBUG`) builds; in a release
// build this test self-skips.

#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

#ifndef NDEBUG

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

void fail(const char* tag) {
  std::fprintf(stderr, "pair_list_cache_hit: %s\n", tag);
  std::exit(1);
}

std::int32_t aa_token(char c) noexcept {
  switch (c) {
    case 'A': return 0;   case 'R': return 1;   case 'N': return 2;
    case 'D': return 3;   case 'C': return 4;   case 'Q': return 5;
    case 'E': return 6;   case 'G': return 7;   case 'H': return 8;
    case 'I': return 9;   case 'L': return 10;  case 'K': return 11;
    case 'M': return 12;  case 'F': return 13;  case 'P': return 14;
    case 'S': return 15;  case 'T': return 16;  case 'W': return 17;
    case 'Y': return 18;  case 'V': return 19;  case 'B': return 20;
    case 'U': return 21;  case 'Z': return 22;  case 'O': return 23;
    default:  return 24;  // X / unknown
  }
}

std::vector<std::int32_t> tokenize(std::string_view aa) {
  std::vector<std::int32_t> tokens;
  tokens.reserve(aa.size());
  for (char c : aa) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    tokens.push_back(aa_token(c));
  }
  return tokens;
}

struct Sample {
  const char* name;
  std::string_view sequence;
};

const std::vector<Sample>& samples() {
  static const std::vector<Sample> kSamples = {
      {"oxytocin",    "CYIQNCPLG"},
      {"vasopressin", "CYFQNCPRG"},
      {"glucagon",    "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"},
      {"defensin",    "DCYCRIPACIAGERRYGTCIYQGRLWAFCC"},
      {"insulin_a",   "GIVEQCCTSICSLYQLENYCN"},
  };
  return kSamples;
}

int run() {
  const hiko_u::Result<hiko_w::PackageHandle> package =
      hiko_w::default_esm2_8m_package();
  if (package.status.code != hiko_u::StatusCode::Ok) {
    fail("hikoboshi-esm2-8m package handle must be resolvable");
  }
  hiko::EngineConfig config{};
  config.package = package.value;
  hiko::Engine engine{config};

  const auto& kSamples = samples();
  const std::size_t unique_protein_count = kSamples.size();
  std::vector<std::vector<std::int32_t>> tokens;
  tokens.reserve(unique_protein_count);
  for (const auto& sample : kSamples) {
    tokens.push_back(tokenize(sample.sequence));
  }
  std::vector<hiko::SequenceEntry> entries(unique_protein_count);
  for (std::size_t i = 0; i < unique_protein_count; ++i) {
    entries[i] = {std::string_view{kSamples[i].name},
                  {tokens[i].data(), tokens[i].size()}};
  }

  // Eight pairs that collectively reference all five proteins, each
  // protein appearing in several pairs. Without dedup a naive driver
  // would re-encode up to 16 times; encode-once must perform exactly 5.
  const std::vector<std::pair<std::string, std::string>> pairs = {
      {"oxytocin", "vasopressin"},
      {"glucagon", "defensin"},
      {"insulin_a", "oxytocin"},
      {"vasopressin", "glucagon"},
      {"defensin", "insulin_a"},
      {"oxytocin", "glucagon"},
      {"vasopressin", "defensin"},
      {"glucagon", "insulin_a"},
  };

  hiko::PairListSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};
  request.pairs = pairs;

  hikoboshi::algorithms::pair_list_debug_reset_encode_count();
  const hiko_u::Result<hiko::AllVsAllResult> result =
      engine.collect_pair_list(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    std::fprintf(stderr, "pair_list_cache_hit: collect_pair_list status %d\n",
                 static_cast<int>(result.status.code));
    fail("collect_pair_list must succeed");
  }
  if (result.value.records.size() != pairs.size()) {
    fail("pair-list must emit exactly one record per input pair");
  }

  const std::size_t encode_count =
      hikoboshi::algorithms::pair_list_debug_encode_count();
  if (encode_count != unique_protein_count) {
    std::fprintf(stderr,
                 "pair_list_cache_hit: encoded %zu proteins for %zu pairs "
                 "over %zu unique proteins — encode-once violated\n",
                 encode_count, pairs.size(), unique_protein_count);
    fail("each unique protein must be encoded exactly once");
  }

  std::printf("pair_list_cache_hit: ok (%zu pairs, %zu unique proteins, "
              "%zu encodes)\n",
              pairs.size(), unique_protein_count, encode_count);
  return 0;
}

#endif  // !NDEBUG

}  // namespace

int main() {
#ifdef NDEBUG
  std::printf("pair_list_cache_hit: skipped (release build — the encode "
              "counter is compiled out)\n");
  return 0;
#else
  return run();
#endif
}

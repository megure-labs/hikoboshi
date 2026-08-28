// End-to-end smoke for the Hikoboshi-ESM2-8M pair-list route.
//
// Drives `Engine::collect_pair_list(PairListSequenceRequest)` over a small
// named-sequence fixture plus a caller-supplied pair list and confirms the
// npc1b dedup -> encode-once -> input-order pipeline:
//
//   - `collect_pair_list` returns ok;
//   - there is exactly one output record per input pair;
//   - records are in input order — record k's (query_index, target_index)
//     are the source-span indices of input pair k's (query_id, target_id);
//   - a duplicated input pair produces its own output record;
//   - the pair list is resolved by case-sensitive ID against
//     `SequenceEntry::name`, independent of source order.
//
// Per-pair numeric parity against standalone `pairwise` is owned by
// `pair_list_vs_pairwise_parity`; this smoke only checks shape + ordering.

#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

void fail(const char* tag) {
  std::fprintf(stderr, "pair_list_smoke: %s\n", tag);
  std::exit(1);
}

// Hikoboshi-ESM2-8M compacted 29-row alphabet; mirrors `aa_token` in the
// other esm2-8m integration tests (the api layer cannot reach the weights
// TU that owns the embedded tokenizer table).
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

// Four short well-known peptides. Index order: oxytocin 0, vasopressin 1,
// glucagon 2, defensin 3.
const std::vector<Sample>& samples() {
  static const std::vector<Sample> kSamples = {
      {"oxytocin",    "CYIQNCPLG"},
      {"vasopressin", "CYFQNCPRG"},
      {"glucagon",    "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"},
      {"defensin",    "DCYCRIPACIAGERRYGTCIYQGRLWAFCC"},
  };
  return kSamples;
}

}  // namespace

int main() {
  const hiko_u::Result<hiko_w::PackageHandle> package =
      hiko_w::default_esm2_8m_package();
  if (package.status.code != hiko_u::StatusCode::Ok) {
    fail("Hikoboshi-ESM2-8M package handle must be resolvable");
  }
  hiko::EngineConfig config{};
  config.package = package.value;
  hiko::Engine engine{config};

  const auto& kSamples = samples();
  const std::size_t n = kSamples.size();
  std::vector<std::vector<std::int32_t>> tokens;
  tokens.reserve(n);
  for (const auto& sample : kSamples) {
    tokens.push_back(tokenize(sample.sequence));
  }
  std::vector<hiko::SequenceEntry> entries(n);
  for (std::size_t i = 0; i < n; ++i) {
    entries[i] = {std::string_view{kSamples[i].name},
                  {tokens[i].data(), tokens[i].size()}};
  }

  // Caller pair list (case-sensitive string IDs), deliberately out of
  // source order and with a duplicated pair (rows 1 and 3) to confirm
  // input-order output and that duplicates each produce their own row.
  const std::vector<std::pair<std::string, std::string>> pairs = {
      {"glucagon", "oxytocin"},
      {"oxytocin", "oxytocin"},      // self pair
      {"defensin", "vasopressin"},
      {"oxytocin", "oxytocin"},      // duplicate of row 1
      {"vasopressin", "glucagon"},
  };
  // Expected (query_index, target_index) into the `entries` source span.
  const std::vector<std::pair<std::size_t, std::size_t>> expected = {
      {2, 0}, {0, 0}, {3, 1}, {0, 0}, {1, 2},
  };

  hiko::PairListSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};
  request.pairs = pairs;

  const hiko_u::Result<hiko::AllVsAllResult> result =
      engine.collect_pair_list(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    std::fprintf(stderr, "pair_list_smoke: collect_pair_list status %d\n",
                 static_cast<int>(result.status.code));
    fail("collect_pair_list must succeed");
  }
  if (result.value.records.size() != pairs.size()) {
    std::fprintf(stderr,
                 "pair_list_smoke: expected %zu records, got %zu\n",
                 pairs.size(), result.value.records.size());
    fail("pair-list must emit exactly one record per input pair");
  }
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const hiko::PairwiseResultRecord& record = result.value.records[k];
    if (record.query_index != expected[k].first ||
        record.target_index != expected[k].second) {
      std::fprintf(stderr,
                   "pair_list_smoke: row %zu (%s, %s) -> (%zu, %zu), "
                   "expected (%zu, %zu)\n",
                   k, pairs[k].first.c_str(), pairs[k].second.c_str(),
                   record.query_index, record.target_index,
                   expected[k].first, expected[k].second);
      fail("records must be in input order with correct source indices");
    }
    if (!std::isfinite(record.result.metrics.raw_sw_score)) {
      fail("raw_sw_score must be finite for every pair");
    }
    if (record.result.metrics.raw_sw_score < -1.0e-6) {
      fail("raw_sw_score must be non-negative under hard SW with raw-dot");
    }
  }

  // A pair referencing an absent ID is a fail-fast error, never a skip.
  hiko::PairListSequenceRequest missing_request{};
  missing_request.sequences = {entries.data(), entries.size()};
  missing_request.pairs = {{"oxytocin", "not_a_real_protein"}};
  const hiko_u::Result<hiko::AllVsAllResult> missing =
      engine.collect_pair_list(missing_request);
  if (missing.status.code != hiko_u::StatusCode::InvalidArgument) {
    std::fprintf(stderr,
                 "pair_list_smoke: absent-ID request returned status %d, "
                 "expected InvalidArgument\n",
                 static_cast<int>(missing.status.code));
    fail("a pair referencing an absent ID must fail fast");
  }

  std::printf("pair_list_smoke: ok\n");
  return 0;
}

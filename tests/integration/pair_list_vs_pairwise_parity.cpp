// Pair-list vs standalone-pairwise parity for the Hikoboshi-ESM2-8M route.
//
// For an 8-pair list over 5 unique protein sequences, each pair-list
// output record must be bit-equal — within the public 0.1.0 parity
// contract (<=1e-5 strict / <=1e-4 fast GEMM parity mode) — to running
// `Engine::pairwise` standalone on the same (query, target) pair. Both
// drivers encode through the same ESM2-8M forward pass and align through
// the same hard local affine SW path, so the only thing under test is
// that pair-list's dedup + encode-once cache changes nothing observable
// per pair. Discrete path fields (aligned span, step indices, aligned
// pair count) are compared exactly; floating-point scores within tol.

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
  std::fprintf(stderr, "pair_list_vs_pairwise_parity: %s\n", tag);
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

// GEMM parity mode controls the public floating-point tolerance, mirroring
// `esm2_8m_pairwise_smoke`. Pair-list and pairwise share an encode + align
// path, so in practice the diff is zero; the tolerance is the contract.
double active_score_tolerance() noexcept {
  const char* mode = std::getenv("HIKOBOSHI_GEMM_PARITY_MODE");
  if (mode != nullptr && std::string_view{mode} == "strict") {
    return 1.0e-5;
  }
  return 1.0e-4;
}

struct Sample {
  const char* name;
  std::string_view sequence;
};

// Five protein sequences; index order is the source-span order.
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

std::size_t sample_index(std::string_view name) {
  const auto& kSamples = samples();
  for (std::size_t i = 0; i < kSamples.size(); ++i) {
    if (std::string_view{kSamples[i].name} == name) {
      return i;
    }
  }
  fail("test pair references a sample name not in the fixture");
  return 0;
}

void compare_record(std::size_t row,
                     const hiko::PairwiseResult& pair_list,
                     const hiko::PairwiseResult& pairwise,
                     double tol) {
  if (std::fabs(pair_list.metrics.raw_sw_score -
                pairwise.metrics.raw_sw_score) > tol) {
    std::fprintf(stderr,
                 "pair_list_vs_pairwise_parity: row %zu raw_sw_score "
                 "pair-list %.8f vs pairwise %.8f (tol %.1e)\n",
                 row, pair_list.metrics.raw_sw_score,
                 pairwise.metrics.raw_sw_score, tol);
    fail("per-pair raw_sw_score must match standalone pairwise");
  }
  const hiko::AlignmentPath& pl = pair_list.path;
  const hiko::AlignmentPath& pw = pairwise.path;
  if (pl.aligned_pairs != pw.aligned_pairs ||
      pl.query_start != pw.query_start || pl.query_end != pw.query_end ||
      pl.target_start != pw.target_start ||
      pl.target_end != pw.target_end ||
      pl.steps.size() != pw.steps.size()) {
    std::fprintf(stderr,
                 "pair_list_vs_pairwise_parity: row %zu alignment-path "
                 "shape diverged from standalone pairwise\n",
                 row);
    fail("per-pair alignment path must match standalone pairwise");
  }
  for (std::size_t s = 0; s < pl.steps.size(); ++s) {
    const hiko_u::AlignmentStep& a = pl.steps[s];
    const hiko_u::AlignmentStep& b = pw.steps[s];
    if (a.query_index != b.query_index ||
        a.target_index != b.target_index ||
        std::fabs(static_cast<double>(a.residue_score - b.residue_score)) >
            tol) {
      std::fprintf(stderr,
                   "pair_list_vs_pairwise_parity: row %zu step %zu "
                   "diverged from standalone pairwise\n",
                   row, s);
      fail("per-pair alignment step must match standalone pairwise");
    }
  }
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
  if (n != 5U) {
    fail("fixture must contain exactly 5 unique proteins");
  }
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

  // Eight pairs over the five proteins, with repeats and a self pair so
  // the dedup + encode-once cache is genuinely exercised.
  const std::vector<std::pair<std::string, std::string>> pairs = {
      {"glucagon", "defensin"},
      {"oxytocin", "vasopressin"},
      {"insulin_a", "glucagon"},
      {"defensin", "defensin"},
      {"vasopressin", "insulin_a"},
      {"oxytocin", "glucagon"},
      {"insulin_a", "vasopressin"},
      {"glucagon", "oxytocin"},
  };

  hiko::PairListSequenceRequest pair_list_request{};
  pair_list_request.sequences = {entries.data(), entries.size()};
  pair_list_request.pairs = pairs;
  // Default AllVsAllOptions selects hard SW; match it for the standalone
  // pairwise comparison below.
  const hiko_u::Result<hiko::AllVsAllResult> pair_list =
      engine.collect_pair_list(pair_list_request);
  if (pair_list.status.code != hiko_u::StatusCode::Ok) {
    std::fprintf(stderr,
                 "pair_list_vs_pairwise_parity: collect_pair_list status "
                 "%d\n",
                 static_cast<int>(pair_list.status.code));
    fail("collect_pair_list must succeed");
  }
  if (pair_list.value.records.size() != pairs.size()) {
    fail("pair-list must emit exactly one record per input pair");
  }

  const double tol = active_score_tolerance();
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const std::size_t query = sample_index(pairs[k].first);
    const std::size_t target = sample_index(pairs[k].second);

    hiko::PairwiseSequenceRequest pairwise_request{};
    pairwise_request.query_token_ids = {tokens[query].data(),
                                        tokens[query].size()};
    pairwise_request.target_token_ids = {tokens[target].data(),
                                         tokens[target].size()};
    pairwise_request.mode = hiko::AlignmentMode::Hard;
    const hiko_u::Result<hiko::PairwiseResult> pairwise =
        engine.pairwise(pairwise_request);
    if (pairwise.status.code != hiko_u::StatusCode::Ok) {
      fail("standalone pairwise must succeed");
    }
    compare_record(k, pair_list.value.records[k].result, pairwise.value, tol);
  }

  std::printf("pair_list_vs_pairwise_parity: ok (%zu pairs, %zu unique "
              "proteins)\n",
              pairs.size(), n);
  return 0;
}

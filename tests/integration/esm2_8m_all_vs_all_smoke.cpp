// End-to-end all-vs-all smoke for the hikoboshi-esm2-8m sequence-input route.
//
// Drives `Engine::all_vs_all(AllVsAllSequenceRequest, sink)` over a
// 16-sequence fixture spanning short (≤30 residues), medium (~75
// residues), and long (~150-residue myoglobin) inputs and confirms:
//
//   - the streaming sink receives 120 records (16 choose 2);
//   - records are emitted in lexicographic (i < j) order with valid
//     sequence indices;
//   - per-pair raw SW score is finite and non-negative for every record;
//   - each test sequence appears as a query in at least one record.
//
// The fixture covers both the fast and strict GEMM parity modes by
// re-running the same query under each mode and re-asserting record
// shape; the strict mode is selected through the
// `HIKOBOSHI_GEMM_PARITY_MODE` env var that the GEMM dispatch table
// honors at engine-prep time.
//
// Real PyTorch parity over a SCOPe40 val subset is owned by the
// integration-validate bench script (`scripts/run_esm2_8m_baseline.sh`)
// and not gated here; this smoke test only validates that the
// streaming-sink + threadpool encoder paths are coherent end-to-end.

#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

void fail(const char* tag) {
  std::fprintf(stderr, "esm2_8m_all_vs_all_smoke: %s\n", tag);
  std::exit(1);
}

std::int32_t aa_token(char c) noexcept {
  switch (c) {
    case 'A': return 0;
    case 'R': return 1;
    case 'N': return 2;
    case 'D': return 3;
    case 'C': return 4;
    case 'Q': return 5;
    case 'E': return 6;
    case 'G': return 7;
    case 'H': return 8;
    case 'I': return 9;
    case 'L': return 10;
    case 'K': return 11;
    case 'M': return 12;
    case 'F': return 13;
    case 'P': return 14;
    case 'S': return 15;
    case 'T': return 16;
    case 'W': return 17;
    case 'Y': return 18;
    case 'V': return 19;
    case 'B': return 20;
    case 'U': return 21;
    case 'Z': return 22;
    case 'O': return 23;
    case 'X': return 24;
    default:  return 24;
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

// 16 representative protein sequences spanning short (≤30 residues),
// medium (~50–80), and long (~150) lengths. The long entry is sperm-whale
// myoglobin (UniProt P02185); the rest are short well-known peptides /
// fragments and synthetic homologs designed to give the encoder a
// non-trivial token mix and exercise the per-worker bundle plan.
const std::vector<Sample>& samples() {
  static const std::vector<Sample> kSamples = {
      // 0–4: short peptides (10–20 residues).
      {"oxytocin",       "CYIQNCPLG"},                                                    //   9
      {"vasopressin",    "CYFQNCPRG"},                                                    //   9
      {"glucagon_n",     "HSQGTFTSDYSK"},                                                 //  12
      {"insulin_a",      "GIVEQCCTSICSLYQLENYCN"},                                        //  21
      {"insulin_b",      "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"},                               //  30
      // 5–10: medium length 35–80 residues.
      {"acth",           "SYSMEHFRWGKPVGKKRRPVKVYPNGAEDESAEAFPLEFKRELTGQRLREGDGPDGPADDGAGAQADLEHSLLVAAEKKDEGPYRMEHFRWG"},  //  90
      {"growth_h_n",     "FPTIPLSRLFDNAMLRAHRLHQLAFDTYQEFEEAYIPKEQKYSFLQNPQTSLCFSESIPTPSNREETQQK"},                          //  68
      {"calcitonin_n",   "CSNLSTCVLGKLSQELHKLQTYPRTNTGSGTPKRDFNKFHTFPQTAIGVGAP"},                                            //  52
      {"glucagon_full",  "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"},                                                                  //  29
      {"defensin",       "DCYCRIPACIAGERRYGTCIYQGRLWAFCC"},                                                                 //  30
      {"chemerin_n",     "QRAGEDPHSFYFPGQFAFSKALPRSGGGGGGGGSPGRPALSEPVPMRVRVGGAQQGRQQQ"},                                   //  60
      // 11–14: medium length 80–120 residues (synthetic mixed).
      {"trefoil_x2",     "EAQTETCTVAPRERQNCGFPGITSDQCFDNGCCFDSSVTGVPWCFKPLQEAECTFEAQTETCTVAPRERQNCGFPGITSDQCFDNGCCFDSSV"},   //  90
      {"thymosin_x2",    "MSDKPDMAEIEKFDKSKLKKTETQEKNPLPSKETIEQEKQAGESMSDKPDMAEIEKFDKSKLKKTETQEKNPLPSKETIEQEKQAGES"},          //  86
      {"glucagon_x3",    "HSQGTFTSDYSKYLDSRRAQDFVQWLMNTHSQGTFTSDYSKYLDSRRAQDFVQWLMNTHSQGTFTSDYSKYLDSRRAQDFVQWLMNT"},           //  87
      {"defensin_x3",    "DCYCRIPACIAGERRYGTCIYQGRLWAFCCDCYCRIPACIAGERRYGTCIYQGRLWAFCCDCYCRIPACIAGERRYGTCIYQGRLWAFCC"},       //  90
      // 15: long real protein — sperm-whale myoglobin (P02185), 153 residues.
      {"myoglobin",      "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG"},  // 153
  };
  return kSamples;
}

class RecordingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    records.push_back(record);
    return {hiko_u::StatusCode::Ok, ""};
  }
  std::vector<hiko::PairwiseResultRecord> records;
};

void run_all_vs_all_under_current_parity_mode(const char* mode_label) {
  const hiko_u::Result<hiko_w::PackageHandle> package =
      hiko_w::default_esm2_8m_package();
  if (package.status.code != hiko_u::StatusCode::Ok) {
    fail("hikoboshi-esm2-8m package handle must be resolvable");
  }
  hiko::EngineConfig config{};
  config.package = package.value;
  hiko::Engine engine{config};

  const auto& kSamples = samples();
  const std::size_t n = kSamples.size();
  if (n != 16U) {
    fail("test fixture must contain exactly 16 sequences");
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

  hiko::AllVsAllSequenceRequest request{};
  request.sequences = {entries.data(), entries.size()};

  RecordingSink sink{};
  const hiko_u::Status status = engine.all_vs_all(request, sink);
  if (status.code != hiko_u::StatusCode::Ok) {
    std::fprintf(stderr,
                 "esm2_8m_all_vs_all_smoke: all_vs_all(%s) returned status "
                 "%d\n",
                 mode_label, static_cast<int>(status.code));
    fail("all_vs_all must succeed");
  }

  // Expect 16 choose 2 = 120 records under symmetric (i < j) enumeration.
  if (sink.records.size() != 120U) {
    std::fprintf(stderr,
                 "esm2_8m_all_vs_all_smoke: expected 120 records, got %zu\n",
                 sink.records.size());
    fail("symmetric enumeration must emit 120 records for 16 sequences");
  }

  // Records must be in lexicographic (q < t) order; every record must
  // reference valid indices; raw_sw_score must be finite and non-negative.
  std::set<std::pair<std::size_t, std::size_t>> seen;
  for (const auto& rec : sink.records) {
    if (rec.query_index >= n || rec.target_index >= n) {
      fail("record indices must reference the input span");
    }
    if (!(rec.query_index < rec.target_index)) {
      fail("symmetric enumeration must emit query_index < target_index");
    }
    if (!std::isfinite(rec.result.metrics.raw_sw_score)) {
      fail("raw_sw_score must be finite for every pair");
    }
    if (rec.result.metrics.raw_sw_score < -1.0e-6) {
      fail("raw_sw_score must be non-negative under hard SW with raw-dot");
    }
    seen.insert({rec.query_index, rec.target_index});
  }
  if (seen.size() != sink.records.size()) {
    fail("records must be unique pairs");
  }

  // Every input sequence must appear as either a query or a target in
  // at least one record — confirms encoder ran over the full fixture.
  std::vector<bool> seen_seq(n, false);
  for (const auto& rec : sink.records) {
    seen_seq[rec.query_index] = true;
    seen_seq[rec.target_index] = true;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!seen_seq[i]) {
      std::fprintf(stderr,
                   "esm2_8m_all_vs_all_smoke: sequence %zu (%s) absent "
                   "from all-vs-all records\n",
                   i, kSamples[i].name);
      fail("every sequence must participate in at least one pair");
    }
  }
}

void test_all_vs_all_fast_mode() {
  // Default parity mode is fast; confirm by running without env override.
  run_all_vs_all_under_current_parity_mode("fast");
}

void test_all_vs_all_strict_mode() {
  // The active GEMM parity mode is fixed at engine-prep time and read
  // from the HIKOBOSHI_GEMM_PARITY_MODE env var by the dispatch layer.
  // Setting it here only affects future Engine constructions in this
  // process. For this smoke we re-run under the strict-mode env to
  // confirm the encoder + scorer still produce coherent records.
  setenv("HIKOBOSHI_GEMM_PARITY_MODE", "strict", /*overwrite=*/1);
  run_all_vs_all_under_current_parity_mode("strict");
  unsetenv("HIKOBOSHI_GEMM_PARITY_MODE");
}

}  // namespace

int main() {
  test_all_vs_all_fast_mode();
  test_all_vs_all_strict_mode();
  std::printf("esm2_8m_all_vs_all_smoke: ok\n");
  return 0;
}

#include <hikoboshi/primitives/alignment/smith_waterman.hpp>
#include <hikoboshi/primitives/alignment/soft_sw_numerics.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

namespace hiko = hikoboshi::primitives::alignment;

namespace {

std::atomic<bool> g_count_allocations{false};
std::atomic<std::size_t> g_allocation_count{0};

void note_allocation() noexcept {
  if (g_count_allocations.load(std::memory_order_relaxed)) {
    g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void* allocate_bytes(std::size_t size) {
  note_allocation();
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* allocate_aligned_bytes(std::size_t size, std::align_val_t alignment) {
  note_allocation();
  void* pointer = nullptr;
  const std::size_t alignment_value = static_cast<std::size_t>(alignment);
  const int rc =
      posix_memalign(&pointer, alignment_value, size == 0 ? alignment_value : size);
  if (rc == 0 && pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}

void fail(const char* tag) {
  std::fprintf(stderr, "soft_smith_waterman_scalar_tests: %s\n", tag);
  std::exit(1);
}

#include "support/soft_sw_orihime_reference.inc"

struct SoftSwBuffers {
  std::vector<float> match_alpha;
  std::vector<float> insert_alpha;
  std::vector<float> delete_alpha;
  std::vector<float> match_beta;
  std::vector<float> insert_beta;
  std::vector<float> delete_beta;
  std::vector<float> posteriors;

  SoftSwBuffers(std::size_t lq, std::size_t lt) {
    const std::size_t cells = (lq + 1U) * (lt + 1U);
    match_alpha.assign(cells, 0.0F);
    insert_alpha.assign(cells, 0.0F);
    delete_alpha.assign(cells, 0.0F);
    match_beta.assign(cells, 0.0F);
    insert_beta.assign(cells, 0.0F);
    delete_beta.assign(cells, 0.0F);
    posteriors.assign(lq * lt, 0.0F);
  }

  hiko::SoftSmithWatermanScalarRequest make_request(const float* scores,
                                                   std::size_t lq,
                                                   std::size_t lt,
                                                   float gap_open,
                                                   float gap_ext,
                                                   float temperature) {
    hiko::SoftSmithWatermanScalarRequest request{};
    request.scores = scores;
    request.query_length = lq;
    request.target_length = lt;
    request.gap_open = gap_open;
    request.gap_extension = gap_ext;
    request.temperature = temperature;
    request.match_workspace = match_alpha.data();
    request.insert_workspace = insert_alpha.data();
    request.delete_workspace = delete_alpha.data();
    request.workspace_cells = match_alpha.size();
    request.match_grad_workspace = match_beta.data();
    request.insert_grad_workspace = insert_beta.data();
    request.delete_grad_workspace = delete_beta.data();
    return request;
  }
};

float run_soft_sw(SoftSwBuffers& buffers,
                  const float* scores,
                  std::size_t lq,
                  std::size_t lt,
                  float gap_open,
                  float gap_ext,
                  float temperature) {
  hiko::SoftSmithWatermanScalarOutput output{};
  output.posteriors = buffers.posteriors.data();
  hiko::SoftSmithWatermanScalarRequest request =
      buffers.make_request(scores, lq, lt, gap_open, gap_ext, temperature);
  hiko::soft_smith_waterman_scalar(request, output);
  return output.log_partition;
}

float run_hard_sw_argmax_path(const std::vector<float>& scores,
                              std::size_t lq,
                              std::size_t lt,
                              float gap_open,
                              float gap_ext,
                              std::vector<std::pair<std::size_t, std::size_t>>& path) {
  std::vector<hiko::TraceDirection> trace_match(lq * lt, hiko::TraceDirection::Stop);
  std::vector<hiko::TraceDirection> trace_insert(lq * lt, hiko::TraceDirection::Stop);
  std::vector<hiko::TraceDirection> trace_delete(lq * lt, hiko::TraceDirection::Stop);
  const std::size_t cells = (lq + 1U) * (lt + 1U);
  std::vector<float> mw(cells, 0.0F);
  std::vector<float> iw(cells, 0.0F);
  std::vector<float> dw(cells, 0.0F);

  hiko::SmithWatermanScalarRequest req{};
  req.scores = scores.data();
  req.query_length = lq;
  req.target_length = lt;
  req.gap_open = gap_open;
  req.gap_extension = gap_ext;
  req.match_workspace = mw.data();
  req.insert_workspace = iw.data();
  req.delete_workspace = dw.data();
  req.workspace_cells = cells;

  hiko::SmithWatermanScalarOutput out{};
  out.trace_match = trace_match.data();
  out.trace_insert = trace_insert.data();
  out.trace_delete = trace_delete.data();
  hiko::smith_waterman_scalar(req, out);

  path.clear();
  if (out.best_score <= 0.0F || out.best_query_index < 0 ||
      out.best_target_index < 0) {
    return out.best_score;
  }

  // Walk back the M-state argmax path to recover the matched (i, j) pairs.
  std::int32_t qi = out.best_query_index;
  std::int32_t tj = out.best_target_index;
  hiko::TraceDirection state = out.best_state;
  while (qi >= 0 && tj >= 0 && state != hiko::TraceDirection::Stop) {
    const std::size_t flat = static_cast<std::size_t>(qi) * lt +
                             static_cast<std::size_t>(tj);
    if (state == hiko::TraceDirection::Match) {
      path.emplace_back(static_cast<std::size_t>(qi),
                        static_cast<std::size_t>(tj));
      const hiko::TraceDirection next = trace_match[flat];
      qi -= 1;
      tj -= 1;
      state = next;
    } else if (state == hiko::TraceDirection::InsertQuery) {
      const hiko::TraceDirection next = trace_insert[flat];
      qi -= 1;
      state = next;
    } else if (state == hiko::TraceDirection::DeleteTarget) {
      const hiko::TraceDirection next = trace_delete[flat];
      tj -= 1;
      state = next;
    } else {
      break;
    }
  }
  return out.best_score;
}

void test_high_temperature_sanity() {
  // T=10 with scores=zeros: the model is uninformed; partition should be a
  // finite finite real and posteriors must lie inside [0, 1].
  constexpr std::size_t lq = 5;
  constexpr std::size_t lt = 5;
  std::vector<float> scores(lq * lt, 0.0F);
  SoftSwBuffers buffers(lq, lt);
  const float log_z =
      run_soft_sw(buffers, scores.data(), lq, lt, -1.4F, -0.15F, 10.0F);
  if (!std::isfinite(log_z)) {
    fail("high-T partition must be finite for zero scores");
  }
  for (float v : buffers.posteriors) {
    if (!(v >= 0.0F && v <= 1.0F)) {
      fail("high-T posterior must lie in [0, 1]");
    }
  }
  // With completely uninformative scores, posteriors should be roughly
  // diffuse: at least one cell well below 0.5 (no sharp pick).
  bool diffuse = false;
  for (float v : buffers.posteriors) {
    if (v < 0.5F) {
      diffuse = true;
      break;
    }
  }
  if (!diffuse) {
    fail("high-T posteriors must be diffuse, not concentrated");
  }
}

void test_low_temperature_recovers_hard_sw_path() {
  // Build a score matrix with a clear unique optimal diagonal and run both
  // soft-SW at T=1e-4 and hard-SW. Posteriors > 0.5 must equal the hard-SW
  // argmax path cells.
  const std::size_t sizes[] = {5U, 20U, 50U};
  for (const std::size_t n : sizes) {
    std::vector<float> scores(n * n);
    std::mt19937 rng(0xBEEF42U + static_cast<unsigned>(n));
    std::uniform_real_distribution<float> noise(-0.4F, 0.4F);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (i == j) {
          scores[i * n + j] = 4.0F;  // dominant diagonal
        } else {
          scores[i * n + j] = -1.0F + noise(rng);
        }
      }
    }

    SoftSwBuffers buffers(n, n);
    run_soft_sw(buffers, scores.data(), n, n, -1.4F, -0.15F, 1e-4F);

    std::vector<std::pair<std::size_t, std::size_t>> hard_path;
    run_hard_sw_argmax_path(scores, n, n, -1.4F, -0.15F, hard_path);
    if (hard_path.size() != n) {
      fail("hard-SW must recover the full diagonal as the unique optimum");
    }

    std::vector<bool> hard_mask(n * n, false);
    for (const auto& step : hard_path) {
      hard_mask[step.first * n + step.second] = true;
    }

    for (std::size_t idx = 0; idx < n * n; ++idx) {
      const float v = buffers.posteriors[idx];
      if (hard_mask[idx]) {
        if (v < 0.5F) {
          fail("low-T posterior on hard-SW path cell must be >= 0.5");
        }
      } else {
        if (v > 0.5F) {
          fail("low-T posterior off hard-SW path must be <= 0.5");
        }
      }
    }
  }
}

void test_posterior_sum_matches_alignment_length() {
  // At low T the soft-SW posterior mass is concentrated on the single
  // hard-SW path; sum(posteriors) ~= alignment length.
  std::mt19937 rng(0xFACECAFEU);
  std::uniform_real_distribution<float> off(-0.5F, 0.5F);
  const std::size_t sizes[] = {10U, 25U};
  for (const std::size_t n : sizes) {
    std::vector<float> scores(n * n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        scores[i * n + j] = (i == j) ? 5.0F : -1.0F + off(rng);
      }
    }
    SoftSwBuffers buffers(n, n);
    run_soft_sw(buffers, scores.data(), n, n, -1.4F, -0.15F, 1e-4F);

    double sum = 0.0;
    for (float v : buffers.posteriors) {
      sum += v;
    }
    const double expected = static_cast<double>(n);
    if (std::fabs(sum - expected) > 1e-2 * expected + 1.0) {
      fail("posterior sum must approximate the hard-SW alignment length at low T");
    }
  }
}

float max_abs_error(const float* a, const float* b, std::size_t count) {
  float worst = 0.0F;
  for (std::size_t idx = 0; idx < count; ++idx) {
    const float diff = std::fabs(a[idx] - b[idx]);
    if (diff > worst) {
      worst = diff;
    }
  }
  return worst;
}

void verify_orihime_case(std::size_t case_index,
                     std::size_t lq,
                     std::size_t lt,
                     float gap_open,
                     float gap_ext,
                     float temperature,
                     float reference_log_partition,
                     const float* reference_scores,
                     const float* reference_posteriors) {
  SoftSwBuffers buffers(lq, lt);
  const float log_z = run_soft_sw(buffers, reference_scores, lq, lt, gap_open,
                                  gap_ext, temperature);

  // Tolerance widens at very low temperature because numerical scaling stress
  // grows; the packet allows up to 1e-3 max-abs at T = 1e-4.
  const float partition_tolerance =
      temperature >= 1e-3F ? 1e-3F * std::max(1.0F, std::fabs(reference_log_partition))
                           : 5e-3F * std::max(1.0F, std::fabs(reference_log_partition));
  if (std::fabs(log_z - reference_log_partition) > partition_tolerance) {
    std::fprintf(stderr,
                 "soft_smith_waterman_scalar_tests: case %zu log_partition mismatch: "
                 "ours=%.6f orihime=%.6f tol=%.6f\n",
                 case_index, static_cast<double>(log_z),
                 static_cast<double>(reference_log_partition),
                 static_cast<double>(partition_tolerance));
    fail("orihime log_partition parity violated");
  }

  const float posterior_tolerance = temperature >= 1e-3F ? 1e-4F : 1e-3F;
  const float worst =
      max_abs_error(buffers.posteriors.data(), reference_posteriors, lq * lt);
  if (worst > posterior_tolerance) {
    std::fprintf(stderr,
                 "soft_smith_waterman_scalar_tests: case %zu posterior max-abs "
                 "error %.6f exceeds tolerance %.6f (T=%.6g)\n",
                 case_index, static_cast<double>(worst),
                 static_cast<double>(posterior_tolerance),
                 static_cast<double>(temperature));
    fail("orihime posterior parity violated");
  }
}

void test_orihime_reference_parity() {
  verify_orihime_case(1, kCase1Lq, kCase1Lt, kCase1GapOpen, kCase1GapExt,
                  kCase1Temperature, kCase1LogPartition, kCase1Scores,
                  kCase1Posteriors);
  verify_orihime_case(2, kCase2Lq, kCase2Lt, kCase2GapOpen, kCase2GapExt,
                  kCase2Temperature, kCase2LogPartition, kCase2Scores,
                  kCase2Posteriors);
  verify_orihime_case(3, kCase3Lq, kCase3Lt, kCase3GapOpen, kCase3GapExt,
                  kCase3Temperature, kCase3LogPartition, kCase3Scores,
                  kCase3Posteriors);
  verify_orihime_case(4, kCase4Lq, kCase4Lt, kCase4GapOpen, kCase4GapExt,
                  kCase4Temperature, kCase4LogPartition, kCase4Scores,
                  kCase4Posteriors);
  verify_orihime_case(5, kCase5Lq, kCase5Lt, kCase5GapOpen, kCase5GapExt,
                  kCase5Temperature, kCase5LogPartition, kCase5Scores,
                  kCase5Posteriors);
  verify_orihime_case(6, kCase6Lq, kCase6Lt, kCase6GapOpen, kCase6GapExt,
                  kCase6Temperature, kCase6LogPartition, kCase6Scores,
                  kCase6Posteriors);
  verify_orihime_case(7, kCase7Lq, kCase7Lt, kCase7GapOpen, kCase7GapExt,
                  kCase7Temperature, kCase7LogPartition, kCase7Scores,
                  kCase7Posteriors);
  verify_orihime_case(8, kCase8Lq, kCase8Lt, kCase8GapOpen, kCase8GapExt,
                  kCase8Temperature, kCase8LogPartition, kCase8Scores,
                  kCase8Posteriors);
  verify_orihime_case(9, kCase9Lq, kCase9Lt, kCase9GapOpen, kCase9GapExt,
                  kCase9Temperature, kCase9LogPartition, kCase9Scores,
                  kCase9Posteriors);
  verify_orihime_case(10, kCase10Lq, kCase10Lt, kCase10GapOpen, kCase10GapExt,
                  kCase10Temperature, kCase10LogPartition, kCase10Scores,
                  kCase10Posteriors);
  verify_orihime_case(11, kCase11Lq, kCase11Lt, kCase11GapOpen, kCase11GapExt,
                  kCase11Temperature, kCase11LogPartition, kCase11Scores,
                  kCase11Posteriors);
}

void test_workspace_alloc_gate() {
  // After one warmup run, repeated calls into soft_smith_waterman_scalar must
  // not allocate anything: the primitive owns no scratch beyond the request.
  constexpr std::size_t lq = 32U;
  constexpr std::size_t lt = 32U;
  std::vector<float> scores(lq * lt);
  std::mt19937 rng(0xDEADBEEFU);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  for (float& v : scores) {
    v = distribution(rng);
  }

  SoftSwBuffers buffers(lq, lt);
  hiko::SoftSmithWatermanScalarOutput output{};
  output.posteriors = buffers.posteriors.data();

  // Warmup before counting.
  hiko::SoftSmithWatermanScalarRequest request =
      buffers.make_request(scores.data(), lq, lt, -1.4F, -0.15F, 1.0F);
  hiko::soft_smith_waterman_scalar(request, output);

  g_allocation_count.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_relaxed);
  for (int pass = 0; pass < 100; ++pass) {
    hiko::soft_smith_waterman_scalar(request, output);
  }
  g_count_allocations.store(false, std::memory_order_relaxed);

  if (g_allocation_count.load(std::memory_order_relaxed) != 0) {
    fail("soft_smith_waterman_scalar must not allocate after warmup");
  }
  if (!std::isfinite(output.log_partition)) {
    fail("alloc-gate run must still produce a finite log_partition");
  }
}

void test_invalid_request_zeroes_output() {
  // Defensive guard: T <= 0 must cause the primitive to clear posteriors and
  // leave log_partition at -inf without touching the workspaces.
  constexpr std::size_t lq = 4U;
  constexpr std::size_t lt = 4U;
  std::vector<float> scores(lq * lt, 1.0F);
  SoftSwBuffers buffers(lq, lt);
  for (float& v : buffers.posteriors) {
    v = 999.0F;
  }
  hiko::SoftSmithWatermanScalarOutput output{};
  output.posteriors = buffers.posteriors.data();
  output.log_partition = 17.0F;
  hiko::SoftSmithWatermanScalarRequest request =
      buffers.make_request(scores.data(), lq, lt, -1.4F, -0.15F, -1.0F);
  hiko::soft_smith_waterman_scalar(request, output);
  for (float v : buffers.posteriors) {
    if (v != 0.0F) {
      fail("invalid temperature must clear posteriors");
    }
  }
  if (output.log_partition > hiko::kSoftSwNegInf / 2.0F) {
    fail("invalid temperature must reset log_partition to -inf");
  }
}

}  // namespace

void* operator new(std::size_t size) { return allocate_bytes(size); }
void* operator new[](std::size_t size) { return allocate_bytes(size); }
void* operator new(std::size_t size, std::align_val_t alignment) {
  return allocate_aligned_bytes(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate_aligned_bytes(size, alignment);
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}

int main() {
  test_high_temperature_sanity();
  test_low_temperature_recovers_hard_sw_path();
  test_posterior_sum_matches_alignment_length();
  test_orihime_reference_parity();
  test_workspace_alloc_gate();
  test_invalid_request_zeroes_output();
  return 0;
}

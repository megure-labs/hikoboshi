#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/algorithms/detail/pair_scheduler.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::algorithms;
namespace hiko_ad = hikoboshi::algorithms::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_ud = hikoboshi::universal::detail;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_query_tiled_scheduler_tests: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double actual, double expected,
                  double tolerance = 1.0e-6) noexcept {
  return std::fabs(actual - expected) <= tolerance;
}

// --- Tile partition correctness ----------------------------------------

void test_ij_to_pair_index_round_trip(bool include_self) {
  for (std::size_t item_count : {2U, 3U, 5U, 8U, 13U}) {
    const std::size_t pair_count =
        hiko_ad::symmetric_pair_count(item_count, include_self);
    for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
      const hiko_ad::PairIndex pair =
          hiko_ad::pair_index_to_ij(pair_index, item_count, include_self);
      const std::size_t round_trip = hiko_ad::ij_to_pair_index(
          pair.query_index, pair.target_index, item_count, include_self);
      if (round_trip != pair_index) {
        fail("ij_to_pair_index is not the inverse of pair_index_to_ij");
      }
    }
  }
}

void test_partition_covers_each_pair_exactly_once(bool include_self,
                                                  std::size_t query_tile,
                                                  std::size_t target_tile) {
  for (std::size_t item_count : {0U, 1U, 2U, 3U, 5U, 8U, 13U, 31U}) {
    std::vector<hiko_ad::PairTile> tiles;
    hiko_ad::partition_pair_tiles(item_count, include_self, query_tile,
                               target_tile, tiles);
    const std::size_t expected_pair_count =
        hiko_ad::symmetric_pair_count(item_count, include_self);

    std::vector<unsigned char> seen(expected_pair_count, 0);
    std::size_t total_pairs_visited = 0;
    for (const hiko_ad::PairTile& tile : tiles) {
      hiko_ad::iterate_pair_tile(
          tile, item_count, include_self,
          [&](std::size_t pair_index, std::size_t q,
              std::size_t t) noexcept {
            if (pair_index >= expected_pair_count) {
              fail("tile iteration emitted out-of-range pair_index");
            }
            if (q >= item_count || t >= item_count) {
              fail("tile iteration emitted out-of-range coordinates");
            }
            if (include_self) {
              if (q > t) {
                fail("tile iteration violated upper-triangular ordering");
              }
            } else {
              if (q >= t) {
                fail("tile iteration violated strict upper-triangular ordering");
              }
            }
            const std::size_t expected =
                hiko_ad::ij_to_pair_index(q, t, item_count, include_self);
            if (expected != pair_index) {
              fail("tile iteration produced inconsistent pair_index");
            }
            if (seen[pair_index] != 0) {
              fail("tile iteration emitted a duplicate pair");
            }
            seen[pair_index] = 1;
            ++total_pairs_visited;
          });
    }
    if (total_pairs_visited != expected_pair_count) {
      fail("tile coverage did not match symmetric pair count");
    }
    for (unsigned char byte : seen) {
      if (byte == 0) {
        fail("tile coverage missed a pair");
      }
    }
  }
}

void test_query_tile_size_one_keeps_one_query_per_tile(bool include_self) {
  constexpr std::size_t kItemCount = 17U;
  constexpr std::size_t kTargetTile = 4U;
  std::vector<hiko_ad::PairTile> tiles;
  hiko_ad::partition_pair_tiles(kItemCount, include_self, /*query_tile=*/1U,
                             kTargetTile, tiles);
  for (const hiko_ad::PairTile& tile : tiles) {
    if (tile.query_end != tile.query_begin + 1U) {
      fail("query_tile=1 must produce single-query tiles");
    }
  }
}

void test_streaming_safe_tile_keeps_pair_id_ascending(bool include_self) {
  // When tiles are emitted with query_tile=1 in lexicographic (Qi, Tj)
  // order, every tile's pair_ids are contiguous and tile order matches
  // pair_id order. The streaming sink relies on this invariant to keep
  // the in-flight pair_id spread bounded by W * T_tile and avoid the
  // p47/p50 rotation deadlock.
  constexpr std::size_t kItemCount = 23U;
  constexpr std::size_t kTargetTile = 3U;
  std::vector<hiko_ad::PairTile> tiles;
  hiko_ad::partition_pair_tiles(kItemCount, include_self, /*query_tile=*/1U,
                             kTargetTile, tiles);
  std::size_t previous_max = 0U;
  bool first = true;
  for (const hiko_ad::PairTile& tile : tiles) {
    std::size_t tile_min = std::numeric_limits<std::size_t>::max();
    std::size_t tile_max = 0U;
    hiko_ad::iterate_pair_tile(
        tile, kItemCount, include_self,
        [&](std::size_t pair_index, std::size_t /*q*/,
            std::size_t /*t*/) noexcept {
          if (pair_index < tile_min) {
            tile_min = pair_index;
          }
          if (pair_index > tile_max) {
            tile_max = pair_index;
          }
        });
    if (tile_min == std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    if (tile_max - tile_min + 1U !=
        static_cast<std::size_t>(tile_max - tile_min + 1U)) {
      fail("query_tile=1 tile produced non-contiguous pair_ids");
    }
    if (!first && tile_min < previous_max) {
      fail("query_tile=1 tiles are not pair_id ascending");
    }
    previous_max = tile_max;
    first = false;
  }
}

void test_query_tiled_plan_picks_sane_tile_size_for_zero_residue() {
  const hiko_ad::QueryTiledPlan plan = hiko_ad::resolve_query_tiled_plan(8U, 0U, 64U);
  if (plan.query_tile == 0U || plan.query_tile > hiko_ad::kMaxQueryTileSize) {
    fail("resolve_query_tiled_plan must clamp Q_tile to [1, kMaxQueryTileSize]");
  }
}

void test_query_tiled_plan_clamps_q_tile_for_long_embeddings() {
  const hiko_ad::QueryTiledPlan plan =
      hiko_ad::resolve_query_tiled_plan(2000U, /*residue_count_max=*/2048U,
                                     /*embedding_dimension=*/64U);
  if (plan.query_tile != 1U) {
    fail("Q_tile must collapse to 1 when L_max exhausts the per-thread budget");
  }
}

void test_query_tiled_plan_grows_q_tile_for_short_embeddings() {
  // For L=64, hidden_dim=64, sizeof(float)=4, embedding bytes per query
  // = 64 * 64 * 4 = 16 KB. Per-thread budget = 256 KB → Q_tile = 16
  // (capped at kMaxQueryTileSize).
  const hiko_ad::QueryTiledPlan plan =
      hiko_ad::resolve_query_tiled_plan(64U, /*residue_count_max=*/64U,
                                     /*embedding_dimension=*/64U);
  if (plan.query_tile == 0U || plan.query_tile > hiko_ad::kMaxQueryTileSize) {
    fail("Q_tile must stay within [1, kMaxQueryTileSize] for short embeddings");
  }
  if (plan.query_tile <= 1U) {
    fail("Q_tile must grow above 1 when the budget allows");
  }
}

// --- End-to-end correctness through run_all_vs_all_embeddings ----------

struct EmbeddingFixture {
  std::vector<float> values;
  std::vector<char> codes;

  hiko_u::EmbeddingView view(std::size_t residue_count,
                          std::size_t dimension) const {
    return {residue_count,
            dimension,
            {values.data(), values.size()},
            {codes.data(), codes.size()},
            {nullptr, 0}};
  }
};

std::vector<EmbeddingFixture> make_embedding_fixtures(
    std::size_t count,
    std::size_t residue_count,
    std::size_t dimension) {
  std::vector<EmbeddingFixture> fixtures;
  fixtures.reserve(count);
  constexpr char kResidues[] = {'A', 'C', 'D', 'E', 'F', 'G'};
  for (std::size_t item = 0; item < count; ++item) {
    EmbeddingFixture fixture{};
    fixture.values.resize(residue_count * dimension);
    fixture.codes.resize(residue_count);
    for (std::size_t residue = 0; residue < residue_count; ++residue) {
      fixture.codes[residue] = kResidues[(item + residue) %
                                         (sizeof(kResidues) /
                                          sizeof(kResidues[0]))];
      for (std::size_t dim = 0; dim < dimension; ++dim) {
        const float diagonal = residue == dim ? 0.25F : 0.0F;
        fixture.values[residue * dimension + dim] =
            0.1F * static_cast<float>((item + 1U) * (residue + 1U) +
                                      (dim + 1U)) +
            diagonal;
      }
    }
    fixtures.push_back(std::move(fixture));
  }
  return fixtures;
}

class CollectingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    records.push_back(record);
    return hiko_u::ok_status();
  }

  std::vector<hiko::PairwiseResultRecord> records;
};

void require_records_match_pair_id_order(
    const std::vector<hiko::PairwiseResultRecord>& records,
    std::size_t item_count,
    bool include_self) {
  const std::size_t expected_count =
      hiko_ad::symmetric_pair_count(item_count, include_self);
  if (records.size() != expected_count) {
    fail("record count mismatch in query_tiled run");
  }
  for (std::size_t pair_index = 0; pair_index < expected_count; ++pair_index) {
    const hiko_ad::PairIndex expected =
        hiko_ad::pair_index_to_ij(pair_index, item_count, include_self);
    if (records[pair_index].query_index != expected.query_index ||
        records[pair_index].target_index != expected.target_index) {
      fail("record (q, t) sequence is not pair_id ascending");
    }
  }
}

void require_records_match_serial(
    const std::vector<hiko::PairwiseResultRecord>& serial,
    const std::vector<hiko::PairwiseResultRecord>& parallel) {
  if (serial.size() != parallel.size()) {
    fail("serial and parallel record counts differ");
  }
  for (std::size_t index = 0; index < serial.size(); ++index) {
    const hiko::PairwiseResultRecord& s = serial[index];
    const hiko::PairwiseResultRecord& p = parallel[index];
    if (s.query_index != p.query_index || s.target_index != p.target_index) {
      fail("serial and parallel record (q, t) order differs");
    }
    if (!nearly_equal(s.result.metrics.raw_sw_score,
                      p.result.metrics.raw_sw_score) ||
        !nearly_equal(s.result.raw_sw_score, p.result.raw_sw_score)) {
      fail("serial and parallel raw SW scores differ");
    }
    if (s.result.path.aligned_pairs != p.result.path.aligned_pairs ||
        s.result.path.steps.size() != p.result.path.steps.size()) {
      fail("serial and parallel paths differ");
    }
    for (std::size_t step = 0; step < s.result.path.steps.size(); ++step) {
      const hiko_u::AlignmentStep& s_step = s.result.path.steps[step];
      const hiko_u::AlignmentStep& p_step = p.result.path.steps[step];
      if (s_step.query_index != p_step.query_index ||
          s_step.target_index != p_step.target_index ||
          !nearly_equal(s_step.residue_score, p_step.residue_score)) {
        fail("serial and parallel path steps differ");
      }
    }
  }
}

void test_parallel_dispatch_matches_serial_baseline(bool include_self) {
  // Item counts chosen to exceed kQueryTiledPairThreshold = 256, so the
  // tile dispatch path is exercised when the binary is built with
  // hikoboshi_allvsall_pair_scheduler=query_tiled. The query_tiled path
  // must produce the same pair-by-pair record sequence as the
  // single-thread serial baseline (which always iterates pair_ids in
  // ascending order through run_serial_embedding_pairs).
  constexpr std::size_t kInputCount = 32U;
  constexpr std::size_t kResidueCount = 6U;
  constexpr std::size_t kDimension = 4U;
  const std::vector<EmbeddingFixture> fixtures =
      make_embedding_fixtures(kInputCount, kResidueCount, kDimension);
  std::vector<hiko_u::EmbeddingView> embeddings;
  embeddings.reserve(fixtures.size());
  for (const EmbeddingFixture& fixture : fixtures) {
    embeddings.push_back(fixture.view(kResidueCount, kDimension));
  }

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};
  request.options.include_self = include_self;

  CollectingSink serial_sink;
  const hiko_u::Status serial_status =
      hiko::run_all_vs_all_embeddings(request, serial_sink);
  if (serial_status.code != hiko_u::StatusCode::Ok) {
    fail("serial all-vs-all run did not return Ok");
  }
  require_records_match_pair_id_order(serial_sink.records, embeddings.size(),
                                      include_self);

  hiko_ud::ThreadPool pool(4U);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  CollectingSink parallel_sink;
  const hiko_u::Status parallel_status = hiko::run_all_vs_all_embeddings(
      request, parallel_sink, &pool, pool.thread_count(),
      {workers.data(), workers.size()});
  if (parallel_status.code != hiko_u::StatusCode::Ok) {
    fail("parallel all-vs-all run did not return Ok");
  }
  require_records_match_pair_id_order(parallel_sink.records,
                                      embeddings.size(), include_self);
  require_records_match_serial(serial_sink.records, parallel_sink.records);
}

void test_dispatch_in_query_tiled_mode_does_not_deadlock_at_rotation_scale() {
  // Builds a fixture small enough to keep the test fast yet large enough
  // that pair_count exceeds the cost-aware scheduling threshold and the
  // query_tiled dispatch path emits multiple tiles per worker. The test
  // is a regression guard for the streaming + tile dispatch interaction.
  constexpr std::size_t kInputCount = 64U;
  constexpr std::size_t kResidueCount = 4U;
  constexpr std::size_t kDimension = 2U;
  const std::vector<EmbeddingFixture> fixtures =
      make_embedding_fixtures(kInputCount, kResidueCount, kDimension);
  std::vector<hiko_u::EmbeddingView> embeddings;
  embeddings.reserve(fixtures.size());
  for (const EmbeddingFixture& fixture : fixtures) {
    embeddings.push_back(fixture.view(kResidueCount, kDimension));
  }

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings.data(), embeddings.size()};

  hiko_ud::ThreadPool pool(8U);
  std::vector<hiko_ad::AllVsAllWorkerWorkspace> workers(pool.thread_count());
  CollectingSink sink;
  const hiko_u::Status status = hiko::run_all_vs_all_embeddings(
      request, sink, &pool, pool.thread_count(),
      {workers.data(), workers.size()});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail("query_tiled-mode all-vs-all did not return Ok");
  }
  const std::size_t expected_count =
      hiko_ad::symmetric_pair_count(embeddings.size(), /*include_self=*/false);
  if (sink.records.size() != expected_count) {
    fail("query_tiled-mode emitted record count mismatch");
  }
  require_records_match_pair_id_order(sink.records, embeddings.size(),
                                      /*include_self=*/false);
}

// --- Locality assertion -------------------------------------------------

// Counts unique queries observed within each tile. With Q_tile=1 the
// answer is exactly 1 per tile; with Q_tile=4 it is at most 4. The
// per-tile unique-query count is the structural witness that each
// worker handles a small set of queries, so an embedding load
// counter incremented at the start of each tile would bound at
// O(N / Q_tile) loads per thread (same as the cost_aware bound is
// O(N) when each thread takes one pair at a time).
std::size_t max_unique_queries_per_tile(const std::vector<hiko_ad::PairTile>& tiles,
                                        std::size_t item_count,
                                        bool include_self) {
  std::size_t maximum = 0U;
  for (const hiko_ad::PairTile& tile : tiles) {
    std::set<std::size_t> queries;
    hiko_ad::iterate_pair_tile(
        tile, item_count, include_self,
        [&](std::size_t /*pair_index*/, std::size_t q,
            std::size_t /*t*/) noexcept { queries.insert(q); });
    if (queries.size() > maximum) {
      maximum = queries.size();
    }
  }
  return maximum;
}

void test_tile_locality_bounds_unique_queries_per_tile() {
  constexpr std::size_t kItemCount = 50U;
  constexpr bool kIncludeSelf = false;
  for (std::size_t q_tile : {1U, 2U, 4U, 8U}) {
    std::vector<hiko_ad::PairTile> tiles;
    hiko_ad::partition_pair_tiles(kItemCount, kIncludeSelf, q_tile,
                               /*target_tile=*/0U, tiles);
    const std::size_t maximum =
        max_unique_queries_per_tile(tiles, kItemCount, kIncludeSelf);
    if (maximum > q_tile) {
      fail("tile contains more queries than the configured Q_tile");
    }
    if (maximum == 0U) {
      fail("tile dispatch produced no observed queries");
    }
  }
}

}  // namespace

int main() {
  test_ij_to_pair_index_round_trip(false);
  test_ij_to_pair_index_round_trip(true);
  for (bool include_self : {false, true}) {
    for (std::size_t q_tile : {1U, 2U, 4U, 8U}) {
      for (std::size_t t_tile : {0U, 4U, 16U}) {
        test_partition_covers_each_pair_exactly_once(include_self, q_tile,
                                                     t_tile);
      }
    }
    test_query_tile_size_one_keeps_one_query_per_tile(include_self);
    test_streaming_safe_tile_keeps_pair_id_ascending(include_self);
  }
  test_query_tiled_plan_picks_sane_tile_size_for_zero_residue();
  test_query_tiled_plan_clamps_q_tile_for_long_embeddings();
  test_query_tiled_plan_grows_q_tile_for_short_embeddings();
  test_parallel_dispatch_matches_serial_baseline(false);
  test_parallel_dispatch_matches_serial_baseline(true);
  test_dispatch_in_query_tiled_mode_does_not_deadlock_at_rotation_scale();
  test_tile_locality_bounds_unique_queries_per_tile();
  return 0;
}

#ifndef HIKOBOSHI_API_ALL_VS_ALL_HPP
#define HIKOBOSHI_API_ALL_VS_ALL_HPP

/// @file
/// Streaming interfaces for symmetric all-vs-all alignment.

#include <hikoboshi/api/requests.hpp>
#include <hikoboshi/api/results.hpp>
#include <hikoboshi/universal/status.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace hikoboshi::api {

/// Streaming receiver for symmetric all-vs-all pair records.
///
/// Records are emitted in stable lexicographic input-index order. Returning a
/// non-ok status stops enumeration and propagates the status to the caller.
class PairwiseResultSink {
 public:
  virtual ~PairwiseResultSink() = default;
  [[nodiscard]] virtual universal::Status receive(
      const PairwiseResultRecord& record) = 0;
};

/// Streaming sink that writes one TSV row per all-vs-all pair record.
///
/// Output schema and column order match the legacy
/// `render_all_vs_all_summary` writer used by the CLI: a single header row,
/// then one TSV row per `receive`, in stable lexicographic input-index order.
/// Optional `pair_id_lookup` and per-pair artifact path lookups are queried by
/// `record.query_index`/`record.target_index` so the caller never needs to
/// pre-materialize an `AllVsAllResult` to populate the row.
///
/// The sink owns no per-pair allocations: the only growable storage is the
/// std::ostream object the caller provides. End-to-end memory is bounded by
/// the algorithms-layer streaming sink ring (p44) plus per-worker scratch.
class TsvStreamingAllVsAllSink final : public PairwiseResultSink {
 public:
  /// Optional callbacks that turn `(query_index, target_index)` into pair-id
  /// and per-pair artifact path strings. Each callback may be empty (default-
  /// constructed) and the corresponding column is then emitted blank.
  struct Callbacks {
    /// Render the `pair_id` column for `(query_index, target_index)`.
    std::string (*pair_id)(std::size_t, std::size_t, void*) = nullptr;
    /// Render the `fasta_path` column for `(query_index, target_index)`.
    std::string (*fasta_path)(std::size_t, std::size_t, void*) = nullptr;
    /// Render the `pdb_path` column for `(query_index, target_index)`.
    std::string (*pdb_path)(std::size_t, std::size_t, void*) = nullptr;
    /// Opaque user-data forwarded as the third argument of every callback.
    void* user_data = nullptr;
    /// Emit soft score columns (`soft_sw_score` and `sw_per_*`).
    bool include_dual_score_schema = false;
  };

  /// Construct the sink. The header row is written to every supplied stream
  /// up front so callers do not have to coordinate header emission. The sink
  /// holds raw pointers to each `std::ostream` for its lifetime.
  TsvStreamingAllVsAllSink(std::vector<std::ostream*> outputs,
                           Callbacks callbacks);

  /// Convenience constructor for a single output stream.
  TsvStreamingAllVsAllSink(std::ostream& output, Callbacks callbacks);

  TsvStreamingAllVsAllSink(const TsvStreamingAllVsAllSink&) = delete;
  TsvStreamingAllVsAllSink& operator=(const TsvStreamingAllVsAllSink&) = delete;
  TsvStreamingAllVsAllSink(TsvStreamingAllVsAllSink&&) = delete;
  TsvStreamingAllVsAllSink& operator=(TsvStreamingAllVsAllSink&&) = delete;

  ~TsvStreamingAllVsAllSink() override = default;

  [[nodiscard]] universal::Status receive(
      const PairwiseResultRecord& record) override;

  /// Write the canonical TSV header for the all-vs-all summary schema.
  ///
  /// Exposed so unit tests and parity diffs can assert the header layout
  /// without instantiating the sink.
  static void write_header(std::ostream& out);

 private:
  std::vector<std::ostream*> outputs_;
  Callbacks callbacks_{};
  std::size_t emitted_ = 0;
};

}  // namespace hikoboshi::api

#endif  // HIKOBOSHI_API_ALL_VS_ALL_HPP

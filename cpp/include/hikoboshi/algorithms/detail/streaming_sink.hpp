#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_STREAMING_SINK_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_STREAMING_SINK_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/algorithms/detail/all_vs_all_workspace.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::algorithms::detail {

inline constexpr std::size_t kDefaultStreamingSinkSlots = 1U << 16;
inline constexpr std::size_t kMaxStreamingSinkSlots = 1U << 20;

/// Default per-worker ring capacity for `StreamingSequencerBuffer`.
inline constexpr std::size_t kDefaultStreamingSequencerPerWorkerCapacity = 64U;

/// Default cap on streaming-sink in-flight bytes (16 GiB). Workers block their
/// own enqueue path when total in-flight bytes would exceed this cap, even if
/// their own per-worker ring still has room. This is a safety valve for
/// pathological variance, not the primary backpressure path.
inline constexpr std::size_t kDefaultStreamingSequencerMaxInFlightBytes =
    static_cast<std::size_t>(16ULL) * 1024ULL * 1024ULL * 1024ULL;

/// Capacity-preserving swap of two PairwiseResultRecord instances.
///
/// `result.path.steps` is swapped via vector::swap so reserved capacity stays
/// with both sides; scalar fields are swapped by value. After this call, no
/// allocations have happened on either side and both records keep whatever
/// reserved capacity they came in with.
void swap_pairwise_result_record(PairwiseResultRecord& a,
                                 PairwiseResultRecord& b) noexcept;

/// Bounded ordered ring buffer for all-vs-all pair records (legacy design).
///
/// Workers `submit(pair_id, ...)` records into the slot pair_id mod K, where K
/// is a power-of-two slot count. The single drain thread forwards records to
/// the downstream `PairwiseResultSink` in pair_id order. Steady-state memory
/// is O(K) records, independent of total pair count. Workers block when their
/// slot still holds an unflushed predecessor; the drainer blocks when the
/// next-to-flush slot is not yet ready. Ordering is preserved by construction.
///
/// This buffer head-of-line-blocks: if the head pair_id is slow, the ring
/// fills with completed but unemitted records and downstream workers stall on
/// backpressure. The replacement `StreamingSequencerBuffer` decouples worker
/// progress from drain progress. The legacy class is kept temporarily so the
/// `streaming_sink_byte_identity_test` can drive both designs from the same
/// process and prove output equivalence; remove after `ssf2-streaming-sink-
/// validation` confirms the new design at 5K/8K/15K.
///
/// `submit` and `drain` are called concurrently. Multiple workers may submit
/// simultaneously; exactly one thread may call `drain`.
///
/// Memory ordering uses the Vyukov-style sequence-counter protocol: each slot
/// carries an atomic `sequence` whose value identifies which generation of
/// pair_id the slot is currently in. Workers wait until `sequence == pair_id`
/// (slot ready for new write of pair_id), then publish `pair_id + 1` after
/// writing. The drainer waits until `sequence == pair_id + 1` (slot ready for
/// read), then publishes `pair_id + slot_count` after reading (slot ready for
/// the next-generation write).
class StreamingRecordBuffer {
 public:
  /// Construct a ring buffer with `slot_count_hint` slots (rounded up to a
  /// power of two and clamped to `[1, kMaxStreamingSinkSlots]`). Reserve
  /// `max_result_step_count` capacity on every slot's `path.steps` so that
  /// per-pair `submit` does not allocate in steady state.
  StreamingRecordBuffer(std::size_t slot_count_hint,
                        std::size_t max_result_step_count,
                        PairwiseResultSink& downstream);

  ~StreamingRecordBuffer() = default;

  StreamingRecordBuffer(const StreamingRecordBuffer&) = delete;
  StreamingRecordBuffer& operator=(const StreamingRecordBuffer&) = delete;
  StreamingRecordBuffer(StreamingRecordBuffer&&) = delete;
  StreamingRecordBuffer& operator=(StreamingRecordBuffer&&) = delete;

  [[nodiscard]] std::size_t slot_count() const noexcept { return slot_count_; }

  /// Worker-side blocking submit: blocks until slot `pair_id mod K` is free
  /// (or until `request_abort()` is observed), then swaps `worker_record`
  /// into the slot without allocating. After return, `worker_record` is left
  /// in the slot's previous state with reserved capacity preserved. On
  /// observed abort the slot is left untouched and `worker_record` is
  /// unchanged.
  void submit(std::size_t pair_id,
              PairwiseResultRecord& worker_record);

  /// Worker-side non-blocking try-submit. Returns true and swaps the record
  /// into the slot if the slot is free for `pair_id`; otherwise returns
  /// false and leaves the worker record untouched. Used by the dual-role
  /// driver thread to avoid deadlocking on its own backpressure.
  [[nodiscard]] bool try_submit(std::size_t pair_id,
                                PairwiseResultRecord& worker_record);

  /// Sink-side: drain `total_pair_count` records in pair_id order, forwarding
  /// each to the downstream sink. Returns the first non-Ok status seen, or Ok
  /// if all records were flushed. On non-Ok, requests abort so workers waiting
  /// on backpressure observe the failure and exit promptly. On observed abort
  /// before all records are drained, returns Ok and leaves error propagation
  /// to the worker side.
  [[nodiscard]] hikoboshi::universal::Status drain(std::size_t total_pair_count);

  /// Single non-blocking drain step. If slot for `pair_id` is ready, forwards
  /// to the downstream sink and sets `drained=true`. Returns ok_status() if
  /// either drained successfully or slot was not ready (or aborted). Returns
  /// the downstream sink's non-Ok status (and requests abort) if the sink
  /// rejected the record.
  [[nodiscard]] hikoboshi::universal::Status try_drain_step(
      std::size_t pair_id,
      bool& drained);

  /// Single blocking drain step. Waits until slot for `pair_id` is ready or
  /// abort is observed, then forwards to the downstream sink. Returns the
  /// sink's non-Ok status (and requests abort) on rejection, ok_status() on
  /// success or abort.
  [[nodiscard]] hikoboshi::universal::Status drain_step_blocking(
      std::size_t pair_id);

  /// Wake all waiters and signal that no further submit/drain progress is
  /// possible. Idempotent.
  void request_abort() noexcept;

  [[nodiscard]] bool aborted() const noexcept {
    return abort_.load(std::memory_order_acquire);
  }

  /// Resolve a slot count for a given pair_count: the smallest power of two
  /// that is at least `min(pair_count, kDefaultStreamingSinkSlots)`. Returns 1
  /// for pair_count == 0.
  [[nodiscard]] static std::size_t resolve_slot_count(
      std::size_t pair_count) noexcept;

 private:
  struct Slot {
    PairwiseResultRecord record;
    std::atomic<std::uint64_t> sequence{0};
  };

  std::unique_ptr<Slot[]> slots_;
  PairwiseResultSink& downstream_;
  std::size_t slot_count_ = 0;
  std::size_t mask_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> abort_{false};
};

/// Per-worker-buffer + single-sequencer streaming sink.
///
/// Design summary:
///
/// - Each worker writes completed `(pair_id, record)` entries into its own
///   bounded SPSC ring with `per_worker_capacity` slots. Each slot holds a
///   `PairwiseResultRecord` whose `path.steps` is reserved to
///   `max_result_step_count` at construction; workers swap their populated
///   record into the slot, picking up the slot's previously-cleared record
///   buffer (capacity preserved). Steady-state per-pair `submit` does not
///   allocate.
/// - A dedicated sequencer thread iterates worker rings and emits records
///   downstream in canonical pair_id order. Within one worker the dispatcher
///   already issues pair_ids in ascending order (a courtesy preserved from
///   the legacy ring path), so each worker ring is naturally sorted; the
///   sequencer only needs to find the worker ring whose head holds the next
///   expected pair_id. When no ring head matches, the sequencer waits on
///   `enqueue_generation_` until any worker submits.
/// - Worker progress is decoupled from drain progress. A slow head pair on
///   one worker no longer blocks any other worker's submit: each worker
///   only blocks when its own per-worker ring is full (and the sequencer
///   has not yet advanced past that worker's head pair_id).
/// - A global cap on in-flight bytes
///   (`HIKOBOSHI_STREAMING_SINK_MAX_INFLIGHT_BYTES`, default 16 GiB) acts as a
///   safety valve for pathological variance. Workers that would push the
///   global in-flight byte count over the cap block on a backpressure
///   condition variable until the sequencer drains.
///
/// Output ordering matches the legacy ring buffer: records are emitted to the
/// downstream sink strictly in ascending `pair_id`. The legacy class is kept
/// alongside this one so the byte-identity test can compare both designs
/// directly; remove the legacy class after `ssf2-streaming-sink-validation`.
class StreamingSequencerBuffer {
 public:
  struct Config {
    /// Number of worker rings to allocate. Must be >= 1.
    std::size_t worker_count = 1;
    /// Total number of pair_ids the sequencer expects to emit.
    std::size_t total_pair_count = 0;
    /// Per-worker ring capacity (rounded up to a power of two). Defaults to
    /// `kDefaultStreamingSequencerPerWorkerCapacity`.
    std::size_t per_worker_capacity = kDefaultStreamingSequencerPerWorkerCapacity;
    /// Reserved `path.steps` capacity per slot. Set to the per-pair max
    /// step count from the workspace plan to avoid per-pair allocation.
    std::size_t max_result_step_count = 0;
    /// Global in-flight byte cap. Set to 0 to disable the global cap (per-
    /// worker rings still bound memory).
    std::size_t max_in_flight_bytes = kDefaultStreamingSequencerMaxInFlightBytes;
  };

  StreamingSequencerBuffer(const Config& config,
                           PairwiseResultSink& downstream);

  ~StreamingSequencerBuffer();

  StreamingSequencerBuffer(const StreamingSequencerBuffer&) = delete;
  StreamingSequencerBuffer& operator=(const StreamingSequencerBuffer&) = delete;
  StreamingSequencerBuffer(StreamingSequencerBuffer&&) = delete;
  StreamingSequencerBuffer& operator=(StreamingSequencerBuffer&&) = delete;

  [[nodiscard]] std::size_t worker_count() const noexcept {
    return worker_count_;
  }

  [[nodiscard]] std::size_t per_worker_capacity() const noexcept {
    return per_worker_cap_;
  }

  [[nodiscard]] std::size_t max_in_flight_bytes() const noexcept {
    return max_in_flight_bytes_;
  }

  /// Worker-side submit. Blocks only if (a) this worker's ring is full
  /// (the sequencer has not yet drained the slot the worker is about to
  /// reuse), or (b) the global in-flight byte cap would be exceeded. After
  /// return, `worker_record` holds the slot's previously-cleared content
  /// (with reserved capacity intact). On observed abort, the worker record
  /// is left unchanged.
  void submit(std::size_t worker_id,
              std::size_t pair_id,
              PairwiseResultRecord& worker_record);

  /// Wake all waiters and signal that no further submit/drain progress is
  /// possible. Idempotent. After this call, the sequencer thread will exit
  /// without emitting any further records and `wait_drain_complete` will
  /// return without waiting for `total_pair_count` records.
  void request_abort() noexcept;

  [[nodiscard]] bool aborted() const noexcept {
    return abort_.load(std::memory_order_acquire);
  }

  /// Block until the sequencer has emitted all `total_pair_count` records or
  /// abort has been observed. Joins the sequencer thread. Returns the first
  /// non-Ok downstream status seen by the sequencer, or ok_status() on
  /// successful drain. Subsequent calls are no-ops returning the recorded
  /// status. The caller must call this exactly once after all workers have
  /// finished submitting; the destructor will do so as a fallback (with
  /// abort) if it has not been called.
  [[nodiscard]] hikoboshi::universal::Status wait_drain_complete();

  /// Number of records emitted downstream so far. Useful for tests.
  [[nodiscard]] std::size_t emitted_count() const noexcept {
    return emitted_count_.load(std::memory_order_acquire);
  }

  /// Resolve `max_in_flight_bytes` from the
  /// `HIKOBOSHI_STREAMING_SINK_MAX_INFLIGHT_BYTES` env var; returns the
  /// default if the env var is unset, empty, or not a positive integer.
  [[nodiscard]] static std::size_t resolve_max_in_flight_bytes_from_env() noexcept;

  /// Read the `HIKOBOSHI_STREAMING_SINK_LEGACY` env var. Returns true if the
  /// env var is set to a non-zero value (e.g. "1", "true").
  [[nodiscard]] static bool legacy_sink_enabled_from_env() noexcept;

 private:
  struct Slot {
    std::size_t pair_id = 0;
    PairwiseResultRecord record;
  };

  struct WorkerRing {
    std::vector<Slot> slots;
    std::size_t mask = 0;
    alignas(64) std::atomic<std::uint64_t> head{0};
    alignas(64) std::atomic<std::uint64_t> tail{0};
    std::mutex producer_mtx;
    std::condition_variable producer_cv;
  };

  void sequencer_loop();
  void notify_all_producers() noexcept;

  PairwiseResultSink& downstream_;
  std::size_t worker_count_ = 0;
  std::size_t total_pair_count_ = 0;
  std::size_t per_worker_cap_ = 0;
  std::size_t max_result_step_count_ = 0;
  std::size_t max_in_flight_bytes_ = 0;
  std::size_t per_slot_byte_estimate_ = 0;

  std::vector<std::unique_ptr<WorkerRing>> rings_;

  std::mutex sequencer_mtx_;
  std::condition_variable sequencer_cv_;
  std::atomic<std::uint64_t> enqueue_generation_{0};

  std::mutex backpressure_mtx_;
  std::condition_variable backpressure_cv_;
  std::atomic<std::size_t> in_flight_bytes_{0};

  std::atomic<bool> abort_{false};
  std::atomic<bool> done_signal_{false};
  std::atomic<bool> joined_{false};
  std::atomic<std::size_t> emitted_count_{0};

  std::mutex status_mtx_;
  hikoboshi::universal::Status sequencer_status_{};

  std::thread sequencer_thread_;
};

}  // namespace hikoboshi::algorithms::detail

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_STREAMING_SINK_HPP

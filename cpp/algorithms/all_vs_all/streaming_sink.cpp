#include <hikoboshi/algorithms/detail/streaming_sink.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::algorithms::detail {
namespace {

std::size_t round_up_pow2_clamped(std::size_t value,
                                  std::size_t maximum) noexcept {
  if (value <= 1U) {
    return 1U;
  }
  if (value >= maximum) {
    return maximum;
  }
  std::size_t k = 1U;
  while (k < value) {
    k <<= 1U;
  }
  return k;
}

std::size_t round_up_pow2(std::size_t value) noexcept {
  if (value <= 1U) {
    return 1U;
  }
  std::size_t k = 1U;
  while (k < value) {
    k <<= 1U;
  }
  return k;
}

bool parse_size_t_env(const char* name, std::size_t& out) noexcept {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  // Skip leading whitespace.
  while (*raw == ' ' || *raw == '\t') {
    ++raw;
  }
  if (*raw == '\0') {
    return false;
  }
  std::size_t value = 0;
  bool any_digit = false;
  while (*raw >= '0' && *raw <= '9') {
    const std::size_t digit = static_cast<std::size_t>(*raw - '0');
    if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
      return false;
    }
    value = value * 10U + digit;
    any_digit = true;
    ++raw;
  }
  if (!any_digit) {
    return false;
  }
  out = value;
  return true;
}

}  // namespace

void swap_pairwise_result_record(PairwiseResultRecord& a,
                                 PairwiseResultRecord& b) noexcept {
  using std::swap;
  swap(a.query_index, b.query_index);
  swap(a.target_index, b.target_index);
  a.result.path.steps.swap(b.result.path.steps);
  swap(a.result.path.aligned_pairs, b.result.path.aligned_pairs);
  swap(a.result.path.query_start, b.result.path.query_start);
  swap(a.result.path.query_end, b.result.path.query_end);
  swap(a.result.path.target_start, b.result.path.target_start);
  swap(a.result.path.target_end, b.result.path.target_end);
  swap(a.result.raw_sw_score, b.result.raw_sw_score);
  swap(a.result.metrics, b.result.metrics);
}

// =====================================================================
// Legacy ring-buffer streaming sink (kept until ssf2 validates the
// per-worker-buffers + sequencer replacement).
// =====================================================================

std::size_t StreamingRecordBuffer::resolve_slot_count(
    std::size_t pair_count) noexcept {
  if (pair_count == 0U) {
    return 1U;
  }
  const std::size_t target =
      std::min<std::size_t>(pair_count, kDefaultStreamingSinkSlots);
  return round_up_pow2_clamped(target, kMaxStreamingSinkSlots);
}

StreamingRecordBuffer::StreamingRecordBuffer(
    std::size_t slot_count_hint,
    std::size_t max_result_step_count,
    PairwiseResultSink& downstream)
    : slots_(std::make_unique<Slot[]>(round_up_pow2_clamped(
          std::max<std::size_t>(slot_count_hint, 1U),
          kMaxStreamingSinkSlots))),
      downstream_(downstream),
      slot_count_(round_up_pow2_clamped(
          std::max<std::size_t>(slot_count_hint, 1U),
          kMaxStreamingSinkSlots)),
      mask_(slot_count_ - 1U) {
  for (std::size_t index = 0; index < slot_count_; ++index) {
    if (max_result_step_count != 0U) {
      slots_[index].record.result.path.steps.reserve(max_result_step_count);
    }
    slots_[index].sequence.store(static_cast<std::uint64_t>(index),
                                 std::memory_order_relaxed);
  }
}

void StreamingRecordBuffer::submit(std::size_t pair_id,
                                   PairwiseResultRecord& worker_record) {
  const std::size_t slot_index = pair_id & mask_;
  const std::uint64_t expected_sequence = static_cast<std::uint64_t>(pair_id);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, slot_index, expected_sequence]() {
      if (abort_.load(std::memory_order_acquire)) {
        return true;
      }
      return slots_[slot_index].sequence.load(std::memory_order_acquire) ==
             expected_sequence;
    });
    if (abort_.load(std::memory_order_acquire)) {
      return;
    }
    swap_pairwise_result_record(slots_[slot_index].record, worker_record);
    slots_[slot_index].sequence.store(expected_sequence + 1U,
                                      std::memory_order_release);
  }
  cv_.notify_all();
}

bool StreamingRecordBuffer::try_submit(std::size_t pair_id,
                                       PairwiseResultRecord& worker_record) {
  const std::size_t slot_index = pair_id & mask_;
  const std::uint64_t expected_sequence = static_cast<std::uint64_t>(pair_id);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (abort_.load(std::memory_order_acquire)) {
      return false;
    }
    if (slots_[slot_index].sequence.load(std::memory_order_acquire) !=
        expected_sequence) {
      return false;
    }
    swap_pairwise_result_record(slots_[slot_index].record, worker_record);
    slots_[slot_index].sequence.store(expected_sequence + 1U,
                                      std::memory_order_release);
  }
  cv_.notify_all();
  return true;
}

hikoboshi::universal::Status StreamingRecordBuffer::try_drain_step(
    std::size_t pair_id,
    bool& drained) {
  drained = false;
  const std::size_t slot_index = pair_id & mask_;
  const std::uint64_t ready_sequence =
      static_cast<std::uint64_t>(pair_id) + 1U;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (abort_.load(std::memory_order_acquire)) {
      return hikoboshi::universal::ok_status();
    }
    if (slots_[slot_index].sequence.load(std::memory_order_acquire) !=
        ready_sequence) {
      return hikoboshi::universal::ok_status();
    }
  }
  const hikoboshi::universal::Status status =
      downstream_.receive(slots_[slot_index].record);
  if (!hikoboshi::universal::is_ok(status)) {
    request_abort();
    return status;
  }
  slots_[slot_index].record.result.path.steps.clear();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    slots_[slot_index].sequence.store(
        static_cast<std::uint64_t>(pair_id) +
            static_cast<std::uint64_t>(slot_count_),
        std::memory_order_release);
  }
  cv_.notify_all();
  drained = true;
  return hikoboshi::universal::ok_status();
}

hikoboshi::universal::Status StreamingRecordBuffer::drain_step_blocking(
    std::size_t pair_id) {
  const std::size_t slot_index = pair_id & mask_;
  const std::uint64_t ready_sequence =
      static_cast<std::uint64_t>(pair_id) + 1U;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, slot_index, ready_sequence]() {
      if (abort_.load(std::memory_order_acquire)) {
        return true;
      }
      return slots_[slot_index].sequence.load(std::memory_order_acquire) ==
             ready_sequence;
    });
    if (abort_.load(std::memory_order_acquire)) {
      return hikoboshi::universal::ok_status();
    }
  }
  const hikoboshi::universal::Status status =
      downstream_.receive(slots_[slot_index].record);
  if (!hikoboshi::universal::is_ok(status)) {
    request_abort();
    return status;
  }
  slots_[slot_index].record.result.path.steps.clear();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    slots_[slot_index].sequence.store(
        static_cast<std::uint64_t>(pair_id) +
            static_cast<std::uint64_t>(slot_count_),
        std::memory_order_release);
  }
  cv_.notify_all();
  return hikoboshi::universal::ok_status();
}

hikoboshi::universal::Status StreamingRecordBuffer::drain(
    std::size_t total_pair_count) {
  for (std::size_t pair_id = 0; pair_id < total_pair_count; ++pair_id) {
    if (abort_.load(std::memory_order_acquire)) {
      return hikoboshi::universal::ok_status();
    }
    const hikoboshi::universal::Status status = drain_step_blocking(pair_id);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
  }
  return hikoboshi::universal::ok_status();
}

void StreamingRecordBuffer::request_abort() noexcept {
  bool expected = false;
  if (!abort_.compare_exchange_strong(expected, true,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // No-op state change; mutex acquisition serialises with waiters that have
    // just evaluated the predicate.
  }
  cv_.notify_all();
}

// =====================================================================
// Per-worker-buffers + single-sequencer streaming sink (ssf1).
// =====================================================================

std::size_t StreamingSequencerBuffer::resolve_max_in_flight_bytes_from_env()
    noexcept {
  std::size_t value = 0;
  if (parse_size_t_env("HIKOBOSHI_STREAMING_SINK_MAX_INFLIGHT_BYTES", value) &&
      value > 0U) {
    return value;
  }
  return kDefaultStreamingSequencerMaxInFlightBytes;
}

bool StreamingSequencerBuffer::legacy_sink_enabled_from_env() noexcept {
  const char* raw = std::getenv("HIKOBOSHI_STREAMING_SINK_LEGACY");
  if (raw == nullptr) {
    return false;
  }
  while (*raw == ' ' || *raw == '\t') {
    ++raw;
  }
  if (*raw == '\0') {
    return false;
  }
  if (raw[0] == '0' && raw[1] == '\0') {
    return false;
  }
  return true;
}

StreamingSequencerBuffer::StreamingSequencerBuffer(const Config& config,
                                                   PairwiseResultSink& downstream)
    : downstream_(downstream),
      worker_count_(config.worker_count == 0U ? 1U : config.worker_count),
      total_pair_count_(config.total_pair_count),
      per_worker_cap_(round_up_pow2(
          config.per_worker_capacity == 0U
              ? kDefaultStreamingSequencerPerWorkerCapacity
              : config.per_worker_capacity)),
      max_result_step_count_(config.max_result_step_count),
      max_in_flight_bytes_(config.max_in_flight_bytes),
      per_slot_byte_estimate_(
          sizeof(Slot) +
          config.max_result_step_count *
              sizeof(hikoboshi::universal::AlignmentStep)) {
  rings_.reserve(worker_count_);
  for (std::size_t w = 0; w < worker_count_; ++w) {
    auto ring = std::make_unique<WorkerRing>();
    ring->slots.resize(per_worker_cap_);
    ring->mask = per_worker_cap_ - 1U;
    if (max_result_step_count_ != 0U) {
      for (auto& slot : ring->slots) {
        slot.record.result.path.steps.reserve(max_result_step_count_);
      }
    }
    rings_.push_back(std::move(ring));
  }
  sequencer_thread_ = std::thread(&StreamingSequencerBuffer::sequencer_loop,
                                  this);
}

StreamingSequencerBuffer::~StreamingSequencerBuffer() {
  if (!joined_.load(std::memory_order_acquire)) {
    request_abort();
    if (sequencer_thread_.joinable()) {
      sequencer_thread_.join();
    }
    joined_.store(true, std::memory_order_release);
  }
}

void StreamingSequencerBuffer::submit(std::size_t worker_id,
                                      std::size_t pair_id,
                                      PairwiseResultRecord& worker_record) {
  if (abort_.load(std::memory_order_acquire)) {
    return;
  }

  WorkerRing& ring = *rings_[worker_id];

  // Wait for a free slot in this worker's own ring.
  {
    std::unique_lock<std::mutex> lock(ring.producer_mtx);
    ring.producer_cv.wait(lock, [&]() {
      if (abort_.load(std::memory_order_acquire)) {
        return true;
      }
      const std::uint64_t cur_tail = ring.tail.load(std::memory_order_relaxed);
      const std::uint64_t cur_head = ring.head.load(std::memory_order_acquire);
      return (cur_tail - cur_head) <
             static_cast<std::uint64_t>(per_worker_cap_);
    });
    if (abort_.load(std::memory_order_acquire)) {
      return;
    }
  }

  // Global memory backpressure (safety valve for pathological variance).
  if (max_in_flight_bytes_ != 0U) {
    std::unique_lock<std::mutex> lock(backpressure_mtx_);
    backpressure_cv_.wait(lock, [&]() {
      if (abort_.load(std::memory_order_acquire)) {
        return true;
      }
      const std::size_t in_flight =
          in_flight_bytes_.load(std::memory_order_acquire);
      return in_flight + per_slot_byte_estimate_ <= max_in_flight_bytes_;
    });
    if (abort_.load(std::memory_order_acquire)) {
      return;
    }
    in_flight_bytes_.fetch_add(per_slot_byte_estimate_,
                               std::memory_order_acq_rel);
  }

  const std::uint64_t cur_tail = ring.tail.load(std::memory_order_relaxed);
  Slot& slot = ring.slots[cur_tail & ring.mask];
  swap_pairwise_result_record(worker_record, slot.record);
  slot.pair_id = pair_id;
  ring.tail.store(cur_tail + 1U, std::memory_order_release);

  // The wake-up state change is the atomic fetch_add itself with release
  // ordering; the sequencer's wait predicate loads enqueue_generation_ with
  // acquire so the fetch_add happens-before the predicate evaluation. We
  // therefore do not need to acquire `sequencer_mtx_` here purely for
  // memory ordering — the atomic carries the synchronization.
  enqueue_generation_.fetch_add(1U, std::memory_order_acq_rel);
  sequencer_cv_.notify_one();
}

void StreamingSequencerBuffer::request_abort() noexcept {
  bool expected = false;
  if (!abort_.compare_exchange_strong(expected, true,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(sequencer_mtx_);
  }
  sequencer_cv_.notify_all();
  {
    std::lock_guard<std::mutex> lock(backpressure_mtx_);
  }
  backpressure_cv_.notify_all();
  notify_all_producers();
}

void StreamingSequencerBuffer::notify_all_producers() noexcept {
  for (auto& ring_ptr : rings_) {
    {
      std::lock_guard<std::mutex> lock(ring_ptr->producer_mtx);
    }
    ring_ptr->producer_cv.notify_all();
  }
}

hikoboshi::universal::Status StreamingSequencerBuffer::wait_drain_complete() {
  if (joined_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(status_mtx_);
    return sequencer_status_;
  }
  done_signal_.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(sequencer_mtx_);
  }
  sequencer_cv_.notify_all();
  if (sequencer_thread_.joinable()) {
    sequencer_thread_.join();
  }
  joined_.store(true, std::memory_order_release);
  std::lock_guard<std::mutex> lock(status_mtx_);
  return sequencer_status_;
}

void StreamingSequencerBuffer::sequencer_loop() {
  std::size_t next_expected = 0;

  while (next_expected < total_pair_count_) {
    if (abort_.load(std::memory_order_acquire)) {
      // On abort, release any backpressure waiters and return without
      // emitting further records.
      in_flight_bytes_.store(0U, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(backpressure_mtx_);
      }
      backpressure_cv_.notify_all();
      notify_all_producers();
      return;
    }

    const std::uint64_t last_gen =
        enqueue_generation_.load(std::memory_order_acquire);

    bool made_progress = false;

    while (next_expected < total_pair_count_) {
      if (abort_.load(std::memory_order_acquire)) {
        break;
      }

      // Find the worker ring whose head pair_id is smallest. Per-worker
      // dispatch order is ascending, so each ring is naturally sorted; the
      // head of any non-empty ring is the smallest unread pair_id from that
      // worker.
      std::size_t found_w = std::numeric_limits<std::size_t>::max();
      std::size_t min_pair_id = std::numeric_limits<std::size_t>::max();

      for (std::size_t w = 0; w < worker_count_; ++w) {
        WorkerRing& ring = *rings_[w];
        const std::uint64_t cur_tail =
            ring.tail.load(std::memory_order_acquire);
        const std::uint64_t cur_head =
            ring.head.load(std::memory_order_relaxed);
        if (cur_head < cur_tail) {
          const Slot& slot = ring.slots[cur_head & ring.mask];
          const std::size_t pid = slot.pair_id;
          if (pid < min_pair_id) {
            min_pair_id = pid;
            found_w = w;
          }
        }
      }

      if (found_w == std::numeric_limits<std::size_t>::max()) {
        break;  // No work currently visible.
      }
      if (min_pair_id != next_expected) {
        // Some worker hasn't yet submitted next_expected; wait on the
        // generation counter for any worker to enqueue.
        break;
      }

      // Emit ring[found_w].head.
      WorkerRing& ring = *rings_[found_w];
      const std::uint64_t cur_head =
          ring.head.load(std::memory_order_relaxed);
      Slot& slot = ring.slots[cur_head & ring.mask];

      const hikoboshi::universal::Status status = downstream_.receive(slot.record);
      // Clear path.steps to preserve reserved capacity for the next worker
      // write into this slot.
      slot.record.result.path.steps.clear();
      ring.head.store(cur_head + 1U, std::memory_order_release);

      ++next_expected;
      emitted_count_.fetch_add(1U, std::memory_order_acq_rel);
      made_progress = true;

      if (max_in_flight_bytes_ != 0U) {
        in_flight_bytes_.fetch_sub(per_slot_byte_estimate_,
                                   std::memory_order_acq_rel);
        {
          std::lock_guard<std::mutex> bp_lock(backpressure_mtx_);
        }
        backpressure_cv_.notify_all();
      }

      // Notify the worker whose ring slot was just freed.
      {
        std::lock_guard<std::mutex> lock(ring.producer_mtx);
      }
      ring.producer_cv.notify_one();

      if (!hikoboshi::universal::is_ok(status)) {
        {
          std::lock_guard<std::mutex> lock(status_mtx_);
          sequencer_status_ = status;
        }
        request_abort();
        return;
      }
    }

    if (made_progress) {
      continue;
    }

    if (next_expected >= total_pair_count_) {
      break;
    }

    // If all workers have signalled they are finished submitting and no
    // ring still holds an unread record, no further work can arrive. This
    // is an unexpected state (the dispatch loop should have submitted every
    // pair) and we exit with an internal error rather than hanging.
    if (done_signal_.load(std::memory_order_acquire)) {
      bool any_ring_nonempty = false;
      for (std::size_t w = 0; w < worker_count_; ++w) {
        WorkerRing& ring = *rings_[w];
        const std::uint64_t cur_tail =
            ring.tail.load(std::memory_order_acquire);
        const std::uint64_t cur_head =
            ring.head.load(std::memory_order_acquire);
        if (cur_head < cur_tail) {
          any_ring_nonempty = true;
          break;
        }
      }
      if (!any_ring_nonempty) {
        if (!abort_.load(std::memory_order_acquire)) {
          std::lock_guard<std::mutex> lock(status_mtx_);
          if (hikoboshi::universal::is_ok(sequencer_status_)) {
            sequencer_status_ = hikoboshi::universal::internal_error_status(
                "streaming sequencer drained before all pairs were submitted");
          }
        }
        return;
      }
    }

    // No progress; wait for new submissions or termination signal.
    std::unique_lock<std::mutex> lock(sequencer_mtx_);
    sequencer_cv_.wait(lock, [&]() {
      if (abort_.load(std::memory_order_acquire)) {
        return true;
      }
      if (enqueue_generation_.load(std::memory_order_acquire) > last_gen) {
        return true;
      }
      if (done_signal_.load(std::memory_order_acquire)) {
        // Even on done_signal, the sequencer should still drain anything
        // that arrived. Re-check the predicate so the outer loop re-scans.
        return true;
      }
      return false;
    });
  }
}

}  // namespace hikoboshi::algorithms::detail

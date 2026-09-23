#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_ENCODING_SCHEDULER_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_ENCODING_SCHEDULER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace hikoboshi::algorithms::detail {

struct EncodingRange {
  std::size_t begin;
  std::size_t end;
};

// Keep input order and choose boundaries nearest equal cumulative work.
// Nonempty ranges cover every input exactly once; each worker can prepare for
// its range's maximum length before encoding. Long double avoids integer cost
// overflow; zero-cost items still receive a unit of scheduling weight.
// The cost callback is pure and nonnegative. Planning is O(items + workers)
// time and O(workers) storage, with no per-protein scheduling allocation.
template <typename Cost>
std::vector<EncodingRange> partition_encoding_ranges(
    std::size_t item_count, std::size_t worker_count, Cost cost) {
  const std::size_t count = std::min(item_count, worker_count);
  if (count == 0) return {};
  const auto weight = [&](std::size_t index) {
    return std::max(1.0L, static_cast<long double>(cost(index)));
  };
  long double total = 0;
  for (std::size_t index = 0; index < item_count; ++index) total += weight(index);
  std::vector<EncodingRange> ranges;
  ranges.reserve(count);
  std::size_t begin = 0;
  long double cumulative = 0;
  for (std::size_t worker = 0; worker + 1 < count; ++worker) {
    const long double target = total * (static_cast<long double>(worker + 1) / count);
    const std::size_t limit = item_count - (count - worker - 1);
    std::size_t end = begin + 1;
    cumulative += weight(begin);
    while (end < limit) {
      const long double next = cumulative + weight(end);
      if (std::fabs(next - target) > std::fabs(cumulative - target)) break;
      cumulative = next;
      ++end;
    }
    ranges.push_back({begin, end});
    begin = end;
  }
  ranges.push_back({begin, item_count});
  return ranges;
}

// Dense message work scales with residues times active neighbors. For short
// proteins the scalar encoder uses only min(L, K) active neighbor slots.
inline long double mpnn_encoding_cost(std::size_t residues,
                                     std::size_t neighbors) noexcept {
  return static_cast<long double>(residues) * std::min(residues, neighbors);
}

}  // namespace hikoboshi::algorithms::detail

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_ENCODING_SCHEDULER_HPP

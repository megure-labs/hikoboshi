#ifndef HIKOBOSHI_ALGORITHMS_DETAIL_PATH_BUILDER_HPP
#define HIKOBOSHI_ALGORITHMS_DETAIL_PATH_BUILDER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::algorithms::detail {

class PathBuilder {
 public:
  void prepare(std::size_t max_step_count);
  void reset() noexcept;

  bool push_reverse(hikoboshi::universal::AlignmentStep step) noexcept;
  void set_span(std::int32_t query_start,
                std::int32_t query_end,
                std::int32_t target_start,
                std::int32_t target_end,
                std::size_t aligned_pairs) noexcept;

  void write_ordered_to(hikoboshi::universal::AlignmentPath& path) const;

  hikoboshi::universal::Span<hikoboshi::universal::AlignmentStep> scratch() noexcept;
  std::size_t capacity() const noexcept;
  std::size_t size() const noexcept;

 private:
  std::vector<hikoboshi::universal::AlignmentStep> reverse_steps_;
  std::size_t step_count_ = 0;
  std::size_t aligned_pairs_ = 0;
  std::int32_t query_start_ = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t query_end_ = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t target_start_ = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t target_end_ = hikoboshi::universal::kAlignmentGapSentinel;
};

}  // namespace hikoboshi::algorithms::detail

#endif  // HIKOBOSHI_ALGORITHMS_DETAIL_PATH_BUILDER_HPP

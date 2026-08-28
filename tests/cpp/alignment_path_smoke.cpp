#include <hikoboshi/universal/alignment_path.hpp>

#include <cstddef>
#include <type_traits>

namespace hiko_u = hikoboshi::universal;

static_assert(hiko_u::kAlignmentGapSentinel == -1,
              "hard-SW gap sentinel must stay -1");
static_assert(std::is_standard_layout<hiko_u::AlignmentStep>::value,
              "AlignmentStep must stay a plain public record");

int main() {
  hiko_u::AlignmentPath path{};
  path.steps.push_back({0, 0, 1.25F});
  path.steps.push_back({1, hiko_u::kAlignmentGapSentinel, 0.0F});
  path.aligned_pairs = 1;
  path.query_start = 0;
  path.query_end = 2;
  path.target_start = 0;
  path.target_end = 1;

  return path.steps.size() == 2 && path.steps[1].target_index == -1 &&
                 path.steps[1].residue_score == 0.0F && path.aligned_pairs == 1
             ? 0
             : 1;
}

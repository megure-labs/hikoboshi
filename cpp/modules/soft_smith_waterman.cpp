#include <hikoboshi/modules/soft_smith_waterman.hpp>

#include <hikoboshi/modules/detail/soft_smith_waterman_inline.hpp>

namespace hikoboshi::modules {

hikoboshi::universal::Status soft_smith_waterman(
    const SoftSmithWatermanRequest& request,
    const SoftSmithWatermanOutput& output) noexcept {
  return detail::soft_smith_waterman_inline(request, output);
}

}  // namespace hikoboshi::modules

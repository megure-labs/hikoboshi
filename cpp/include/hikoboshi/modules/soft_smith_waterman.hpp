#ifndef HIKOBOSHI_MODULES_SOFT_SMITH_WATERMAN_HPP
#define HIKOBOSHI_MODULES_SOFT_SMITH_WATERMAN_HPP

#include <cstddef>

#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::modules {

inline constexpr float kSoftSmithWatermanDefaultGapOpen = -3.21337F;
inline constexpr float kSoftSmithWatermanDefaultGapExtension = -0.111704F;
inline constexpr float kSoftSmithWatermanDefaultTemperature = 1.0F;

// Caller-owned soft Smith-Waterman request. The forward + backward
// recurrences reuse three (Lq+1)x(Lt+1) workspaces for forward alphas and
// three for backward grads; both groups must each provide at least
// `workspace_cells` floats.
struct SoftSmithWatermanRequest {
  const float* scores = nullptr;
  std::size_t query_length = 0;
  std::size_t target_length = 0;
  float gap_open = kSoftSmithWatermanDefaultGapOpen;
  float gap_extension = kSoftSmithWatermanDefaultGapExtension;
  float temperature = kSoftSmithWatermanDefaultTemperature;
  float* match_workspace = nullptr;
  float* insert_workspace = nullptr;
  float* delete_workspace = nullptr;
  std::size_t workspace_cells = 0;
  float* match_grad_workspace = nullptr;
  float* insert_grad_workspace = nullptr;
  float* delete_grad_workspace = nullptr;
};

// Caller-owned soft Smith-Waterman output. `posteriors` is row-major
// [query_length, target_length]; the module clears it before computing the
// forward+backward pass. `log_partition` is a single-float scalar that the
// module writes through.
struct SoftSmithWatermanOutput {
  float* log_partition = nullptr;
  float* posteriors = nullptr;
};

hikoboshi::universal::Status soft_smith_waterman(
    const SoftSmithWatermanRequest& request,
    const SoftSmithWatermanOutput& output) noexcept;

}  // namespace hikoboshi::modules

#endif  // HIKOBOSHI_MODULES_SOFT_SMITH_WATERMAN_HPP

#ifndef HIKOBOSHI_MODULES_DETAIL_SOFT_SMITH_WATERMAN_INLINE_HPP
#define HIKOBOSHI_MODULES_DETAIL_SOFT_SMITH_WATERMAN_INLINE_HPP

#include <cmath>
#include <cstddef>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/soft_smith_waterman.hpp>
#include <hikoboshi/universal/inline.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::modules::detail {

HIKOBOSHI_FORCE_INLINE hikoboshi::universal::Status soft_smith_waterman_inline(
    const SoftSmithWatermanRequest& request,
    const SoftSmithWatermanOutput& output) noexcept {
  using hikoboshi::universal::Status;
  using hikoboshi::universal::StatusCode;
  if (request.scores == nullptr) {
    return {StatusCode::InvalidArgument,
            "soft_smith_waterman scores pointer is null"};
  }
  if (output.posteriors == nullptr) {
    return {StatusCode::InvalidArgument,
            "soft_smith_waterman posteriors pointer is null"};
  }
  if (output.log_partition == nullptr) {
    return {StatusCode::InvalidArgument,
            "soft_smith_waterman log_partition pointer is null"};
  }
  if (request.query_length == 0 || request.target_length == 0) {
    return {StatusCode::InvalidArgument,
            "soft_smith_waterman dimensions must be non-zero"};
  }
  if (request.match_workspace == nullptr ||
      request.insert_workspace == nullptr ||
      request.delete_workspace == nullptr ||
      request.match_grad_workspace == nullptr ||
      request.insert_grad_workspace == nullptr ||
      request.delete_grad_workspace == nullptr) {
    return {StatusCode::InvalidArgument,
            "soft_smith_waterman forward and grad workspaces are required"};
  }
  if (!(std::isfinite(request.gap_open) &&
        std::isfinite(request.gap_extension) &&
        std::isfinite(request.temperature))) {
    return {
        StatusCode::InvalidArgument,
        "soft_smith_waterman gap and temperature parameters must be finite"};
  }
  if (!(request.temperature > 0.0F)) {
    return {StatusCode::InvalidArgument,
            "soft_smith_waterman temperature must be positive"};
  }
  const std::size_t row_stride = request.target_length + 1U;
  const std::size_t needed_cells = (request.query_length + 1U) * row_stride;
  if (request.workspace_cells < needed_cells) {
    return {StatusCode::FailedPrecondition,
            "soft_smith_waterman workspace_cells is smaller than "
            "(Lq+1)*(Lt+1)"};
  }
  hikoboshi::primitives::alignment::SoftSmithWatermanScalarRequest scalar_request{};
  scalar_request.scores = request.scores;
  scalar_request.query_length = request.query_length;
  scalar_request.target_length = request.target_length;
  scalar_request.gap_open = request.gap_open;
  scalar_request.gap_extension = request.gap_extension;
  scalar_request.temperature = request.temperature;
  scalar_request.match_workspace = request.match_workspace;
  scalar_request.insert_workspace = request.insert_workspace;
  scalar_request.delete_workspace = request.delete_workspace;
  scalar_request.workspace_cells = request.workspace_cells;
  scalar_request.match_grad_workspace = request.match_grad_workspace;
  scalar_request.insert_grad_workspace = request.insert_grad_workspace;
  scalar_request.delete_grad_workspace = request.delete_grad_workspace;

  hikoboshi::primitives::alignment::SoftSmithWatermanScalarOutput scalar_output{};
  scalar_output.posteriors = output.posteriors;
  scalar_output.log_partition = 0.0F;

  hikoboshi::dispatch::soft_smith_waterman_forward(hikoboshi::dispatch::ScalarTag{},
                                                 scalar_request, scalar_output);

  *output.log_partition = scalar_output.log_partition;
  return {StatusCode::Ok, ""};
}

}  // namespace hikoboshi::modules::detail

#endif  // HIKOBOSHI_MODULES_DETAIL_SOFT_SMITH_WATERMAN_INLINE_HPP

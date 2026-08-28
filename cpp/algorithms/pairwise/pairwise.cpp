#include <hikoboshi/algorithms/pairwise.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <pthread.h>
#include <string>
#include <string_view>
#include <vector>

#include <hikoboshi/algorithms/detail/resolved_alignment.hpp>
#include <hikoboshi/modules/common/weights_views.hpp>
#include <hikoboshi/modules/detail/esm2_layers.hpp>
#include <hikoboshi/modules/detail/esm2_workspace.hpp>
#include <hikoboshi/modules/esm2.hpp>
#include <hikoboshi/modules/similarity.hpp>
#include <hikoboshi/modules/soft_smith_waterman.hpp>

namespace hikoboshi::algorithms {
namespace {

using detail::PairwiseWorkspace;
using detail::SwTraceDirection;
using hikoboshi::universal::AlignmentPath;
using hikoboshi::universal::AlignmentStep;
using hikoboshi::universal::EmbeddingView;
using hikoboshi::universal::ScoreMatrixView;
using hikoboshi::universal::Span;
using hikoboshi::universal::Status;
using hikoboshi::universal::StructureView;

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

struct SwOutput {
  float best_score = 0.0F;
  std::int32_t best_query_index = -1;
  std::int32_t best_target_index = -1;
};

struct TracebackPathOutput {
  Span<AlignmentStep> steps{};
  std::size_t step_count = 0;
  std::size_t aligned_pairs = 0;
  std::int32_t query_start = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t query_end = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t target_start = hikoboshi::universal::kAlignmentGapSentinel;
  std::int32_t target_end = hikoboshi::universal::kAlignmentGapSentinel;
};

bool valid_embedding_view(const EmbeddingView& embedding) noexcept {
  return embedding.values.data != nullptr && embedding.dimension != 0 &&
         embedding.values.size >= embedding.residue_count * embedding.dimension;
}

hikoboshi::universal::AffineGapModel affine_gap_model(
    const PairwiseOptions& options) noexcept {
  return {hikoboshi::universal::GapModel::Affine,
          options.gap_open,
          options.gap_extension,
          hikoboshi::universal::GapConvention::
              GapOpenPlusKMinusOneGapExtension};
}

bool workspace_can_align_scores(const PairwiseWorkspace& workspace,
                                std::size_t query_length,
                                std::size_t target_length) noexcept {
  if (query_length > workspace.plan().max_query_length ||
      target_length > workspace.plan().max_target_length) {
    return false;
  }
  if (query_length == std::numeric_limits<std::size_t>::max() ||
      target_length == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const std::size_t row_count = query_length + 1U;
  const std::size_t row_stride = target_length + 1U;
  if (row_stride != 0U &&
      row_count > std::numeric_limits<std::size_t>::max() / row_stride) {
    return false;
  }
  return workspace.sw_workspace_cells() >= row_count * row_stride;
}

bool checked_score_matrix_cell_count(std::size_t query_length,
                                     std::size_t target_length,
                                     std::size_t& cell_count) noexcept {
  if (query_length != 0U &&
      target_length > std::numeric_limits<std::size_t>::max() / query_length) {
    return false;
  }
  cell_count = query_length * target_length;
  return true;
}

Status validate_resolved_alignment_problem(
    const detail::ResolvedAlignmentProblem& problem,
    const PairwiseWorkspace& workspace) noexcept {
  if (problem.scores.values == nullptr) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment score matrix values pointer is null");
  }
  if (problem.scores.query_length == 0U ||
      problem.scores.target_length == 0U) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment score matrix dimensions must be non-zero");
  }
  if (problem.scores.row_stride != problem.scores.target_length) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment score matrix must be contiguous row-major");
  }
  if (!workspace_can_align_scores(workspace, problem.scores.query_length,
                                  problem.scores.target_length)) {
    return hikoboshi::universal::failed_precondition_status(
        "pairwise workspace has insufficient score-matrix capacity");
  }
  if (!detail::is_raw_dot_v1_score_semantics(problem.semantics)) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment score semantics are unsupported");
  }
  if (problem.gaps.model != hikoboshi::universal::GapModel::Affine ||
      problem.gaps.convention !=
          hikoboshi::universal::GapConvention::
              GapOpenPlusKMinusOneGapExtension) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment gap model is unsupported");
  }
  if (!std::isfinite(problem.gaps.gap_open) ||
      !std::isfinite(problem.gaps.gap_extension)) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment gap parameters must be finite");
  }
  if (problem.algorithm != detail::kHardLocalAffineSwV1Algorithm) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment algorithm is unsupported");
  }
  if (!detail::is_public_traceback_required(problem.traceback)) {
    return hikoboshi::universal::invalid_argument_status("resolved alignment traceback policy is unsupported");
  }
  return hikoboshi::universal::ok_status();
}

Span<const char> structure_codes(const StructureView& structure) noexcept {
  if (structure.residue_codes.data != nullptr &&
      structure.residue_codes.size >= structure.residue_count) {
    return structure.residue_codes;
  }
  return {nullptr, 0};
}

EmbeddingView make_workspace_embedding(const float* values,
                                       const StructureView& structure,
                                       std::size_t dimension) noexcept {
  return {structure.residue_count,
          dimension,
          {values, structure.residue_count * dimension},
          structure_codes(structure),
          structure.residues};
}

void clear_tracebacks(PairwiseWorkspace& workspace, std::size_t cell_count) noexcept {
  std::fill(workspace.trace_match_data(),
            workspace.trace_match_data() + cell_count,
            SwTraceDirection::Stop);
  std::fill(workspace.trace_insert_data(),
            workspace.trace_insert_data() + cell_count,
            SwTraceDirection::Stop);
  std::fill(workspace.trace_delete_data(),
            workspace.trace_delete_data() + cell_count,
            SwTraceDirection::Stop);
}

SwOutput run_hard_sw(const float* scores,
                     std::size_t query_length,
                     std::size_t target_length,
                     float gap_open,
                     float gap_extension,
                     PairwiseWorkspace& workspace) noexcept {
  SwOutput output{};
  const std::size_t trace_cells = query_length * target_length;
  clear_tracebacks(workspace, trace_cells);
  if (query_length == 0 || target_length == 0) {
    return output;
  }

  const std::size_t row_stride = target_length + 1;
  const std::size_t workspace_cells = (query_length + 1) * row_stride;
  float* match = workspace.match_workspace_data();
  float* insert_q = workspace.insert_workspace_data();
  float* delete_t = workspace.delete_workspace_data();

  std::fill(match, match + workspace_cells, 0.0F);
  std::fill(insert_q, insert_q + workspace_cells, kNegativeInfinity);
  std::fill(delete_t, delete_t + workspace_cells, kNegativeInfinity);

  for (std::size_t j = 0; j <= target_length; ++j) {
    insert_q[j] = kNegativeInfinity;
    delete_t[j] = kNegativeInfinity;
  }
  for (std::size_t i = 0; i <= query_length; ++i) {
    insert_q[i * row_stride] = kNegativeInfinity;
    delete_t[i * row_stride] = kNegativeInfinity;
  }

  for (std::size_t i = 1; i <= query_length; ++i) {
    for (std::size_t j = 1; j <= target_length; ++j) {
      const std::size_t trace_index = (i - 1) * target_length + (j - 1);
      const float residue_score = scores[trace_index];

      const float prev_match = match[(i - 1) * row_stride + (j - 1)];
      const float prev_insert = insert_q[(i - 1) * row_stride + (j - 1)];
      const float prev_delete = delete_t[(i - 1) * row_stride + (j - 1)];

      SwTraceDirection match_pred = SwTraceDirection::Stop;
      float best_pred = 0.0F;
      if (prev_match > best_pred) {
        best_pred = prev_match;
        match_pred = SwTraceDirection::Match;
      }
      if (prev_insert > best_pred) {
        best_pred = prev_insert;
        match_pred = SwTraceDirection::InsertQuery;
      }
      if (prev_delete > best_pred) {
        best_pred = prev_delete;
        match_pred = SwTraceDirection::DeleteTarget;
      }
      match[i * row_stride + j] = residue_score + best_pred;
      workspace.trace_match_data()[trace_index] = match_pred;

      const float open_insert =
          match[(i - 1) * row_stride + j] + gap_open;
      const float extend_insert =
          insert_q[(i - 1) * row_stride + j] + gap_extension;
      SwTraceDirection insert_pred = SwTraceDirection::Match;
      float insert_value = open_insert;
      if (extend_insert > insert_value) {
        insert_value = extend_insert;
        insert_pred = SwTraceDirection::InsertQuery;
      }
      insert_q[i * row_stride + j] = insert_value;
      workspace.trace_insert_data()[trace_index] = insert_pred;

      const float open_delete = match[i * row_stride + (j - 1)] + gap_open;
      const float extend_delete =
          delete_t[i * row_stride + (j - 1)] + gap_extension;
      SwTraceDirection delete_pred = SwTraceDirection::Match;
      float delete_value = open_delete;
      if (extend_delete > delete_value) {
        delete_value = extend_delete;
        delete_pred = SwTraceDirection::DeleteTarget;
      }
      delete_t[i * row_stride + j] = delete_value;
      workspace.trace_delete_data()[trace_index] = delete_pred;
    }
  }

  float best_score = 0.0F;
  std::int32_t best_i = -1;
  std::int32_t best_j = -1;
  for (std::size_t i = 1; i <= query_length; ++i) {
    for (std::size_t j = 1; j <= target_length; ++j) {
      const float candidate = match[i * row_stride + j];
      if (candidate > best_score) {
        best_score = candidate;
        best_i = static_cast<std::int32_t>(i - 1);
        best_j = static_cast<std::int32_t>(j - 1);
      }
    }
  }
  output.best_score = best_score;
  output.best_query_index = best_i;
  output.best_target_index = best_j;
  return output;
}

void reset_empty_path(AlignmentPath& path) noexcept {
  path.steps.clear();
  path.aligned_pairs = 0;
  path.query_start = hikoboshi::universal::kAlignmentGapSentinel;
  path.query_end = hikoboshi::universal::kAlignmentGapSentinel;
  path.target_start = hikoboshi::universal::kAlignmentGapSentinel;
  path.target_end = hikoboshi::universal::kAlignmentGapSentinel;
}

void reset_traceback_output(TracebackPathOutput& output) noexcept {
  output.step_count = 0;
  output.aligned_pairs = 0;
  output.query_start = hikoboshi::universal::kAlignmentGapSentinel;
  output.query_end = hikoboshi::universal::kAlignmentGapSentinel;
  output.target_start = hikoboshi::universal::kAlignmentGapSentinel;
  output.target_end = hikoboshi::universal::kAlignmentGapSentinel;
}

bool push_traceback_step(TracebackPathOutput& output,
                         std::size_t& write_index,
                         AlignmentStep step) noexcept {
  if (output.steps.data == nullptr || write_index == 0) {
    return false;
  }
  --write_index;
  output.steps.data[write_index] = step;
  return true;
}

void compact_traceback_steps(TracebackPathOutput& output,
                             std::size_t write_index) noexcept {
  const std::size_t count = output.steps.size - write_index;
  for (std::size_t index = 0; index < count; ++index) {
    output.steps.data[index] = output.steps.data[write_index + index];
  }
  output.step_count = count;
}

void write_traceback_output_to_path(const TracebackPathOutput& output,
                                    AlignmentPath& path) {
  if (output.step_count == 0) {
    reset_empty_path(path);
    return;
  }
  path.steps.assign(output.steps.data,
                    output.steps.data + output.step_count);
  path.aligned_pairs = output.aligned_pairs;
  path.query_start = output.query_start;
  path.query_end = output.query_end;
  path.target_start = output.target_start;
  path.target_end = output.target_end;
}

Status build_traceback_path(const float* scores,
                            std::size_t query_length,
                            std::size_t target_length,
                            const SwOutput& sw,
                            PairwiseWorkspace& workspace,
                            AlignmentPath& path) {
  (void)query_length;
  reset_empty_path(path);
  workspace.path_builder().reset();
  if (sw.best_query_index < 0 || sw.best_target_index < 0) {
    return hikoboshi::universal::ok_status();
  }

  TracebackPathOutput output{};
  output.steps = workspace.traceback_step_scratch();
  reset_traceback_output(output);

  std::int32_t i = sw.best_query_index;
  std::int32_t j = sw.best_target_index;
  output.query_end = i;
  output.target_end = j;
  output.query_start = i;
  output.target_start = j;
  std::size_t write_index = output.steps.size;
  SwTraceDirection state = SwTraceDirection::Match;

  while (i >= 0 && j >= 0) {
    const std::size_t cell =
        static_cast<std::size_t>(i) * target_length +
        static_cast<std::size_t>(j);
    if (state == SwTraceDirection::Match) {
      AlignmentStep step{};
      step.query_index = i;
      step.target_index = j;
      step.residue_score = scores[cell];
      if (!push_traceback_step(output, write_index, step)) {
        return hikoboshi::universal::failed_precondition_status("pairwise path scratch is too small");
      }
      ++output.aligned_pairs;
      output.query_start = i;
      output.target_start = j;
      const SwTraceDirection predecessor = workspace.trace_match_data()[cell];
      if (predecessor == SwTraceDirection::Stop) {
        break;
      }
      state = predecessor;
      --i;
      --j;
    } else if (state == SwTraceDirection::InsertQuery) {
      AlignmentStep step{};
      step.query_index = i;
      step.target_index = hikoboshi::universal::kAlignmentGapSentinel;
      step.residue_score = 0.0F;
      if (!push_traceback_step(output, write_index, step)) {
        return hikoboshi::universal::failed_precondition_status("pairwise path scratch is too small");
      }
      const SwTraceDirection predecessor = workspace.trace_insert_data()[cell];
      state = predecessor == SwTraceDirection::InsertQuery
                  ? SwTraceDirection::InsertQuery
                  : SwTraceDirection::Match;
      --i;
    } else {
      AlignmentStep step{};
      step.query_index = hikoboshi::universal::kAlignmentGapSentinel;
      step.target_index = j;
      step.residue_score = 0.0F;
      if (!push_traceback_step(output, write_index, step)) {
        return hikoboshi::universal::failed_precondition_status("pairwise path scratch is too small");
      }
      const SwTraceDirection predecessor = workspace.trace_delete_data()[cell];
      state = predecessor == SwTraceDirection::DeleteTarget
                  ? SwTraceDirection::DeleteTarget
                  : SwTraceDirection::Match;
      --j;
    }
  }

  compact_traceback_steps(output, write_index);
  write_traceback_output_to_path(output, path);
  return hikoboshi::universal::ok_status();
}

Status build_similarity_score_matrix(const PairwiseEmbeddingRequest& request,
                                     PairwiseWorkspace& workspace,
                                     ScoreMatrixView& scores) noexcept {
  std::size_t score_cells = 0;
  if (!checked_score_matrix_cell_count(request.query_embedding.residue_count,
                                       request.target_embedding.residue_count,
                                       score_cells)) {
    return hikoboshi::universal::invalid_argument_status(
        "pairwise score matrix size overflows");
  }
  if (workspace.score_matrix_capacity() < score_cells) {
    return hikoboshi::universal::failed_precondition_status(
        "pairwise workspace score-matrix buffer is smaller than the input pair");
  }

  hikoboshi::modules::SimilarityScalarRequest similarity_request{};
  similarity_request.query_embedding = request.query_embedding;
  similarity_request.target_embedding = request.target_embedding;
  hikoboshi::modules::SimilarityScalarOutput similarity_output{};
  similarity_output.similarity_matrix = workspace.similarity_data();
  similarity_output.query_count = request.query_embedding.residue_count;
  similarity_output.target_count = request.target_embedding.residue_count;

  const Status similarity_status =
      hikoboshi::modules::similarity_scalar(similarity_request, similarity_output);
  if (!hikoboshi::universal::is_ok(similarity_status)) {
    return similarity_status;
  }
  scores = hikoboshi::modules::score_matrix_view(similarity_output);
  return hikoboshi::universal::ok_status();
}

Status run_hard_pairwise_alignment(const PairwiseEmbeddingRequest& request,
                                   ScoreMatrixView scores,
                                   PairwiseWorkspace& workspace,
                                   AlignmentPath& path,
                                   double& raw_sw_score) {
  const detail::ResolvedAlignmentProblem problem{
      scores,
      hikoboshi::modules::kSimilarityScalarScoreSemantics,
      affine_gap_model(request.options),
      detail::kHardLocalAffineSwV1Algorithm,
      detail::kPublicPairwiseTracebackPolicy,
      request.query_structure,
      request.target_structure,
  };

  return detail::run_resolved_alignment_problem(problem, workspace, path,
                                                raw_sw_score);
}

Status run_soft_pairwise_alignment(const PairwiseEmbeddingRequest& request,
                                   ScoreMatrixView scores,
                                   PairwiseWorkspace& workspace,
                                   AlignmentPath* path,
                                   double& raw_sw_score) {
  if (path != nullptr) {
    reset_empty_path(*path);
  }
  raw_sw_score = 0.0;

  if (!std::isfinite(request.temperature) || !(request.temperature > 0.0F)) {
    return hikoboshi::universal::invalid_argument_status(
        "pairwise soft mode temperature must be a positive finite float");
  }
  if (!workspace.has_soft_sw_workspaces()) {
    return hikoboshi::universal::failed_precondition_status(
        "pairwise soft mode requires the workspace to be prepared with allocate_soft_sw=true");
  }
  std::size_t score_cells = 0;
  if (!checked_score_matrix_cell_count(scores.query_length, scores.target_length,
                                       score_cells)) {
    return hikoboshi::universal::invalid_argument_status(
        "pairwise soft mode score matrix size overflows");
  }
  if (workspace.posteriors_capacity() < score_cells) {
    return hikoboshi::universal::failed_precondition_status(
        "pairwise soft mode posterior workspace is smaller than the score matrix");
  }
  float log_partition = 0.0F;
  hikoboshi::modules::SoftSmithWatermanRequest soft_request{};
  soft_request.scores = scores.values;
  soft_request.query_length = scores.query_length;
  soft_request.target_length = scores.target_length;
  soft_request.gap_open = request.options.soft_gap_open;
  soft_request.gap_extension = request.options.soft_gap_extension;
  soft_request.temperature = request.temperature;
  soft_request.match_workspace = workspace.match_workspace_data();
  soft_request.insert_workspace = workspace.insert_workspace_data();
  soft_request.delete_workspace = workspace.delete_workspace_data();
  soft_request.workspace_cells = workspace.sw_workspace_cells();
  soft_request.match_grad_workspace = workspace.match_grad_workspace_data();
  soft_request.insert_grad_workspace = workspace.insert_grad_workspace_data();
  soft_request.delete_grad_workspace = workspace.delete_grad_workspace_data();
  hikoboshi::modules::SoftSmithWatermanOutput soft_output{};
  soft_output.log_partition = &log_partition;
  soft_output.posteriors = workspace.posteriors_data();
  const Status soft_status =
      hikoboshi::modules::soft_smith_waterman(soft_request, soft_output);
  if (!hikoboshi::universal::is_ok(soft_status)) {
    return soft_status;
  }

  raw_sw_score = static_cast<double>(log_partition);
  if (path == nullptr) {
    return hikoboshi::universal::ok_status();
  }

  const float* posteriors = soft_output.posteriors;
  const float threshold = kPairwiseSoftPosteriorThreshold;

  std::size_t aligned_pairs = 0;
  std::int32_t qmin = std::numeric_limits<std::int32_t>::max();
  std::int32_t qmax = -1;
  std::int32_t tmin = std::numeric_limits<std::int32_t>::max();
  std::int32_t tmax = -1;
  for (std::size_t i = 0; i < scores.query_length; ++i) {
    for (std::size_t j = 0; j < scores.target_length; ++j) {
      if (posteriors[i * scores.target_length + j] > threshold) {
        ++aligned_pairs;
        const std::int32_t qi = static_cast<std::int32_t>(i);
        const std::int32_t tj = static_cast<std::int32_t>(j);
        if (qi < qmin) qmin = qi;
        if (qi > qmax) qmax = qi;
        if (tj < tmin) tmin = tj;
        if (tj > tmax) tmax = tj;
      }
    }
  }

  if (aligned_pairs == 0) {
    return hikoboshi::universal::ok_status();
  }

  path->steps.assign(aligned_pairs, AlignmentStep{});
  std::size_t step_index = 0;
  for (std::size_t i = 0; i < scores.query_length; ++i) {
    for (std::size_t j = 0; j < scores.target_length; ++j) {
      const std::size_t cell = i * scores.target_length + j;
      if (posteriors[cell] > threshold) {
        AlignmentStep& step = path->steps[step_index++];
        step.query_index = static_cast<std::int32_t>(i);
        step.target_index = static_cast<std::int32_t>(j);
        step.residue_score = scores.values[cell];
      }
    }
  }
  path->aligned_pairs = aligned_pairs;
  path->query_start = qmin;
  path->query_end = qmax;
  path->target_start = tmin;
  path->target_end = tmax;
  return hikoboshi::universal::ok_status();
}

// Runtime override that forces sequential structure encoding. Read once per
// `run_pairwise_structures` call so test harnesses can flip between the
// parallel and sequential paths through `setenv`/`unsetenv`. Any non-empty
// value other than "0" (case-sensitive) disables the parallel path.
bool parallel_structure_encode_disabled() noexcept {
  const char* raw = std::getenv("HIKOBOSHI_PAIRWISE_DISABLE_PARALLEL_ENCODE");
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  return std::strcmp(raw, "0") != 0;
}

Status encode_mpnn64_target_for_pairwise(
    hikoboshi::modules::Mpnn64ForwardRequest target_request,
    hikoboshi::modules::Mpnn64ForwardOutput target_output) noexcept {
  return hikoboshi::modules::mpnn64_forward_scalar(target_request, target_output);
}

struct Mpnn64TargetEncodeJob {
  hikoboshi::modules::Mpnn64ForwardRequest request{};
  hikoboshi::modules::Mpnn64ForwardOutput output{};
  Status status = hikoboshi::universal::ok_status();
};

void* run_mpnn64_target_encode_job(void* raw_job) noexcept {
  auto* job = static_cast<Mpnn64TargetEncodeJob*>(raw_job);
  job->status = encode_mpnn64_target_for_pairwise(job->request, job->output);
  return nullptr;
}

Status score_embeddings_and_run_resolved(const PairwiseEmbeddingRequest& request,
                                         PairwiseWorkspace& workspace,
                                         PairwiseResult& result) {
  if (!valid_embedding_view(request.query_embedding) ||
      !valid_embedding_view(request.target_embedding)) {
    return hikoboshi::universal::invalid_argument_status("pairwise embedding inputs are invalid");
  }
  if (request.query_embedding.dimension != request.target_embedding.dimension) {
    return hikoboshi::universal::invalid_argument_status("pairwise embedding dimensions differ");
  }
  if (!std::isfinite(request.options.gap_open) ||
      !std::isfinite(request.options.gap_extension)) {
    return hikoboshi::universal::invalid_argument_status("pairwise gap parameters must be finite");
  }
  if (!workspace.can_align(request.query_embedding.residue_count,
                           request.target_embedding.residue_count,
                           request.query_embedding.dimension)) {
    return hikoboshi::universal::failed_precondition_status("pairwise workspace has insufficient capacity");
  }
  const bool run_soft = request.soft_mode;
  const bool run_hard = request.hard_mode || !run_soft;

  ScoreMatrixView scores{};
  const Status similarity_status =
      build_similarity_score_matrix(request, workspace, scores);
  if (!hikoboshi::universal::is_ok(similarity_status)) {
    return similarity_status;
  }

  double raw_sw_score = 0.0;
  hikoboshi::universal::MetricValue soft_sw_score =
      invalid_metric(hikoboshi::universal::MetricInvalidReason::Unavailable);
  if (run_hard) {
    const Status resolved_status = run_hard_pairwise_alignment(
        request, scores, workspace, result.path, raw_sw_score);
    if (!hikoboshi::universal::is_ok(resolved_status)) {
      return resolved_status;
    }
  }
  if (run_soft) {
    double soft_score = 0.0;
    if (run_hard) {
      const Status soft_status = run_soft_pairwise_alignment(
          request, scores, workspace, nullptr, soft_score);
      if (!hikoboshi::universal::is_ok(soft_status)) {
        return soft_status;
      }
    } else {
      const Status soft_status = run_soft_pairwise_alignment(
          request, scores, workspace, &result.path, soft_score);
      if (!hikoboshi::universal::is_ok(soft_status)) {
        return soft_status;
      }
      raw_sw_score = soft_score;
    }
    soft_sw_score = valid_metric(soft_score);
  }

  assemble_pairwise_result(raw_sw_score, result.path,
                           request.query_embedding, request.target_embedding,
                           request.query_structure, request.target_structure,
                           result);
  result.metrics.soft_sw_score = soft_sw_score;
  return hikoboshi::universal::ok_status();
}

}  // namespace

namespace detail {

Status run_resolved_alignment_problem(
    const ResolvedAlignmentProblem& problem,
    PairwiseWorkspace& workspace,
    hikoboshi::universal::AlignmentPath& path,
    double& raw_sw_score) {
  raw_sw_score = 0.0;
  reset_empty_path(path);

  const Status validation_status =
      validate_resolved_alignment_problem(problem, workspace);
  if (!hikoboshi::universal::is_ok(validation_status)) {
    return validation_status;
  }

  const SwOutput sw = run_hard_sw(problem.scores.values,
                                  problem.scores.query_length,
                                  problem.scores.target_length,
                                  problem.gaps.gap_open,
                                  problem.gaps.gap_extension, workspace);
  raw_sw_score = static_cast<double>(sw.best_score);

  return build_traceback_path(problem.scores.values,
                              problem.scores.query_length,
                              problem.scores.target_length, sw, workspace,
                              path);
}

}  // namespace detail

Status run_pairwise_embeddings(const PairwiseEmbeddingRequest& request,
                               PairwiseWorkspace& workspace,
                               PairwiseResult& result) noexcept {
  return score_embeddings_and_run_resolved(request, workspace, result);
}

namespace {

namespace hiko_m = hikoboshi::modules;
namespace hiko_mc = hikoboshi::modules::common;
namespace hiko_md = hikoboshi::modules::detail;
namespace hiko_mt = hikoboshi::modules::transformer::detail;
namespace hiko_mf = hikoboshi::modules::ffn::detail;
namespace hiko_u = hikoboshi::universal;

// ESM2-8M prepared state owned by the sequence-input pipeline.
//
// The weights layer hands the algorithms layer a borrowed `WeightsView`
// whose tensor metadata enumerates every embedded safetensors slot by
// name; this struct binds those slots into a typed `Esm2Weights` view and
// hosts the heap-owned `Esm2Workspace` buffers the forward pass borrows.
// One instance covers both the query and target forward passes within a
// single pairwise call (they run sequentially and reuse the same scratch
// arena).
struct Esm2PreparedState {
  hiko_md::Esm2Weights weights{};
  std::vector<hiko_md::Esm2LayerWeights> layers;
  hiko_md::Esm2Workspace workspace{};
  std::vector<float> hidden_state;
  std::vector<float> hidden_state_post_attn;
  std::vector<float> ffn_norm_buffer;
  std::vector<float> ffn_residual_buffer;
  std::vector<float> rope_cos;
  std::vector<float> rope_sin;
  std::vector<float> attn_norm_buffer;
  std::vector<float> attn_q_buffer;
  std::vector<float> attn_k_buffer;
  std::vector<float> attn_v_buffer;
  std::vector<float> attn_q_head_buffer;
  std::vector<float> attn_k_head_buffer;
  std::vector<float> attn_v_head_buffer;
  std::vector<float> attn_scores_buffer;
  std::vector<float> attn_head_out_buffer;
  std::vector<float> attn_concat_buffer;
  std::vector<float> attn_attn_buffer;
  std::vector<float> ffn_intermediate_buffer;
};

const hiko_u::TensorView* find_tensor_view(const hiko_u::WeightsView& view,
                                        std::string_view name) noexcept {
  for (std::size_t index = 0; index < view.tensors.size; ++index) {
    if (view.tensors.data[index].name == name) {
      return &view.tensors.data[index];
    }
  }
  return nullptr;
}

bool tensor_dims_match(const hiko_u::TensorView& tensor,
                       std::size_t expected_rank,
                       std::size_t expected_dim0,
                       std::size_t expected_dim1) noexcept {
  if (tensor.shape.size != expected_rank) {
    return false;
  }
  if (tensor.shape.data[0] != expected_dim0) {
    return false;
  }
  if (expected_rank >= 2U && tensor.shape.data[1] != expected_dim1) {
    return false;
  }
  return true;
}

const float* tensor_as_float(const hiko_u::TensorView* tensor) noexcept {
  if (tensor == nullptr || tensor->dtype != hiko_u::DataType::Float32) {
    return nullptr;
  }
  return static_cast<const float*>(tensor->data);
}

Status bind_norm_view(const hiko_u::WeightsView& view,
                      std::string_view prefix,
                      std::size_t hidden,
                      hiko_mc::NormLayerWeightsView& out) noexcept {
  const std::string weight_name = std::string(prefix) + ".weight";
  const std::string bias_name = std::string(prefix) + ".bias";
  const hiko_u::TensorView* weight = find_tensor_view(view, weight_name);
  const hiko_u::TensorView* bias = find_tensor_view(view, bias_name);
  if (weight == nullptr || bias == nullptr) {
    return hiko_u::invalid_argument_status(
        "ESM2 weights view is missing a required LayerNorm slot");
  }
  if (!tensor_dims_match(*weight, 1, hidden, 0)) {
    return hiko_u::invalid_argument_status(
        "ESM2 LayerNorm weight tensor has unexpected shape");
  }
  if (!tensor_dims_match(*bias, 1, hidden, 0)) {
    return hiko_u::invalid_argument_status(
        "ESM2 LayerNorm bias tensor has unexpected shape");
  }
  out.gamma = tensor_as_float(weight);
  out.beta = tensor_as_float(bias);
  out.dim = hidden;
  out.epsilon = hiko_md::kEsm2LayerNormEpsilon;
  if (out.gamma == nullptr || out.beta == nullptr) {
    return hiko_u::invalid_argument_status(
        "ESM2 LayerNorm tensors must be float32");
  }
  return hiko_u::ok_status();
}

Status bind_linear_view(const hiko_u::WeightsView& view,
                        std::string_view prefix,
                        std::size_t output_dim,
                        std::size_t input_dim,
                        hiko_mc::LinearLayerWeightsView& out) noexcept {
  const std::string weight_name = std::string(prefix) + ".weight";
  const std::string bias_name = std::string(prefix) + ".bias";
  const hiko_u::TensorView* weight = find_tensor_view(view, weight_name);
  const hiko_u::TensorView* bias = find_tensor_view(view, bias_name);
  if (weight == nullptr) {
    return hiko_u::invalid_argument_status(
        "ESM2 weights view is missing a required linear weight slot");
  }
  if (!tensor_dims_match(*weight, 2, output_dim, input_dim)) {
    return hiko_u::invalid_argument_status(
        "ESM2 linear weight tensor has unexpected shape");
  }
  out.weight = tensor_as_float(weight);
  out.bias = tensor_as_float(bias);
  out.output_dim = output_dim;
  out.input_dim = input_dim;
  if (out.weight == nullptr) {
    return hiko_u::invalid_argument_status(
        "ESM2 linear weight tensor must be float32");
  }
  // Bias is required for ESM2-8M but may be absent in future bias-free
  // architectures; the existing ExpectedTensor table guarantees presence.
  if (bias == nullptr || !tensor_dims_match(*bias, 1, output_dim, 0)) {
    return hiko_u::invalid_argument_status(
        "ESM2 linear bias tensor missing or has unexpected shape");
  }
  return hiko_u::ok_status();
}

Status build_esm2_weights(const hiko_u::WeightsView& view,
                          const hiko_m::Esm2Descriptor& descriptor,
                          Esm2PreparedState& state) noexcept {
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t ffn = descriptor.ffn_hidden_dimension;
  if (descriptor.layer_count == 0 || hidden == 0 || ffn == 0) {
    return hiko_u::invalid_argument_status(
        "ESM2 descriptor must declare a non-zero layer count and dims");
  }

  const hiko_u::TensorView* embedding =
      find_tensor_view(view, "embedding_table");
  if (embedding == nullptr ||
      !tensor_dims_match(*embedding, 2, descriptor.vocab_size, hidden)) {
    return hiko_u::invalid_argument_status(
        "ESM2 weights view is missing or mis-shaped embedding_table");
  }
  const float* embedding_data = tensor_as_float(embedding);
  if (embedding_data == nullptr) {
    return hiko_u::invalid_argument_status(
        "ESM2 embedding_table must be float32");
  }

  state.layers.assign(descriptor.layer_count, hiko_md::Esm2LayerWeights{});
  for (std::size_t layer = 0; layer < descriptor.layer_count; ++layer) {
    hiko_md::Esm2LayerWeights& w = state.layers[layer];
    const std::string base = "layers." + std::to_string(layer);
    Status status =
        bind_norm_view(view, base + ".attn.pre_norm", hidden, w.attn_pre_norm);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
    status = bind_linear_view(view, base + ".attn.q_proj", hidden, hidden, w.wq);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
    status = bind_linear_view(view, base + ".attn.k_proj", hidden, hidden, w.wk);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
    status = bind_linear_view(view, base + ".attn.v_proj", hidden, hidden, w.wv);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
    status = bind_linear_view(view, base + ".attn.out_proj", hidden, hidden, w.wo);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
    status = bind_norm_view(view, base + ".ffn.pre_norm", hidden, w.ffn_pre_norm);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
    status = bind_linear_view(view, base + ".ffn.in", ffn, hidden, w.ffn_in);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
    status = bind_linear_view(view, base + ".ffn.out", hidden, ffn, w.ffn_out);
    if (!hiko_u::is_ok(status)) {
      return status;
    }
  }
  Status status =
      bind_norm_view(view, "final_norm", hidden, state.weights.final_norm);
  if (!hiko_u::is_ok(status)) {
    return status;
  }

  state.weights.embedding_table = embedding_data;
  state.weights.vocab_size = descriptor.vocab_size;
  state.weights.hidden_dimension = hidden;
  state.weights.layer_count = descriptor.layer_count;
  state.weights.layers = state.layers.data();
  return hiko_u::ok_status();
}

void prepare_esm2_workspace(const hiko_m::Esm2Descriptor& descriptor,
                            std::size_t seq_len,
                            Esm2PreparedState& state) {
  const std::size_t hidden = descriptor.hidden_dimension;
  const std::size_t head_count = descriptor.head_count;
  const std::size_t head_dim = descriptor.head_dim;
  const std::size_t ffn = descriptor.ffn_hidden_dimension;
  const std::size_t half = head_dim / 2U;

  const std::size_t hidden_count = seq_len * hidden;
  const std::size_t rope_count = seq_len * half;
  const std::size_t head_major = head_count * seq_len * head_dim;

  state.hidden_state.assign(hidden_count, 0.0F);
  state.hidden_state_post_attn.assign(hidden_count, 0.0F);
  state.ffn_norm_buffer.assign(hidden_count, 0.0F);
  state.ffn_residual_buffer.assign(hidden_count, 0.0F);
  state.rope_cos.assign(rope_count, 0.0F);
  state.rope_sin.assign(rope_count, 0.0F);
  for (std::size_t s = 0; s < seq_len; ++s) {
    for (std::size_t d = 0; d < half; ++d) {
      const float theta = static_cast<float>(s) /
                          std::pow(10000.0F, static_cast<float>(2 * d) /
                                                 static_cast<float>(head_dim));
      state.rope_cos[s * half + d] = std::cos(theta);
      state.rope_sin[s * half + d] = std::sin(theta);
    }
  }
  state.attn_norm_buffer.assign(hidden_count, 0.0F);
  state.attn_q_buffer.assign(hidden_count, 0.0F);
  state.attn_k_buffer.assign(hidden_count, 0.0F);
  state.attn_v_buffer.assign(hidden_count, 0.0F);
  state.attn_q_head_buffer.assign(head_major, 0.0F);
  state.attn_k_head_buffer.assign(head_major, 0.0F);
  state.attn_v_head_buffer.assign(head_major, 0.0F);
  state.attn_scores_buffer.assign(seq_len * seq_len, 0.0F);
  state.attn_head_out_buffer.assign(head_major, 0.0F);
  state.attn_concat_buffer.assign(hidden_count, 0.0F);
  state.attn_attn_buffer.assign(hidden_count, 0.0F);
  state.ffn_intermediate_buffer.assign(seq_len * ffn, 0.0F);

  state.workspace.plan = {seq_len, hidden, head_count, head_dim, ffn};
  state.workspace.hidden_state = {state.hidden_state.data(),
                                  state.hidden_state.size()};
  state.workspace.hidden_state_post_attn = {
      state.hidden_state_post_attn.data(),
      state.hidden_state_post_attn.size()};
  state.workspace.ffn_norm_buffer = {state.ffn_norm_buffer.data(),
                                     state.ffn_norm_buffer.size()};
  state.workspace.ffn_residual_buffer = {state.ffn_residual_buffer.data(),
                                         state.ffn_residual_buffer.size()};
  state.workspace.rope_cos = {state.rope_cos.data(), state.rope_cos.size()};
  state.workspace.rope_sin = {state.rope_sin.data(), state.rope_sin.size()};
  state.workspace.attention_workspace.plan = {seq_len, hidden, head_count,
                                              head_dim};
  state.workspace.attention_workspace.norm_buffer = {
      state.attn_norm_buffer.data(), state.attn_norm_buffer.size()};
  state.workspace.attention_workspace.q_buffer = {state.attn_q_buffer.data(),
                                                  state.attn_q_buffer.size()};
  state.workspace.attention_workspace.k_buffer = {state.attn_k_buffer.data(),
                                                  state.attn_k_buffer.size()};
  state.workspace.attention_workspace.v_buffer = {state.attn_v_buffer.data(),
                                                  state.attn_v_buffer.size()};
  state.workspace.attention_workspace.q_head_buffer = {
      state.attn_q_head_buffer.data(), state.attn_q_head_buffer.size()};
  state.workspace.attention_workspace.k_head_buffer = {
      state.attn_k_head_buffer.data(), state.attn_k_head_buffer.size()};
  state.workspace.attention_workspace.v_head_buffer = {
      state.attn_v_head_buffer.data(), state.attn_v_head_buffer.size()};
  state.workspace.attention_workspace.scores_buffer = {
      state.attn_scores_buffer.data(), state.attn_scores_buffer.size()};
  state.workspace.attention_workspace.head_out_buffer = {
      state.attn_head_out_buffer.data(), state.attn_head_out_buffer.size()};
  state.workspace.attention_workspace.concat_buffer = {
      state.attn_concat_buffer.data(), state.attn_concat_buffer.size()};
  state.workspace.attention_workspace.attn_buffer = {
      state.attn_attn_buffer.data(), state.attn_attn_buffer.size()};
  state.workspace.ffn_workspace.intermediate_buffer =
      state.ffn_intermediate_buffer.data();
  state.workspace.ffn_workspace.intermediate_capacity =
      state.ffn_intermediate_buffer.size();
}

Status encode_sequence_into(Esm2PreparedState& state,
                            const hiko_m::Esm2Descriptor& descriptor,
                            hiko_u::Span<const std::int32_t> token_ids,
                            float* embeddings_out) noexcept {
  hiko_m::Esm2ForwardRequest request{};
  request.token_ids = token_ids.data;
  request.seq_len = token_ids.size;
  request.descriptor = descriptor;
  request.weights = &state.weights;
  request.workspace = &state.workspace;
  hiko_m::Esm2ForwardOutput output{};
  output.embeddings = embeddings_out;
  output.seq_len = token_ids.size;
  output.hidden_dimension = descriptor.hidden_dimension;
  return hiko_m::esm2_forward_scalar(request, output);
}

}  // namespace

namespace detail {

Status encode_esm2_sequence(
    const hikoboshi::universal::WeightsView& weights_view,
    const hikoboshi::modules::Esm2Descriptor& descriptor,
    hikoboshi::universal::Span<const std::int32_t> token_ids,
    float* embeddings_out) noexcept {
  if (embeddings_out == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "ESM2 encode output buffer must be non-null");
  }
  if (token_ids.data == nullptr || token_ids.size == 0) {
    return hikoboshi::universal::invalid_argument_status(
        "ESM2 encode token span must be non-empty");
  }
  try {
    Esm2PreparedState state;
    const Status status =
        build_esm2_weights(weights_view, descriptor, state);
    if (!hikoboshi::universal::is_ok(status)) {
      return status;
    }
    prepare_esm2_workspace(descriptor, token_ids.size, state);
    return encode_sequence_into(state, descriptor, token_ids, embeddings_out);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "ESM2 encode workspace allocation failed");
  }
}

}  // namespace detail

Status run_pairwise_structures(const PairwiseStructureRequest& request,
                               PairwiseWorkspace& workspace,
                               PairwiseResult& result) noexcept {
  if (request.weights == nullptr) {
    return hikoboshi::universal::invalid_argument_status("pairwise structure request requires MPNN weights");
  }
  if (!workspace.has_mpnn_workspaces()) {
    return hikoboshi::universal::failed_precondition_status(
        "pairwise structure request requires MPNN workspace allocation");
  }
  if (!workspace.can_align(request.query.residue_count,
                           request.target.residue_count,
                           request.descriptor.hidden_dimension)) {
    return hikoboshi::universal::failed_precondition_status("pairwise workspace has insufficient capacity");
  }

  hikoboshi::modules::Mpnn64ForwardRequest query_request{};
  query_request.coordinates = request.query.coordinates.data;
  query_request.atom_sources = request.query.atom_sources.data;
  query_request.residue_count = request.query.residue_count;
  query_request.descriptor = request.descriptor;
  query_request.weights = request.weights;
  query_request.workspace = workspace.query_mpnn_workspace();
  hikoboshi::modules::Mpnn64ForwardOutput query_output{};
  query_output.embeddings = workspace.query_embedding_data();
  query_output.residue_count = workspace.plan().max_query_length;
  query_output.hidden_dimension = request.descriptor.hidden_dimension;

  hikoboshi::modules::Mpnn64ForwardRequest target_request{};
  target_request.coordinates = request.target.coordinates.data;
  target_request.atom_sources = request.target.atom_sources.data;
  target_request.residue_count = request.target.residue_count;
  target_request.descriptor = request.descriptor;
  target_request.weights = request.weights;
  target_request.workspace = workspace.target_mpnn_workspace();
  hikoboshi::modules::Mpnn64ForwardOutput target_output{};
  target_output.embeddings = workspace.target_embedding_data();
  target_output.residue_count = workspace.plan().max_target_length;
  target_output.hidden_dimension = request.descriptor.hidden_dimension;

  // The two MPNN-64 encodes share no mutable state: workspaces, embedding
  // output buffers, atom-source spans, and weights views are all caller-owned
  // and disjoint. Dispatch the target encode to a stack-owned pthread job and
  // run the query encode on the caller thread, then join. Avoiding std::async
  // keeps the steady-state hot path free of heap-backed future allocations.
  Status query_status = hikoboshi::universal::ok_status();
  Status target_status = hikoboshi::universal::ok_status();
  bool ran_parallel = false;
  if (!parallel_structure_encode_disabled()) {
    Mpnn64TargetEncodeJob target_job{};
    target_job.request = target_request;
    target_job.output = target_output;
    pthread_t target_thread{};
    if (pthread_create(&target_thread, nullptr, &run_mpnn64_target_encode_job,
                       &target_job) == 0) {
      query_status =
          hikoboshi::modules::mpnn64_forward_scalar(query_request, query_output);
      if (pthread_join(target_thread, nullptr) == 0) {
        target_status = target_job.status;
        ran_parallel = true;
      } else {
        return hikoboshi::universal::internal_error_status(
            "pairwise target encode worker join failed");
      }
    }
  }
  if (!ran_parallel) {
    query_status =
        hikoboshi::modules::mpnn64_forward_scalar(query_request, query_output);
    target_status =
        hikoboshi::modules::mpnn64_forward_scalar(target_request, target_output);
  }
  if (!hikoboshi::universal::is_ok(query_status)) {
    return query_status;
  }
  if (!hikoboshi::universal::is_ok(target_status)) {
    return target_status;
  }

  PairwiseEmbeddingRequest embedding_request{};
  embedding_request.query_embedding =
      make_workspace_embedding(workspace.query_embedding_data(), request.query,
                               request.descriptor.hidden_dimension);
  embedding_request.target_embedding =
      make_workspace_embedding(workspace.target_embedding_data(), request.target,
                               request.descriptor.hidden_dimension);
  embedding_request.query_structure = request.query;
  embedding_request.target_structure = request.target;
  embedding_request.options = request.options;
  embedding_request.hard_mode = request.hard_mode;
  embedding_request.soft_mode = request.soft_mode;
  embedding_request.temperature = request.temperature;
  return score_embeddings_and_run_resolved(embedding_request, workspace, result);
}

Status run_pairwise_sequences(const PairwiseSequenceRequest& request,
                              PairwiseWorkspace& workspace,
                              PairwiseResult& result) noexcept {
  if (request.weights_view == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pairwise sequence request requires an ESM2 weights view");
  }
  if (request.query_token_ids.data == nullptr ||
      request.target_token_ids.data == nullptr) {
    return hikoboshi::universal::invalid_argument_status(
        "pairwise sequence request token spans must be non-null");
  }
  if (request.query_token_ids.size == 0 ||
      request.target_token_ids.size == 0) {
    return hikoboshi::universal::invalid_argument_status(
        "pairwise sequence request token spans must be non-empty");
  }
  if (!std::isfinite(request.options.gap_open) ||
      !std::isfinite(request.options.gap_extension)) {
    return hikoboshi::universal::invalid_argument_status(
        "pairwise gap parameters must be finite");
  }

  // Wrap each raw AA token span as `[<cls>, aa..., <eos>]` for the ESM2
  // encoder, then expose only the residue rows (1..L+1) to similarity and
  // pairwise. Matches the PyTorch training pipeline; see fe2 findings for
  // the embedding-drift measurement (~1e-1 .. 4.7e-1 per cell without the
  // wrap, ~1e-4 with it).
  constexpr std::int32_t kEsm2ClsTokenId = 26;
  constexpr std::int32_t kEsm2EosTokenId = 27;
  constexpr std::size_t kEsm2SpecialTokenOverhead = 2U;
  const std::size_t hidden = request.descriptor.hidden_dimension;
  const std::size_t wrapped_query_count =
      request.query_token_ids.size + kEsm2SpecialTokenOverhead;
  const std::size_t wrapped_target_count =
      request.target_token_ids.size + kEsm2SpecialTokenOverhead;
  if (!workspace.can_align(wrapped_query_count, wrapped_target_count,
                           hidden)) {
    return hikoboshi::universal::failed_precondition_status(
        "pairwise workspace has insufficient capacity for the sequence pair "
        "(including CLS/EOS overhead)");
  }

  std::vector<std::int32_t> wrapped_query;
  std::vector<std::int32_t> wrapped_target;
  try {
    wrapped_query.assign(wrapped_query_count, 0);
    wrapped_target.assign(wrapped_target_count, 0);
  } catch (const std::bad_alloc&) {
    return hikoboshi::universal::unavailable_status(
        "pairwise sequence wrapped-token buffer allocation failed");
  }
  wrapped_query[0] = kEsm2ClsTokenId;
  std::copy(request.query_token_ids.data,
            request.query_token_ids.data + request.query_token_ids.size,
            wrapped_query.begin() + 1);
  wrapped_query[wrapped_query_count - 1] = kEsm2EosTokenId;
  wrapped_target[0] = kEsm2ClsTokenId;
  std::copy(request.target_token_ids.data,
            request.target_token_ids.data + request.target_token_ids.size,
            wrapped_target.begin() + 1);
  wrapped_target[wrapped_target_count - 1] = kEsm2EosTokenId;

  Status status = detail::encode_esm2_sequence(
      *request.weights_view, request.descriptor,
      hikoboshi::universal::Span<const std::int32_t>{wrapped_query.data(),
                                                   wrapped_query_count},
      workspace.query_embedding_data());
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }
  status = detail::encode_esm2_sequence(
      *request.weights_view, request.descriptor,
      hikoboshi::universal::Span<const std::int32_t>{wrapped_target.data(),
                                                   wrapped_target_count},
      workspace.target_embedding_data());
  if (!hikoboshi::universal::is_ok(status)) {
    return status;
  }

  // Skip the leading CLS row and exclude the trailing EOS row by capping
  // the residue count at `request.query_token_ids.size`.
  hikoboshi::universal::EmbeddingView query_embedding{};
  query_embedding.residue_count = request.query_token_ids.size;
  query_embedding.dimension = hidden;
  query_embedding.values = {workspace.query_embedding_data() + hidden,
                            request.query_token_ids.size * hidden};
  hikoboshi::universal::EmbeddingView target_embedding{};
  target_embedding.residue_count = request.target_token_ids.size;
  target_embedding.dimension = hidden;
  target_embedding.values = {workspace.target_embedding_data() + hidden,
                             request.target_token_ids.size * hidden};

  PairwiseEmbeddingRequest embedding_request{};
  embedding_request.query_embedding = query_embedding;
  embedding_request.target_embedding = target_embedding;
  embedding_request.options = request.options;
  embedding_request.hard_mode = request.hard_mode;
  embedding_request.soft_mode = request.soft_mode;
  embedding_request.temperature = request.temperature;
  return score_embeddings_and_run_resolved(embedding_request, workspace,
                                           result);
}

}  // namespace hikoboshi::algorithms

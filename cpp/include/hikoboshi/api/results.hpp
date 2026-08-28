#ifndef HIKOBOSHI_API_RESULTS_HPP
#define HIKOBOSHI_API_RESULTS_HPP

/// @file
/// Public result records for encode, pairwise, and all-vs-all workflows.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <hikoboshi/api/requests.hpp>
#include <hikoboshi/universal/alignment_path.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::api {

/// Owning embedding result returned by encode requests.
///
/// `values` is row-major `[residue_count, dimension]`. Residue codes and
/// metadata are copied when available so the result can be used as a later
/// embedding input without keeping the original structure object alive.
struct EncodedEmbedding {
  std::size_t residue_count = 0;
  std::size_t dimension = 0;
  std::vector<float> values;
  std::vector<char> residue_codes;
  std::vector<universal::ResidueMetadataView> residues;
};

/// Result of an encode request.
struct EncodeResult {
  EncodedEmbedding embedding{};
};

/// One designed sequence emitted by inverse folding.
struct InverseFoldSequenceResult {
  std::string sequence;
  double score = 0.0;
  universal::MetricValue recovery_vs_native{
      0.0, false, universal::MetricInvalidReason::Unavailable};
  std::uint64_t seed = 0;
  InverseFoldDecodeOrder decode_order = InverseFoldDecodeOrder::Random;
};

/// Opt-in per-position log-prob artifact.
///
/// `values` is row-major `[num_seqs, residue_count, vocab_size]` float32 and
/// is populated only when `InverseFoldRequest::logprobs_out` is non-empty.
struct InverseFoldLogProbsArtifact {
  std::string path;
  std::size_t num_seqs = 0;
  std::size_t residue_count = 0;
  std::size_t vocab_size = 0;
  std::vector<float> values;
};

/// Result of an inverse-folding request.
struct InverseFoldResult {
  std::vector<InverseFoldSequenceResult> sequences;
  InverseFoldLogProbsArtifact logprobs{};
};

/// Public alignment step alias from the universal layer.
using AlignmentStep = universal::AlignmentStep;
/// Public alignment path alias from the universal layer.
using AlignmentPath = universal::AlignmentPath;

/// Metrics returned for one pairwise alignment.
///
/// `raw_sw_score` is the primary score for the selected mode: hard local
/// affine SW for hard/both, and the soft log-partition score for soft-only.
/// `soft_sw_score` is valid whenever a soft pass ran and invalid otherwise.
/// Optional metrics use `MetricValue` so unavailable structure/sequence
/// metadata is represented explicitly instead of being reported as zero.
struct PairwiseMetrics {
  double raw_sw_score = 0.0;
  universal::MetricValue soft_sw_score{
      0.0, false, universal::MetricInvalidReason::Unavailable};
  universal::MetricValue coverage_query{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_target{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_mean{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue identity{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue rmsd{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue tm_score_query{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue tm_score_target{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt_byA{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt_byB{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue lddt_aln{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_byA{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue coverage_byB{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
  universal::MetricValue ecs{
      0.0, false, universal::MetricInvalidReason::Unimplemented};
};

/// Result for one pairwise alignment workflow.
struct PairwiseResult {
  AlignmentPath path{};
  PairwiseMetrics metrics{};
  std::vector<universal::PackageWarning> warnings{};
};

/// One pair record emitted by all-vs-all enumeration.
struct PairwiseResultRecord {
  std::size_t query_index = 0;
  std::size_t target_index = 0;
  PairwiseResult result{};
};

/// Collected all-vs-all records.
struct AllVsAllResult {
  std::vector<PairwiseResultRecord> records;
};

}  // namespace hikoboshi::api

#endif  // HIKOBOSHI_API_RESULTS_HPP

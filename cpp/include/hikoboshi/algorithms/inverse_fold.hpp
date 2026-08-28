#ifndef HIKOBOSHI_ALGORITHMS_INVERSE_FOLD_HPP
#define HIKOBOSHI_ALGORITHMS_INVERSE_FOLD_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::algorithms {

enum class InverseFoldDecodeOrder {
  Random,
  NToC,
};

struct InverseFoldOptions {
  float sampling_temp = 0.1F;
  std::size_t num_seqs = 1;
  std::uint64_t seed = 0;
  InverseFoldDecodeOrder decode_order = InverseFoldDecodeOrder::Random;
  float backbone_noise = 0.0F;
  bool keep_log_probs = false;
};

struct InverseFoldSequence {
  std::string sequence;
  double score = 0.0;
  universal::MetricValue recovery_vs_native{
      0.0, false, universal::MetricInvalidReason::Unavailable};
  std::uint64_t seed = 0;
  InverseFoldDecodeOrder decode_order = InverseFoldDecodeOrder::Random;
};

struct InverseFoldLogProbs {
  std::size_t num_seqs = 0;
  std::size_t residue_count = 0;
  std::size_t vocab_size = 0;
  std::vector<float> values;
};

struct InverseFoldRequest {
  universal::StructureView structure{};
  const universal::detail::ProteinMpnnV48020Weights* weights = nullptr;
  InverseFoldOptions options{};
};

struct InverseFoldResult {
  std::vector<InverseFoldSequence> sequences;
  InverseFoldLogProbs logprobs{};
};

[[nodiscard]] universal::Result<InverseFoldResult>
inverse_fold_proteinmpnn_v48_020(const InverseFoldRequest& request);

}  // namespace hikoboshi::algorithms

#endif  // HIKOBOSHI_ALGORITHMS_INVERSE_FOLD_HPP

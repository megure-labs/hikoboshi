#include <hikoboshi/dispatch/registry/scoring.hpp>

#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::dispatch::registry {
namespace {

namespace hiko_u = hikoboshi::universal;

constexpr std::string_view kRawDotV1ScoringId{"raw_dot_v1"};
constexpr std::string_view kRawDotV1ScoringOpKind{"dot_product_similarity_v1"};

constexpr hiko_u::ScoreInputKind kRawDotV1Inputs[] = {
    hiko_u::ScoreInputKind::ResidueEmbeddings,
};

}  // namespace

universal::Span<const RegisteredScoringRecord> scoring_registry() noexcept {
  static const RegisteredScoringRecord kRecords[] = {
      {
          hiko_u::ScoreMethod::RawDotV1,
          kRawDotV1ScoringId,
          kRawDotV1ScoringOpKind,
          nullptr,
          {kRawDotV1Inputs, sizeof(kRawDotV1Inputs) / sizeof(kRawDotV1Inputs[0])},
          hiko_u::DataType::Float32,
      },
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

const RegisteredScoringRecord* find_scoring(
    const hiko_u::ScoreMethod kind) noexcept {
  const universal::Span<const RegisteredScoringRecord> records =
      scoring_registry();
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].kind == kind) {
      return &records.data[index];
    }
  }
  return nullptr;
}

}  // namespace hikoboshi::dispatch::registry

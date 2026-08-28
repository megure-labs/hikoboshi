#ifndef HIKOBOSHI_DISPATCH_REGISTRY_SCORING_HPP
#define HIKOBOSHI_DISPATCH_REGISTRY_SCORING_HPP

/// @file
/// Closed registry of Hikoboshi scoring methods.
///
/// One record per `ScoreMethod`. The record binds the scoring id, the scoring
/// op tag it composes, a builder hook, the score input kinds it accepts, and
/// the output dtype it emits. Hikoboshi 0.1.0 registers `raw_dot_v1`.

#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::dispatch::registry {

/// Builder hook for a registered scoring method.
///
/// Reserved for later packets that lower scoring through dispatch tables;
/// Hikoboshi 0.1.0 uses a null builder because raw-dot scoring is composed
/// directly in the modules layer.
using ScoringBuilderFn = void (*)(const universal::PackageHandle& package);

/// One row in the scoring registry.
///
/// `inputs` references static storage owned by the dispatch registry TU.
struct RegisteredScoringRecord {
  universal::ScoreMethod kind;
  std::string_view scoring_id;
  std::string_view scoring_op_kind;
  ScoringBuilderFn builder;
  universal::Span<const universal::ScoreInputKind> inputs;
  universal::DataType output_dtype;
};

/// Return the closed set of scoring records.
///
/// Hikoboshi 0.1.0 reports exactly one record (`raw_dot_v1`).
universal::Span<const RegisteredScoringRecord> scoring_registry() noexcept;

/// Resolve a registered scoring method by enum tag.
///
/// Returns `nullptr` if no record matches.
const RegisteredScoringRecord* find_scoring(
    universal::ScoreMethod kind) noexcept;

}  // namespace hikoboshi::dispatch::registry

#endif  // HIKOBOSHI_DISPATCH_REGISTRY_SCORING_HPP

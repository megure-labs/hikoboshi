#ifndef HIKOBOSHI_DISPATCH_REGISTRY_ALIGNMENT_HPP
#define HIKOBOSHI_DISPATCH_REGISTRY_ALIGNMENT_HPP

/// @file
/// Closed registry of Hikoboshi alignment algorithms.
///
/// One record per `AlignmentAlgorithmId`. The record binds the alignment id,
/// the primitive op tags the algorithm depends on, and the gap-model families
/// it supports. Hikoboshi 0.1.0 registers `hard_local_affine_sw_v1`.

#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::dispatch::registry {

/// One row in the alignment registry.
///
/// `primitive_op_kinds` and `gap_families_supported` reference static storage
/// owned by the dispatch registry TU.
struct RegisteredAlignmentRecord {
  universal::AlignmentAlgorithmId kind;
  std::string_view alignment_id;
  universal::Span<const std::string_view> primitive_op_kinds;
  universal::Span<const universal::GapModel> gap_families_supported;
};

/// Return the closed set of alignment records.
///
/// Hikoboshi 0.1.0 reports exactly one record (`hard_local_affine_sw_v1`).
universal::Span<const RegisteredAlignmentRecord> alignment_registry() noexcept;

/// Resolve a registered alignment algorithm by enum tag.
///
/// Returns `nullptr` if no record matches.
const RegisteredAlignmentRecord* find_alignment(
    universal::AlignmentAlgorithmId kind) noexcept;

}  // namespace hikoboshi::dispatch::registry

#endif  // HIKOBOSHI_DISPATCH_REGISTRY_ALIGNMENT_HPP

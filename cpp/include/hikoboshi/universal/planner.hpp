#ifndef HIKOBOSHI_UNIVERSAL_PLANNER_HPP
#define HIKOBOSHI_UNIVERSAL_PLANNER_HPP

/// @file
/// Public planner policy records for backend selection.

#include <cstdint>

#include <hikoboshi/universal/backend.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::universal {

/// Planner policy families understood by the public configuration surface.
enum class PlannerPolicyKind : std::uint8_t {
  ScalarOnly = 0,
};

/// Reserved execution fusion rule tag.
enum class PlannerFusionRule : std::uint8_t {
  Disabled = 0,
};

/// Low-level backend planning policy used by EngineConfig.
///
/// Hikoboshi 0.1.0 ships a scalar-only policy. The structure is intentionally
/// explicit so future backend-capable builds can report fallback order and
/// planner behavior without changing the EngineConfig shape.
struct PlannerPolicy {
  std::uint32_t policy_version;
  PlannerPolicyKind kind;
  Span<const Backend> fallback_order;
  Backend default_backend;
  PlannerFusionRule fusion_rule;
  std::uint64_t reserved_flags;
};

inline constexpr std::uint32_t kScalarPlannerPolicyVersion = 1;
inline constexpr Backend kScalarPlannerFallbackOrder[] = {Backend::Scalar};
inline constexpr PlannerPolicy kScalarDefaultPlannerPolicy{
    kScalarPlannerPolicyVersion,
    PlannerPolicyKind::ScalarOnly,
    {kScalarPlannerFallbackOrder, 1},
    Backend::Scalar,
    PlannerFusionRule::Disabled,
    0,
};

/// Return the supplied policy or the scalar-only 0.1.0 default.
inline constexpr const PlannerPolicy& scalar_default_planner_policy(
    const PlannerPolicy* policy) noexcept {
  return policy == nullptr ? kScalarDefaultPlannerPolicy : *policy;
}

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_PLANNER_HPP

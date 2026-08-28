from __future__ import annotations

import math

import hikoboshi as hiko

from helpers import matrix, require

MPNN64_SOFT_GAP_OPEN = -3.21337
MPNN64_SOFT_GAP_EXTENSION = -0.111704
PUBLIC_PAIRWISE_METRIC_FIELDS = {
    "raw_sw_score",
    "soft_sw_score",
    "sw_per_query_len",
    "sw_per_target_len",
    "sw_per_aligned",
    "coverage_query",
    "coverage_target",
    "coverage_mean",
    "identity",
    "rmsd",
    "tm_score_query",
    "tm_score_target",
    "lddt",
    "lddt_byA",
    "lddt_byB",
    "lddt_aln",
    "coverage_byA",
    "coverage_byB",
    "ecs",
    "invalid_reasons",
}


def require_soft_score_match(actual, expected, message: str) -> None:
    require(actual.metrics.soft_sw_score is not None, message)
    require(expected.metrics.soft_sw_score is not None, message)
    require(
        math.isclose(
            actual.metrics.soft_sw_score,
            expected.metrics.soft_sw_score,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        message,
    )


identity = matrix(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)

soft_result = hiko.pairwise.from_embeddings(
    identity,
    identity,
    mode="soft",
    temperature=1.0,
)
require(
    set(vars(soft_result.metrics)) == PUBLIC_PAIRWISE_METRIC_FIELDS,
    "soft pairwise must expose only the public score and ordinary metric schema",
)
require(
    soft_result.path.aligned_pairs > 0,
    "soft pairwise must populate the consensus alignment",
)
require(
    math.isfinite(soft_result.metrics.raw_sw_score),
    "soft pairwise raw_sw_score must be finite",
)
require(
    soft_result.metrics.soft_sw_score == soft_result.metrics.raw_sw_score,
    "soft pairwise raw_sw_score must mirror soft_sw_score",
)

gap_sensitive_query = identity
gap_sensitive_target = matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
default_soft_gap_result = hiko.pairwise.from_embeddings(
    gap_sensitive_query,
    gap_sensitive_target,
    mode="soft",
)
explicit_soft_gap_result = hiko.pairwise.from_embeddings(
    gap_sensitive_query,
    gap_sensitive_target,
    mode="soft",
    gap_open=MPNN64_SOFT_GAP_OPEN,
    gap_extension=MPNN64_SOFT_GAP_EXTENSION,
)
require(
    math.isclose(
        default_soft_gap_result.metrics.raw_sw_score,
        explicit_soft_gap_result.metrics.raw_sw_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ),
    "default MPNN soft pairwise score must use calibrated soft gaps",
)

soft_result_again = hiko.pairwise.from_embeddings(identity, identity, mode="soft")
require(
    soft_result_again.path.aligned_pairs == soft_result.path.aligned_pairs,
    "repeat soft pairwise must preserve the consensus alignment size",
)
require(
    math.isclose(
        soft_result_again.metrics.raw_sw_score,
        soft_result.metrics.raw_sw_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ),
    "repeat soft pairwise must preserve the soft score",
)

hard_result_a = hiko.pairwise.from_embeddings(identity, identity, mode="hard")
hard_result_b = hiko.pairwise.from_embeddings(
    identity,
    identity,
    mode="hard",
    temperature=42.0,
)
require(
    hard_result_a.metrics.raw_sw_score == hard_result_b.metrics.raw_sw_score,
    "hard mode must ignore temperature",
)
require(
    hard_result_a.path.aligned_pairs == hard_result_b.path.aligned_pairs,
    "hard mode path must ignore temperature",
)
require(
    hard_result_a.metrics.soft_sw_score is None,
    "hard mode must not populate soft_sw_score",
)

both_result = hiko.pairwise.from_embeddings(identity, identity, mode="both")
require(
    both_result.metrics.raw_sw_score == hard_result_a.metrics.raw_sw_score,
    "both pairwise raw_sw_score must keep the hard score",
)
require(
    both_result.path.aligned_pairs == hard_result_a.path.aligned_pairs,
    "both pairwise path must keep the hard path",
)
require(
    both_result.metrics.soft_sw_score is not None
    and math.isfinite(both_result.metrics.soft_sw_score),
    "both pairwise soft_sw_score must be finite",
)
require(
    math.isclose(
        both_result.metrics.soft_sw_score,
        soft_result.metrics.raw_sw_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ),
    "both pairwise soft_sw_score must match explicit soft raw score",
)
require(
    math.isclose(both_result.metrics.sw_per_query_len, 1.0, abs_tol=1e-6),
    "both pairwise sw_per_query_len mismatch",
)
require(
    math.isclose(both_result.metrics.sw_per_target_len, 1.0, abs_tol=1e-6),
    "both pairwise sw_per_target_len mismatch",
)
require(
    math.isclose(both_result.metrics.sw_per_aligned, 1.0, abs_tol=1e-6),
    "both pairwise sw_per_aligned mismatch",
)

hard_gap_result = hiko.pairwise.from_embeddings(
    gap_sensitive_query,
    gap_sensitive_target,
    mode="hard",
)
both_gap_result = hiko.pairwise.from_embeddings(
    gap_sensitive_query,
    gap_sensitive_target,
    mode="both",
)
require(
    both_gap_result.metrics.raw_sw_score == hard_gap_result.metrics.raw_sw_score,
    "both pairwise gap-sensitive raw score must keep the hard score",
)
require(
    both_gap_result.path.aligned_pairs == hard_gap_result.path.aligned_pairs,
    "both pairwise gap-sensitive path must keep the hard path",
)
require_soft_score_match(
    both_gap_result,
    explicit_soft_gap_result,
    "both pairwise soft score must use calibrated soft gaps",
)

try:
    hiko.pairwise.from_embeddings(
        matrix([[1.0, 0.0]]),
        matrix([[1.0, 0.0]]),
        mode="bogus",
    )
except hiko.InvalidArgumentError as exc:
    require(exc.code == "invalid_argument", "invalid mode error code mismatch")
    require(
        "'hard', 'soft', or 'both'" in exc.detail,
        "invalid mode detail should explain accepted values",
    )
else:
    raise SystemExit("invalid mode kwarg should raise InvalidArgumentError")

try:
    hiko.pairwise.from_embeddings(
        matrix([[1.0, 0.0]]),
        matrix([[1.0, 0.0]]),
        mode="soft",
        temperature=0.0,
    )
except hiko.InvalidArgumentError as exc:
    require(
        exc.code == "invalid_argument",
        "zero-temperature error code mismatch",
    )
    require("temperature" in exc.detail, "error detail should mention temperature")
else:
    raise SystemExit("soft mode with temperature=0 should raise")

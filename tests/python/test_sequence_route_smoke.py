#!/usr/bin/env python3
"""Sequence-route Python smoke tests for hikoboshi-esm2-8m.

Drives `encode_from_sequence` and `pairwise_from_sequence` through the
low-level `hikoboshi._core` extension. The python-side wrapper classes
(`hikoboshi.encode`, `hikoboshi.pairwise`) are intentionally not extended in
this packet; the lower-level core surface is exercised here so the
adapter packets land on a known-good binding shape.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ESM2_HARD_GAP_OPEN = -1.01982
ESM2_HARD_GAP_EXTENSION = 0.225736
ESM2_SOFT_GAP_OPEN = -6.72805
ESM2_SOFT_GAP_EXTENSION = -0.0159468


def _setup_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    python_root = repo_root / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    build_root = os.environ.get("HIKOBOSHI_BUILD_ROOT")
    if build_root:
        os.environ.setdefault("HIKOBOSHI_BUILD_ROOT", build_root)


def _load_core():
    _setup_path()
    from hikoboshi._arrays import load_core  # type: ignore

    return load_core()


def fail(message: str) -> None:
    raise SystemExit(f"sequence_route_smoke: {message}")


def _to_floats(values) -> list[float]:
    if hasattr(values, "shape"):
        flat = bytes(values.buffer())  # type: ignore[attr-defined]
        import struct

        count = len(flat) // 4
        return list(struct.unpack(f"<{count}f", flat))
    return list(values)


def require_soft_metrics_match(actual: dict, expected: dict, message: str) -> None:
    for key in ("soft_sw_score",):
        actual_metric = actual.get(key)
        expected_metric = expected.get(key)
        if not (
            isinstance(actual_metric, dict)
            and isinstance(expected_metric, dict)
            and actual_metric.get("valid") is True
            and expected_metric.get("valid") is True
        ):
            fail(message)
        if not math.isclose(
            actual_metric["value"], expected_metric["value"], rel_tol=0.0, abs_tol=1e-6
        ):
            fail(message)


def test_encode_returns_shape() -> None:
    core = _load_core()
    result = core.encode_from_sequence("AAAAAA", package="esm2-8m")
    metadata = result["metadata"]
    if metadata["residue_count"] != 6:
        fail(f"encode residue_count={metadata['residue_count']}, want 6")
    if metadata["dimension"] != 320:
        fail(f"encode dimension={metadata['dimension']}, want 320")
    tensor = result["embeddings"]
    if getattr(tensor, "shape", None) != (6, 320):
        fail(f"encode tensor shape={getattr(tensor, 'shape', None)}")


def test_pairwise_self_self() -> None:
    core = _load_core()
    result = core.pairwise_from_sequence(
        query="ACDEFG", target="ACDEFG", package="esm2-8m", mode="hard"
    )
    path = result["path"]
    if path["aligned_pairs"] != 6:
        fail(f"self-self aligned_pairs={path['aligned_pairs']}, want 6")
    metrics = result["metrics"]
    if metrics["raw_sw_score"] <= 0.0:
        fail(
            f"self-self raw_sw_score={metrics['raw_sw_score']}, expected >0"
        )


def test_pairwise_cross() -> None:
    core = _load_core()
    result = core.pairwise_from_sequence(
        query="ACDEFG", target="GFEDCA", package="esm2-8m", mode="hard"
    )
    # The reverse-residue pair must still execute without error; the
    # specific score depends on the encoder activations.
    metrics = result["metrics"]
    if "raw_sw_score" not in metrics:
        fail("pairwise result missing raw_sw_score metric")


def test_pairwise_hard_defaults_use_esm2_annealed_gaps() -> None:
    core = _load_core()
    default_result = core.pairwise_from_sequence(
        query="ACDEFG", target="ACDE", package="esm2-8m", mode="hard"
    )
    explicit_result = core.pairwise_from_sequence(
        query="ACDEFG",
        target="ACDE",
        package="esm2-8m",
        mode="hard",
        gap_open=ESM2_HARD_GAP_OPEN,
        gap_extension=ESM2_HARD_GAP_EXTENSION,
    )
    if not math.isclose(
        default_result["metrics"]["raw_sw_score"],
        explicit_result["metrics"]["raw_sw_score"],
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        fail("default ESM2 hard score must use the annealed hard gaps")
    if default_result["path"]["aligned_pairs"] != explicit_result["path"]["aligned_pairs"]:
        fail("default ESM2 hard path must use the annealed hard gaps")


def test_pairwise_soft_defaults_use_esm2_soft_gaps() -> None:
    core = _load_core()
    default_result = core.pairwise_from_sequence(
        query="ACDEFG", target="ACDE", package="esm2-8m", mode="soft"
    )
    explicit_result = core.pairwise_from_sequence(
        query="ACDEFG",
        target="ACDE",
        package="esm2-8m",
        mode="soft",
        gap_open=ESM2_SOFT_GAP_OPEN,
        gap_extension=ESM2_SOFT_GAP_EXTENSION,
    )
    default_metrics = default_result["metrics"]
    explicit_metrics = explicit_result["metrics"]
    if not math.isclose(default_metrics["raw_sw_score"],
                        explicit_metrics["raw_sw_score"],
                        rel_tol=0.0, abs_tol=1e-6):
        fail("default ESM2 soft raw score must use recovered soft gaps")
    require_soft_metrics_match(
        default_metrics,
        explicit_metrics,
        "default ESM2 soft metrics must use recovered soft gaps",
    )


def test_pairwise_both_uses_hard_primary_and_esm2_soft_gaps() -> None:
    core = _load_core()
    hard_result = core.pairwise_from_sequence(
        query="ACDEFG", target="ACDE", package="esm2-8m", mode="hard"
    )
    explicit_soft_result = core.pairwise_from_sequence(
        query="ACDEFG",
        target="ACDE",
        package="esm2-8m",
        mode="soft",
        gap_open=ESM2_SOFT_GAP_OPEN,
        gap_extension=ESM2_SOFT_GAP_EXTENSION,
    )
    both_result = core.pairwise_from_sequence(
        query="ACDEFG", target="ACDE", package="esm2-8m", mode="both"
    )
    hard_metrics = hard_result["metrics"]
    both_metrics = both_result["metrics"]
    if not math.isclose(both_metrics["raw_sw_score"],
                        hard_metrics["raw_sw_score"],
                        rel_tol=0.0, abs_tol=1e-6):
        fail("both ESM2 pairwise raw score must keep the hard score")
    if both_result["path"]["aligned_pairs"] != hard_result["path"]["aligned_pairs"]:
        fail("both ESM2 pairwise path must keep the hard path")
    require_soft_metrics_match(
        both_metrics,
        explicit_soft_result["metrics"],
        "both ESM2 pairwise soft fields must match explicit recovered soft gaps",
    )


def main() -> int:
    test_encode_returns_shape()
    test_pairwise_self_self()
    test_pairwise_cross()
    test_pairwise_hard_defaults_use_esm2_annealed_gaps()
    test_pairwise_soft_defaults_use_esm2_soft_gaps()
    test_pairwise_both_uses_hard_primary_and_esm2_soft_gaps()
    print("sequence_route_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

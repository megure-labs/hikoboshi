#!/usr/bin/env python3
"""Baked ProteinMPNN v_48_020 recovery parity gate.

This test intentionally has no torch or numpy dependency. It executes the
Meson-built native helper, reads committed JSON goldens, and fails on any
teacher-forced log-prob or argmax mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("HIKOBOSHI_SOURCE_ROOT", SCRIPT_DIR.parents[2]))
DEFAULT_TOLERANCE = REPO_ROOT / "bench" / "numerical_tolerance.json"
DEFAULT_FIXTURE_DIR = SCRIPT_DIR / "fixtures"
DEFAULT_ABS_TOLERANCE = 1.0e-4


def fail(message: str) -> None:
    raise SystemExit(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tolerance(path: Path) -> float:
    if not path.is_file():
        return DEFAULT_ABS_TOLERANCE
    payload = load_json(path)
    value = payload.get("proteinmpnn_logprob_abs", DEFAULT_ABS_TOLERANCE)
    if not isinstance(value, int | float):
        fail(f"{path}: proteinmpnn_logprob_abs must be numeric")
    return min(float(value), DEFAULT_ABS_TOLERANCE)


def as_float_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list):
        fail(f"{field} must be a list")
    out: list[float] = []
    for index, item in enumerate(value):
        if not isinstance(item, int | float):
            fail(f"{field}[{index}] must be numeric")
        out.append(float(item))
    return out


def run_native(driver: Path, pdb: Path, seed: int, sampling_temp: float) -> dict[str, Any]:
    completed = subprocess.run(
        [str(driver), str(pdb), str(seed), f"{sampling_temp:.9g}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        fail(f"native recovery parity driver failed for {pdb}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        fail(f"native driver emitted invalid JSON for {pdb}: {exc}")
    if not isinstance(payload, dict):
        fail(f"native driver output for {pdb} must be a JSON object")
    return payload


def compare_log_probs(
    fixture_id: str,
    native: list[float],
    expected: list[float],
    tolerance: float,
) -> float:
    if len(native) != len(expected):
        fail(
            f"{fixture_id}: log_probs length mismatch "
            f"{len(native)} != {len(expected)}"
        )
    max_abs = 0.0
    max_index = 0
    for index, (actual, wanted) in enumerate(zip(native, expected, strict=True)):
        diff = abs(actual - wanted)
        if diff > max_abs:
            max_abs = diff
            max_index = index
    if max_abs > tolerance:
        fail(
            f"{fixture_id}: teacher-forced log_probs max_abs={max_abs:.9g} "
            f"at flat index {max_index}, tolerance={tolerance:.9g}"
        )
    return max_abs


def sequence_agreement(lhs: str, rhs: str) -> float:
    if len(lhs) != len(rhs) or not lhs:
        return 0.0
    matches = sum(1 for left, right in zip(lhs, rhs, strict=True) if left == right)
    return matches / len(lhs)


def check_fixture(driver: Path, golden_path: Path, tolerance: float) -> tuple[float, str]:
    golden = load_json(golden_path)
    fixture_id = str(golden.get("fixture_id", golden_path.stem))
    pdb_rel = golden.get("pdb")
    if not isinstance(pdb_rel, str) or not pdb_rel:
        fail(f"{golden_path}: pdb must be a non-empty relative path")
    pdb = golden_path.parent / pdb_rel
    seed = golden.get("seed")
    sampling_temp = golden.get("sampling_temp")
    if not isinstance(seed, int):
        fail(f"{golden_path}: seed must be an integer")
    if not isinstance(sampling_temp, int | float):
        fail(f"{golden_path}: sampling_temp must be numeric")

    native = run_native(driver, pdb, seed, float(sampling_temp))
    expected_count = golden.get("residue_count")
    if native.get("residue_count") != expected_count:
        fail(
            f"{fixture_id}: residue_count mismatch "
            f"{native.get('residue_count')} != {expected_count}"
        )
    if native.get("native_sequence") != golden.get("native_sequence"):
        fail(f"{fixture_id}: native sequence mismatch")

    max_abs = compare_log_probs(
        fixture_id,
        as_float_list(native.get("teacher_log_probs"), "native teacher_log_probs"),
        as_float_list(golden.get("teacher_log_probs"), "golden teacher_log_probs"),
        tolerance,
    )

    native_argmax = native.get("teacher_argmax_sequence")
    expected_argmax = golden.get("teacher_argmax_sequence")
    if not isinstance(native_argmax, str) or not isinstance(expected_argmax, str):
        fail(f"{fixture_id}: teacher_argmax_sequence fields must be strings")
    agreement = sequence_agreement(native_argmax, expected_argmax)
    minimum = float(golden.get("teacher_argmax_agreement_min", 1.0))
    if agreement < minimum:
        fail(
            f"{fixture_id}: teacher argmax agreement {agreement:.6f} "
            f"< threshold {minimum:.6f}"
        )
    if minimum >= 1.0 and native_argmax != expected_argmax:
        fail(f"{fixture_id}: teacher argmax sequence mismatch")

    reference_recovery = float(golden.get("teacher_argmax_recovery", 0.0))
    native_recovery = float(native.get("teacher_argmax_recovery", -1.0))
    if abs(native_recovery - reference_recovery) > 1.0e-9:
        fail(
            f"{fixture_id}: teacher argmax recovery {native_recovery:.9g} "
            f"!= baked reference {reference_recovery:.9g}"
        )

    if bool(golden.get("require_exact_sample_sequence", False)):
        if native.get("sample_sequence") != golden.get("sample_sequence"):
            fail(f"{fixture_id}: sampled sequence mismatch")

    greedy_min = float(golden.get("greedy_recovery_min", 0.0))
    greedy_recovery = float(native.get("greedy_recovery", -1.0))
    if greedy_recovery < greedy_min:
        fail(
            f"{fixture_id}: greedy recovery {greedy_recovery:.6f} "
            f"< threshold {greedy_min:.6f}"
        )

    return max_abs, fixture_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--tolerance", type=Path, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    if not args.driver.is_file():
        fail(f"missing native parity driver: {args.driver}")
    if not args.fixtures.is_dir():
        fail(f"missing fixture directory: {args.fixtures}")

    goldens = sorted(args.fixtures.glob("*.golden.json"))
    if not goldens:
        fail(f"no baked goldens found under {args.fixtures}")

    tolerance = load_tolerance(args.tolerance)
    worst = 0.0
    checked: list[str] = []
    for golden_path in goldens:
        max_abs, fixture_id = check_fixture(args.driver, golden_path, tolerance)
        worst = max(worst, max_abs)
        checked.append(fixture_id)

    print(
        "proteinmpnn_v48_020_recovery_parity: OK "
        f"fixtures={len(checked)} max_logprob_abs={worst:.9g} "
        f"tolerance={tolerance:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

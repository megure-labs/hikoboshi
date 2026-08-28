from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import hikoboshi as hiko

from helpers import require, skip


DEFAULT_T6_STRUCTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "demo/all-vs-all/pdbs"
)
T6_STRUCTURE_ROOT = Path(
    os.environ.get("HIKOBOSHI_GIL_PROBE_STRUCTURE_ROOT", DEFAULT_T6_STRUCTURE_ROOT)
)
THREAD_COUNT = int(os.environ.get("HIKOBOSHI_GIL_PROBE_THREAD_COUNT", "4"))
HELD_GIL_BASELINE_CONCURRENT_OVER_SINGLE_T4 = 2.042929
MAX_CONCURRENT_OVER_SINGLE_T4 = float(
    os.environ.get("HIKOBOSHI_GIL_PROBE_MAX_CONCURRENT_OVER_SINGLE_T4", "1.85")
)

T6_INPUTS = tuple(("", name) for name in (
    "1HBS.pdb", "1IGY.pdb", "1MBA.pdb", "1MBO.pdb", "1MYT.pdb",
    "1REX.pdb", "1RNH.pdb", "2LYZ.pdb", "2NRL.pdb",
))


def t6_structure_inputs() -> list[str]:
    paths = [T6_STRUCTURE_ROOT / group / name for group, name in T6_INPUTS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        skip("missing t6 structure fixture inputs: " + ", ".join(missing))
    return [str(path) for path in paths]


def result_signature(result: hiko.AllVsAllResult) -> tuple[Any, ...]:
    return tuple(
        (
            record.query_index,
            record.target_index,
            record.result.metrics.raw_sw_score,
            record.result.path.aligned_pairs,
            tuple(
                (
                    step.query_index,
                    step.target_index,
                    step.residue_score,
                )
                for step in record.result.path.steps
            ),
        )
        for record in result.records
    )


def timed_call(inputs: list[str]) -> tuple[float, hiko.AllVsAllResult]:
    started = time.perf_counter()
    result = hiko.all_vs_all.from_structure(inputs, thread_count=THREAD_COUNT)
    return time.perf_counter() - started, result


def timed_concurrent_calls(
    inputs: list[str],
) -> tuple[float, list[tuple[float, hiko.AllVsAllResult]]]:
    barrier = threading.Barrier(3)
    results: list[tuple[float, hiko.AllVsAllResult] | None] = [None, None]
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait()
            results[index] = timed_call(inputs)
        except BaseException as exc:  # noqa: BLE001 - preserve worker failure.
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    started = time.perf_counter()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    if errors:
        raise errors[0]

    complete = [result for result in results if result is not None]
    require(len(complete) == 2, "concurrent GIL probe did not collect two results")
    return elapsed, complete


inputs = t6_structure_inputs()
warmup_elapsed, warmup_result = timed_call(inputs)
single_elapsed, single_result = timed_call(inputs)
single_signature = result_signature(single_result)
require(
    result_signature(warmup_result) == single_signature,
    "warmup and single-call all-vs-all results differ",
)

concurrent_elapsed, concurrent_results = timed_concurrent_calls(inputs)
for _, result in concurrent_results:
    require(
        result_signature(result) == single_signature,
        "concurrent all-vs-all result changed",
    )

concurrent_over_single_t4 = concurrent_elapsed / single_elapsed
payload = {
    "concurrent_over_single_t4": concurrent_over_single_t4,
    "concurrent_wall_seconds": concurrent_elapsed,
    "held_gil_baseline_concurrent_over_single_t4": (
        HELD_GIL_BASELINE_CONCURRENT_OVER_SINGLE_T4
    ),
    "input_count": len(inputs),
    "max_concurrent_over_single_t4": MAX_CONCURRENT_OVER_SINGLE_T4,
    "single_wall_seconds": single_elapsed,
    "thread_count": THREAD_COUNT,
    "warmup_wall_seconds": warmup_elapsed,
    "worker_wall_seconds": [elapsed for elapsed, _ in concurrent_results],
}
print("gil_probe: " + json.dumps(payload, sort_keys=True))
require(
    concurrent_over_single_t4 < MAX_CONCURRENT_OVER_SINGLE_T4,
    "concurrent all-vs-all calls still look serialized by the GIL",
)

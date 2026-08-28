from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import hikoboshi as hiko
from hikoboshi._arrays import load_core

from helpers import require, scalar_matrix, write_simple_pdb


def signature(result: hiko.AllVsAllResult) -> tuple[Any, ...]:
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


def require_invalid_argument(callable_object: Any, detail: str) -> None:
    try:
        callable_object()
    except hiko.InvalidArgumentError as exc:
        require(exc.code == "invalid_argument", "thread_count exception code mismatch")
        require(detail in exc.detail, "thread_count diagnostic mismatch")
    else:
        raise SystemExit("invalid thread_count should raise InvalidArgumentError")


inputs = [scalar_matrix(float(index + 1)) for index in range(10)]

serial = hiko.all_vs_all.from_embeddings(inputs, thread_count=1)
auto = hiko.all_vs_all.from_embeddings(inputs, thread_count=0)
threaded = hiko.all_vs_all.from_embeddings(inputs, thread_count=4)
require(signature(auto) == signature(serial), "auto thread output changed from serial")
require(
    signature(threaded) == signature(serial),
    "threaded embedding output changed from serial",
)

for thread_count in (0, 1, 4):
    require(
        len(hiko.all_vs_all(inputs[:2], thread_count=thread_count).records) == 1,
        f"callable all_vs_all rejected thread_count={thread_count}",
    )
    require(
        len(hiko.all_vs_all.from_embeddings(inputs[:2], thread_count=thread_count).records)
        == 1,
        f"embedding all_vs_all rejected thread_count={thread_count}",
    )

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    paths = [root / "a.pdb", root / "b.pdb"]
    for path in paths:
        write_simple_pdb(path, residue_count=3)
    string_paths = [str(path) for path in paths]
    for route in (
        hiko.all_vs_all.from_structure,
        hiko.all_vs_all.from_pdb,
        hiko.all_vs_all.from_cif,
        hiko.all_vs_all.from_coords,
    ):
        for thread_count in (0, 1, 4):
            result = route(string_paths, thread_count=thread_count)
            require(
                [(record.query_index, record.target_index) for record in result.records]
                == [(0, 1)],
                f"{route!r} rejected thread_count={thread_count}",
            )

require_invalid_argument(
    lambda: hiko.all_vs_all.from_embeddings(inputs[:2], thread_count=-1),
    "thread_count must be a non-negative integer",
)
require_invalid_argument(
    lambda: hiko.all_vs_all.from_embeddings(inputs[:2], thread_count="4"),
    "thread_count must be a non-negative integer",
)

core = load_core()
for value in (-1, "4"):
    try:
        core.all_vs_all_from_embeddings(inputs[:2], thread_count=value)
    except Exception as exc:  # noqa: BLE001 - extension exception type.
        require(
            getattr(exc, "code", "") == "invalid_argument",
            "core thread_count exception code mismatch",
        )
        require(
            "thread_count" in getattr(exc, "detail", str(exc)),
            "core thread_count diagnostic mismatch",
        )
    else:
        raise SystemExit("core invalid thread_count should fail")

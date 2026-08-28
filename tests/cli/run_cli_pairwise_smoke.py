#!/usr/bin/env python3
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path


def write_npy(path: Path, rows: list[list[float]]) -> None:
    row_count = len(rows)
    col_count = len(rows[0])
    header = (
        "{'descr': '<f4', 'fortran_order': False, "
        f"'shape': ({row_count}, {col_count}), }}"
    )
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header_bytes = (header + (" " * padding) + "\n").encode("ascii")
    values = [value for row in rows for value in row]
    with path.open("wb") as out:
        out.write(b"\x93NUMPY")
        out.write(bytes([1, 0]))
        out.write(struct.pack("<H", len(header_bytes)))
        out.write(header_bytes)
        out.write(struct.pack("<" + "f" * len(values), *values))


def require_line(text: str, expected: str) -> None:
    if expected not in text.splitlines():
        raise SystemExit(f"missing line {expected!r}\n{text}")


def require_metric_line(text: str, key: str) -> None:
    for line in text.splitlines():
        cells = line.split("\t")
        if len(cells) >= 2 and cells[0] == key:
            if cells[1] == "NA":
                raise SystemExit(f"{key} was not populated\n{text}")
            return
    raise SystemExit(f"missing metric line {key!r}\n{text}")


def require_no_metric_line(text: str, key: str) -> None:
    for line in text.splitlines():
        if line.split("\t", 1)[0] == key:
            raise SystemExit(f"unexpected metric line {key!r}\n{text}")


def run_pairwise(
    binary: Path,
    query: Path,
    target: Path,
    summary: Path,
    *,
    mode: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(binary),
        "pairwise",
        "embeddings",
        str(query),
        str(target),
        "--summary",
        str(summary),
    ]
    if mode is not None:
        command.extend(["--mode", mode])
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)
    return result


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        query = root / "query.npy"
        target = root / "target.npy"
        rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        write_npy(query, rows)
        write_npy(target, rows)

        hard_summary = root / "pairwise.tsv"
        hard = run_pairwise(binary, query, target, hard_summary)
        require_line(hard.stdout, "command\tpairwise")
        require_line(hard.stdout, "input_mode\tembeddings")
        require_line(hard.stdout, "raw_sw_score\t3")
        require_line(hard.stdout, "aligned_pairs\t3")
        for key in (
            "soft_sw_score",
            "sw_per_query_len",
            "sw_per_target_len",
            "sw_per_aligned",
        ):
            require_no_metric_line(hard.stdout, key)
        if hard_summary.read_text() != hard.stdout:
            raise SystemExit("pairwise summary file must match stdout summary")

        soft_summary = root / "pairwise_soft.tsv"
        soft = run_pairwise(
            binary,
            query,
            target,
            soft_summary,
            mode="soft",
        )
        for key in (
            "soft_sw_score",
            "sw_per_query_len",
            "sw_per_target_len",
            "sw_per_aligned",
        ):
            require_metric_line(soft.stdout, key)
        if soft_summary.read_text() != soft.stdout:
            raise SystemExit("soft pairwise summary file must match stdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

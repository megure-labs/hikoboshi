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


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        query = root / "query.npy"
        target = root / "target.npy"
        write_npy(query, [[1.0, 0.0], [0.0, 1.0]])
        write_npy(target, [[1.0, 0.0], [0.0, 1.0]])

        result = subprocess.run(
            [binary, "pairwise", "embeddings", str(query), str(target)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)

        for expected in ("identity\tNA", "rmsd\tNA", "tm_score_query\tNA", "lddt\tNA"):
            if expected not in result.stdout.splitlines():
                raise SystemExit(f"missing {expected!r}\n{result.stdout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path


def require_line(text: str, expected: str) -> None:
    if expected not in text.splitlines():
        raise SystemExit(f"missing line {expected!r}\n{text}")


def require_absent(text: str, forbidden: str) -> None:
    if forbidden in text:
        raise SystemExit(f"unexpected public CLI text {forbidden!r}\n{text}")


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
    result = subprocess.run(
        [binary, "info"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)

    require_line(result.stdout, "default_weights\thikoboshi-mpnn-d64")
    require_line(result.stdout, "default_weights_status\tresolved")
    require_line(result.stdout, "default_weights_hidden_dim\t64")
    require_line(result.stdout, "default_weights_neighbor_count\t64")
    require_line(result.stdout, "default_weights_rbf_count\t16")
    require_line(result.stdout, "default_weights_layer_count\t3")
    require_line(result.stdout, "gap_open\t-1.4")
    require_line(result.stdout, "gap_extension\t-0.15")
    require_line(result.stdout, "soft_gap_open\t-3.21337")
    require_line(result.stdout, "soft_gap_extension\t-0.111704")
    require_line(result.stdout, "similarity\traw_dot_product")
    require_line(
        result.stdout,
        "checksum\tc61e079b539af5e31ba145ab91f7f607634295e90240e444a5bf34013304175b",
    )

    help_result = subprocess.run(
        [binary, "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if help_result.returncode != 0:
        raise SystemExit(help_result.stdout + help_result.stderr)
    public_help = (help_result.stdout + help_result.stderr).lower()
    for forbidden in ("substitution", "sequence-token", "blosum", "pam"):
        require_absent(public_help, forbidden)

    result = subprocess.run(
        [binary, "substitution-matrix"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        raise SystemExit("substitution-matrix CLI command must not exist")
    if "unknown command: substitution-matrix" not in result.stderr:
        raise SystemExit("missing unknown-command diagnostic for substitution-matrix")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        query = root / "query.npy"
        target = root / "target.npy"
        write_npy(query, [[1.0, 0.0], [0.0, 1.0]])
        write_npy(target, [[1.0, 0.0], [0.0, 1.0]])

        result = subprocess.run(
            [
                binary,
                "pairwise",
                "embeddings",
                str(query),
                str(target),
                "--gap-open",
                "-2.0",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        require_line(result.stdout, "raw_sw_score\t2")
        if "warning: gap_defaults_overridden" not in result.stderr:
            raise SystemExit("gap override warning missing from stderr")

        result = subprocess.run(
            [
                binary,
                "pairwise",
                "embeddings",
                str(query),
                str(target),
                "--score-method",
                "cosine",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            raise SystemExit("reserved score-method option must fail")
        if "cosine scoring" not in result.stderr or "score-only" not in result.stderr:
            raise SystemExit("reserved scoring diagnostic missing from stderr")

        result = subprocess.run(
            [
                binary,
                "pairwise",
                "embeddings",
                str(query),
                str(target),
                "--substitution-matrix",
                "BLOSUM62",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            raise SystemExit("substitution-matrix CLI option must not exist")
        if "unknown pairwise option" not in result.stderr:
            raise SystemExit("missing unknown-option diagnostic for substitution matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

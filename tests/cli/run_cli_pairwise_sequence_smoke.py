#!/usr/bin/env python3
"""CLI smoke test for `hikoboshi pairwise --package esm2-8m <q> <t>`.

Confirms the binary exits 0, prints sequence-mode summary fields, and
writes a valid FASTA file when `--fasta` is supplied. Pre-RC1 the
expected behavior is that hard SW on identical AA strings yields a
self-self identity alignment with positive raw_sw_score.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def require_line(text: str, expected: str) -> None:
    if expected not in text.splitlines():
        raise SystemExit(
            f"sequence_route_cli_smoke: missing line {expected!r}\n"
            f"stdout:\n{text}"
        )


def require_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(
            f"sequence_route_cli_smoke: missing fragment {needle!r}\n"
            f"text:\n{text}"
        )


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi_cli"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fasta = root / "alignment.fasta"

        result = subprocess.run(
            [
                str(binary),
                "pairwise",
                "--package",
                "esm2-8m",
                "--mode",
                "hard",
                "--fasta",
                str(fasta),
                "ACDEFG",
                "ACDEFG",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise SystemExit(
                "sequence_route_cli_smoke: command exited "
                f"{result.returncode}\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        require_line(result.stdout, "command\tpairwise")
        require_line(result.stdout, "input_mode\tsequence")
        require_contains(result.stdout, "raw_sw_score\t")
        require_line(result.stdout, "aligned_pairs\t6")

        if not fasta.exists():
            raise SystemExit(
                "sequence_route_cli_smoke: FASTA output file was not written"
            )
        contents = fasta.read_text()
        if "ACDEFG" not in contents:
            raise SystemExit(
                "sequence_route_cli_smoke: FASTA output missing AA letters"
                f"\n{contents}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

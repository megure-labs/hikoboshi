#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

PDB_BODY = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1       1.500   1.500   0.000  1.00  0.00           C\n"
    "ATOM      6  N   ALA A   2       4.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      7  CA  ALA A   2       5.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      8  C   ALA A   2       6.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      9  O   ALA A   2       7.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM     10  CB  ALA A   2       5.500   1.500   0.000  1.00  0.00           C\n"
    "ATOM     11  N   ALA A   3       8.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM     12  CA  ALA A   3       9.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM     13  C   ALA A   3      10.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM     14  O   ALA A   3      11.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM     15  CB  ALA A   3       9.500   1.500   0.000  1.00  0.00           C\n"
    "END\n"
)


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
    if not binary.exists():
        raise SystemExit(f"hikoboshi CLI not found at {binary}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        query = root / "query.pdb"
        target = root / "target.pdb"
        query.write_text(PDB_BODY, encoding="ascii")
        target.write_text(PDB_BODY, encoding="ascii")

        correspondences = root / "pairs.tsv"
        correspondences.write_text("0\t0\n1\t1\n2\t2\n", encoding="ascii")

        # --help surface should mention score-alignment in the top-level help
        # and print the subcommand-specific usage when invoked directly.
        top_level_help = subprocess.run(
            [str(binary), "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if "score-alignment" not in top_level_help.stdout:
            raise SystemExit(
                "hikoboshi --help must list the score-alignment subcommand: "
                f"{top_level_help.stdout!r}"
            )

        subcommand_help = subprocess.run(
            [str(binary), "score-alignment", "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for needle in ("--query", "--target", "--correspondences",
                       "--output-format", "0-based"):
            if needle not in subcommand_help.stdout:
                raise SystemExit(
                    f"score-alignment --help missing {needle!r}:\n"
                    f"{subcommand_help.stdout}"
                )

        # TSV output (default).
        summary = root / "score.tsv"
        tsv_run = subprocess.run(
            [
                str(binary),
                "score-alignment",
                "--query",
                str(query),
                "--target",
                str(target),
                "--correspondences",
                str(correspondences),
                "--summary",
                str(summary),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tsv_lines = tsv_run.stdout.strip().split("\n")
        if len(tsv_lines) != 2:
            raise SystemExit(
                f"score-alignment TSV output must be header + one row, got: {tsv_run.stdout!r}"
            )
        header_columns = tsv_lines[0].split("\t")
        for column in (
            "rmsd",
            "tm_score_query",
            "tm_score_target",
            "lddt",
            "identity",
            "coverage_query",
            "coverage_target",
            "coverage_mean",
            "aligned_pairs",
        ):
            if column not in header_columns:
                raise SystemExit(
                    f"TSV header missing column {column!r}: {header_columns!r}"
                )
        row_values = tsv_lines[1].split("\t")
        aligned_pairs = row_values[header_columns.index("aligned_pairs")]
        if aligned_pairs != "3":
            raise SystemExit(
                f"score-alignment self-self diagonal must report aligned_pairs=3, got {aligned_pairs}"
            )
        if summary.read_text() != tsv_run.stdout:
            raise SystemExit(
                "score-alignment --summary file must match stdout output"
            )

        # JSON output.
        json_run = subprocess.run(
            [
                str(binary),
                "score-alignment",
                "--query",
                str(query),
                "--target",
                str(target),
                "--correspondences",
                str(correspondences),
                "--output-format",
                "json",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = json.loads(json_run.stdout)
        for key in (
            "rmsd",
            "tm_score_query",
            "tm_score_target",
            "lddt",
            "identity",
            "coverage_query",
            "coverage_target",
            "coverage_mean",
            "aligned_pairs",
        ):
            if key not in payload:
                raise SystemExit(
                    f"JSON output missing key {key!r}: {payload!r}"
                )
        if payload["aligned_pairs"] != 3:
            raise SystemExit(
                f"JSON aligned_pairs must be 3, got {payload['aligned_pairs']}"
            )

        # Out-of-range index must exit with a non-zero invalid-arguments code
        # (2) and a descriptive stderr message.
        bad_pairs = root / "bad_pairs.tsv"
        bad_pairs.write_text("0\t99\n", encoding="ascii")
        bad_run = subprocess.run(
            [
                str(binary),
                "score-alignment",
                "--query",
                str(query),
                "--target",
                str(target),
                "--correspondences",
                str(bad_pairs),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if bad_run.returncode != 2:
            raise SystemExit(
                "out-of-range correspondence must exit with code 2, got "
                f"{bad_run.returncode}\nstderr: {bad_run.stderr}"
            )
        if "out of range" not in bad_run.stderr.lower():
            raise SystemExit(
                "out-of-range correspondence must surface 'out of range' in stderr: "
                f"{bad_run.stderr!r}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

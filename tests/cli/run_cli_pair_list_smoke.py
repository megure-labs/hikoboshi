#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def parse_summary(text: str) -> list[dict[str, str]]:
    lines = text.strip().splitlines()
    if not lines:
        raise SystemExit("pair-list summary was empty")
    header = lines[0].split("\t")
    if header[:3] != ["query_index", "target_index", "pair_id"]:
        raise SystemExit(f"unexpected pair-list header: {lines[0]!r}")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def require_soft_schema(rows: list[dict[str, str]], label: str) -> None:
    if not rows:
        raise SystemExit(f"{label} emitted no rows")
    for column in (
        "soft_sw_score",
        "sw_per_query_len",
        "sw_per_target_len",
        "sw_per_aligned",
    ):
        if column not in rows[0]:
            raise SystemExit(f"{label} missing soft score column {column}")
        if rows[0][column] == "NA":
            raise SystemExit(f"{label} column {column} was not populated")


def run_pair_list(
    binary: Path,
    pairs: Path,
    fasta: Path,
    summary: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(binary),
            "pair-list",
            "--pairs",
            str(pairs),
            "--fasta",
            str(fasta),
            "--summary",
            str(summary),
            *extra,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fasta = root / "proteins.fa"
        fasta.write_text(
            ">alpha\nACDEFG\n"
            ">beta\nACDFGG\n"
            ">gamma\nHHHHHH\n",
            encoding="ascii",
        )
        pairs = root / "pairs.tsv"
        input_pairs = [
            ("alpha", "beta"),
            ("gamma", "alpha"),
            ("alpha", "beta"),
        ]
        pairs.write_text(
            "# query_id\ttarget_id\n\n"
            + "\n".join(f"{query}\t{target}" for query, target in input_pairs)
            + "\n",
            encoding="ascii",
        )

        summary = root / "pair_list.tsv"
        result = run_pair_list(binary, pairs, fasta, summary)
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        if summary.read_text(encoding="ascii") != result.stdout:
            raise SystemExit("pair-list summary file must match stdout summary")

        rows = parse_summary(result.stdout)
        if len(rows) != len(input_pairs):
            raise SystemExit(
                f"expected {len(input_pairs)} pair-list rows, got {len(rows)}"
            )
        expected_indices = [("0", "1"), ("2", "0"), ("0", "1")]
        observed_indices = [
            (row["query_index"], row["target_index"]) for row in rows
        ]
        if observed_indices != expected_indices:
            raise SystemExit(
                f"pair-list rows not in input order: {observed_indices!r}"
            )
        expected_pair_ids = [f"{query}__{target}" for query, target in input_pairs]
        if [row["pair_id"] for row in rows] != expected_pair_ids:
            raise SystemExit("pair-list pair IDs did not preserve input order")
        for column in (
            "soft_sw_score",
            "sw_per_query_len",
            "sw_per_target_len",
            "sw_per_aligned",
        ):
            if column in rows[0]:
                raise SystemExit(f"hard pair-list unexpectedly emitted {column}")

        soft_summary = root / "pair_list_soft.tsv"
        soft = run_pair_list(
            binary,
            pairs,
            fasta,
            soft_summary,
            "--mode",
            "soft",
        )
        if soft.returncode != 0:
            raise SystemExit(soft.stdout + soft.stderr)
        if soft_summary.read_text(encoding="ascii") != soft.stdout:
            raise SystemExit("soft pair-list summary file must match stdout")
        soft_rows = parse_summary(soft.stdout)
        if len(soft_rows) != len(input_pairs):
            raise SystemExit("soft pair-list row count mismatch")
        require_soft_schema(soft_rows, "soft pair-list")

        missing_pairs = root / "missing.tsv"
        missing_pairs.write_text("alpha\tabsent_id\n", encoding="ascii")
        missing_summary = root / "missing.tsv.out"
        missing = run_pair_list(
            binary,
            missing_pairs,
            fasta,
            missing_summary,
        )
        if missing.returncode == 0:
            raise SystemExit("pair-list missing-ID case must fail")
        if "absent_id" not in missing.stderr:
            raise SystemExit(
                "pair-list missing-ID diagnostic must name the absent ID\n"
                f"stderr:\n{missing.stderr}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

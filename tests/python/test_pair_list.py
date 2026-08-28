from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

import hikoboshi as hiko

from helpers import require, write_simple_pdb

_METRIC_COLUMNS = (
    "coverage_query",
    "coverage_target",
    "coverage_mean",
    "identity",
    "rmsd",
    "tm_score_query",
    "tm_score_target",
    "lddt",
    "lddt_byA",
    "lddt_byB",
    "lddt_aln",
    "coverage_byA",
    "coverage_byB",
    "ecs",
)
_TOLERANCE = 1e-5


def _binary() -> Path:
    build_root = os.environ.get("HIKOBOSHI_BUILD_ROOT")
    require(build_root is not None, "HIKOBOSHI_BUILD_ROOT is required")
    return Path(build_root) / "hikoboshi"


def _write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text(
        "# query_id\ttarget_id\n"
        + "\n".join(f"{query}\t{target}" for query, target in pairs)
        + "\n",
        encoding="ascii",
    )


def _parse_tsv(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line]
    require(bool(lines), "pair-list CLI output was empty")
    header = lines[0].split("\t")
    require(
        header[:3] == ["query_index", "target_index", "pair_id"],
        f"unexpected pair-list header: {header!r}",
    )
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        require(
            len(cells) == len(header),
            f"row column count {len(cells)} != header {len(header)}",
        )
        rows.append(dict(zip(header, cells)))
    return rows


def _run_cli(
    pairs_path: Path,
    *,
    fasta_path: Path | None = None,
    pdb_dir: Path | None = None,
) -> list[dict[str, str]]:
    summary = pairs_path.with_suffix(".out.tsv")
    command = [
        str(_binary()),
        "pair-list",
        "--pairs",
        str(pairs_path),
        "--summary",
        str(summary),
    ]
    if fasta_path is not None:
        command += ["--fasta", str(fasta_path)]
    else:
        require(pdb_dir is not None, "pdb_dir is required for structure CLI run")
        command.append(str(pdb_dir))
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stdout + completed.stderr)
    require(
        summary.read_text(encoding="ascii") == completed.stdout,
        "pair-list --summary file must match stdout",
    )
    return _parse_tsv(completed.stdout)


def _optional_float(token: str) -> float | None:
    if token == "NA":
        return None
    return float(token)


def _assert_results_match_cli(
    results: list[hiko.PairwiseResult],
    rows: list[dict[str, str]],
    pairs: list[tuple[str, str]],
) -> None:
    require(len(results) == len(pairs), "Python pair-list result count mismatch")
    require(len(rows) == len(pairs), "CLI pair-list row count mismatch")
    require(
        [row["pair_id"] for row in rows]
        == [f"{query}__{target}" for query, target in pairs],
        "CLI pair-list rows are not in input pair order",
    )
    for index, (result, row) in enumerate(zip(results, rows)):
        require(
            result.path.aligned_pairs == int(row["aligned_pairs"]),
            f"row {index} aligned_pairs mismatch",
        )
        require(
            math.isclose(
                result.metrics.raw_sw_score,
                float(row["raw_sw_score"]),
                rel_tol=_TOLERANCE,
                abs_tol=_TOLERANCE,
            ),
            f"row {index} raw_sw_score mismatch",
        )
        for column in _METRIC_COLUMNS:
            observed = getattr(result.metrics, column)
            expected = _optional_float(row[column])
            if observed is None or expected is None:
                require(
                    observed is None and expected is None,
                    f"row {index} {column} validity mismatch",
                )
                continue
            require(
                math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=_TOLERANCE,
                    abs_tol=_TOLERANCE,
                ),
                f"row {index} {column} mismatch",
            )


def test_sequence_route(root: Path) -> None:
    fasta = root / "proteins.fa"
    fasta.write_text(
        ">alpha\nACDEFG\n"
        ">beta\nACDFGG\n"
        ">gamma\nHHHH\n",
        encoding="ascii",
    )
    pairs = [("gamma", "alpha"), ("alpha", "beta"), ("alpha", "beta")]
    pairs_path = root / "sequence_pairs.tsv"
    _write_pairs(pairs_path, pairs)

    results = hiko.pair_list_from_sequence(pairs, fasta)
    rows = _run_cli(pairs_path, fasta_path=fasta)
    _assert_results_match_cli(results, rows, pairs)


def test_structure_route(root: Path) -> None:
    pdb_dir = root / "pdbs"
    pdb_dir.mkdir()
    write_simple_pdb(pdb_dir / "alpha.pdb", residue_count=2)
    write_simple_pdb(pdb_dir / "beta.pdb", residue_count=3)
    write_simple_pdb(pdb_dir / "gamma.pdb", residue_count=2)
    pairs = [("beta.pdb", "alpha.pdb"), ("alpha.pdb", "gamma.pdb")]
    pairs_path = root / "structure_pairs.tsv"
    _write_pairs(pairs_path, pairs)

    results = hiko.pair_list_from_structure(pairs, pdb_dir)
    rows = _run_cli(pairs_path, pdb_dir=pdb_dir)
    _assert_results_match_cli(results, rows, pairs)


def main() -> int:
    require(
        callable(getattr(hiko, "pair_list_from_sequence", None)),
        "hikoboshi.pair_list_from_sequence must resolve",
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_sequence_route(root)
        test_structure_route(root)
    print("pair_list_python: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

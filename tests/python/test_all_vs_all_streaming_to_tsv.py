"""Parity test for the p46 streaming Python `to_tsv` entry point.

Compares the streaming `hikoboshi.all_vs_all.to_tsv_from_X` path against the
buffered `hikoboshi.all_vs_all.from_X` path on the same inputs. The streaming
path writes its TSV through the C++ `TsvStreamingAllVsAllSink`; the buffered
path returns the same records the streaming sink would have emitted, so the
two paths must agree on:

- pair count and pair (query_index, target_index) order
- raw_sw_score and aligned_pairs (exact)
- optional metric values to within a tight float tolerance
- TSV header and column count

Byte-identical comparison between the streaming TSV writer and a Python-side
hand renderer is exercised by the C++ parity test
``tests/cpp/cli_all_vs_all_streaming_summary_tests.cpp`` so this Python test
can stay focused on the binding and on the round-trip from records to TSV.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import hikoboshi as hiko

from helpers import matrix, require, scalar_matrix, write_simple_pdb


_NOT_AVAILABLE = "NA"
_HEADER_COLUMNS = (
    "query_index",
    "target_index",
    "pair_id",
    "raw_sw_score",
    "aligned_pairs",
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
    "fasta_path",
    "pdb_path",
)
_SOFT_SCHEMA_COLUMNS = (
    "soft_sw_score",
    "sw_per_query_len",
    "sw_per_target_len",
    "sw_per_aligned",
)
_SOFT_HEADER_COLUMNS = (
    _HEADER_COLUMNS[:4] + _SOFT_SCHEMA_COLUMNS + _HEADER_COLUMNS[4:]
)
# Streaming TSV serializes raw_sw_score and metrics with six significant
# digits (`std::setprecision(6)` in `format_double`), so a parsed value
# can differ from the in-memory double by up to 5e-7 * value. A 1e-5
# relative tolerance (10x serialization round-trip error) keeps the
# comparison robust without masking real divergence in the engine path.
_METRIC_TOLERANCE = 1e-5


def _parse_optional_float(token: str) -> float | None:
    if token == _NOT_AVAILABLE:
        return None
    return float(token)


def _parse_streaming_tsv(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline().rstrip("\n")
        header = header_line.split("\t")
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            require(
                len(cells) == len(header),
                f"streaming TSV row column count {len(cells)} != header {len(header)}",
            )
            row = dict(zip(header, cells))
            rows.append(row)
    return header, rows


def _check_metric_close(
    streamed: float | None,
    buffered: float | None,
    field: str,
    record_index: int,
) -> None:
    if streamed is None and buffered is None:
        return
    require(
        streamed is not None and buffered is not None,
        f"record[{record_index}] {field} validity mismatch: "
        f"streamed={streamed!r}, buffered={buffered!r}",
    )
    # The streaming path renders metrics with six significant digits, so
    # within a few ulps of 1e-6 relative tolerance is the right comparison.
    require(
        math.isclose(
            float(streamed),
            float(buffered),
            rel_tol=_METRIC_TOLERANCE,
            abs_tol=_METRIC_TOLERANCE,
        ),
        f"record[{record_index}] {field} differs: "
        f"streamed={streamed}, buffered={buffered}",
    )


def _check_streaming_summary(
    summary: hiko.results.AllVsAllStreamingSummary,
    expected_pair_count: int,
    output_path: Path,
) -> None:
    require(
        summary.pair_count == expected_pair_count,
        f"streaming pair_count {summary.pair_count} != {expected_pair_count}",
    )
    require(
        summary.wall_time_seconds >= 0.0,
        f"streaming wall time must be non-negative, got {summary.wall_time_seconds}",
    )
    require(
        Path(summary.output_path) == output_path,
        f"streaming output_path {summary.output_path!r} != {output_path!r}",
    )


def _check_streaming_matches_buffered(
    streaming_path: Path,
    buffered_records,
    *,
    label: str,
    expected_header: tuple[str, ...] = _HEADER_COLUMNS,
    optional_metric_columns: tuple[str, ...] = (),
) -> None:
    header, rows = _parse_streaming_tsv(streaming_path)
    require(
        tuple(header) == expected_header,
        f"{label}: TSV header mismatch: {header!r}",
    )
    require(
        len(rows) == len(buffered_records),
        f"{label}: streaming row count {len(rows)} != buffered records {len(buffered_records)}",
    )
    for index, (row, record) in enumerate(zip(rows, buffered_records)):
        require(
            int(row["query_index"]) == record.query_index,
            f"{label}: record[{index}] query_index mismatch",
        )
        require(
            int(row["target_index"]) == record.target_index,
            f"{label}: record[{index}] target_index mismatch",
        )
        require(
            int(row["aligned_pairs"]) == record.result.path.aligned_pairs,
            f"{label}: record[{index}] aligned_pairs mismatch",
        )
        # raw_sw_score is a double rendered with six significant digits.
        require(
            math.isclose(
                float(row["raw_sw_score"]),
                float(record.result.metrics.raw_sw_score),
                rel_tol=_METRIC_TOLERANCE,
                abs_tol=_METRIC_TOLERANCE,
            ),
            f"{label}: record[{index}] raw_sw_score differs",
        )
        for column in (
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
        ):
            _check_metric_close(
                _parse_optional_float(row[column]),
                getattr(record.result.metrics, column),
                column,
                index,
            )
        for column in optional_metric_columns:
            _check_metric_close(
                _parse_optional_float(row[column]),
                getattr(record.result.metrics, column),
                column,
                index,
            )
        # The Python `to_tsv` entry point does not pass a pair_id callback,
        # so the pair_id column is empty.
        require(
            row["pair_id"] == "",
            f"{label}: record[{index}] pair_id must be empty when no callback supplied",
        )
        require(
            row["fasta_path"] == "" and row["pdb_path"] == "",
            f"{label}: record[{index}] artifact paths must be empty",
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Embedding parity on a multi-residue input set so the alignment
        # produces nontrivial metrics on the records side.
        embeddings = [
            matrix(
                [
                    [float(index + 1), float(index + 2), float(index + 3)],
                    [float(index + 4), float(index + 5), float(index + 6)],
                ]
            )
            for index in range(4)
        ]
        streaming_path = root / "streaming_embeddings.tsv"
        summary = hiko.all_vs_all.to_tsv_from_embeddings(
            embeddings, streaming_path
        )
        _check_streaming_summary(
            summary, expected_pair_count=6, output_path=streaming_path
        )
        collected = hiko.all_vs_all.from_embeddings(embeddings)
        _check_streaming_matches_buffered(
            streaming_path, collected.records, label="embeddings"
        )

        soft_streaming_path = root / "streaming_embeddings_soft.tsv"
        summary_soft = hiko.all_vs_all.to_tsv_from_embeddings(
            embeddings, soft_streaming_path, mode="soft", temperature=1.0
        )
        _check_streaming_summary(
            summary_soft, expected_pair_count=6, output_path=soft_streaming_path
        )
        collected_soft = hiko.all_vs_all.from_embeddings(
            embeddings, mode="soft", temperature=1.0
        )
        _check_streaming_matches_buffered(
            soft_streaming_path,
            collected_soft.records,
            label="soft embeddings",
            expected_header=_SOFT_HEADER_COLUMNS,
            optional_metric_columns=_SOFT_SCHEMA_COLUMNS,
        )

        # include_self covers the symmetric pair_count formula and exercises
        # the diagonal records.
        single_residue = [scalar_matrix(float(index + 1)) for index in range(3)]
        streaming_self = root / "streaming_self.tsv"
        summary_self = hiko.all_vs_all.to_tsv_from_embeddings(
            single_residue, streaming_self, include_self=True
        )
        _check_streaming_summary(
            summary_self, expected_pair_count=6, output_path=streaming_self
        )
        collected_self = hiko.all_vs_all.from_embeddings(
            single_residue, include_self=True
        )
        _check_streaming_matches_buffered(
            streaming_self, collected_self.records, label="include_self embeddings"
        )

        # Empty run still emits a header so downstream consumers can parse.
        empty_path = root / "empty.tsv"
        summary_empty = hiko.all_vs_all.to_tsv_from_embeddings(
            [scalar_matrix(1.0)], empty_path
        )
        _check_streaming_summary(
            summary_empty, expected_pair_count=0, output_path=empty_path
        )
        with empty_path.open("r", encoding="utf-8") as handle:
            empty_header = handle.readline().rstrip("\n").split("\t")
            empty_remainder = handle.read()
        require(
            tuple(empty_header) == _HEADER_COLUMNS,
            f"empty TSV header mismatch: {empty_header!r}",
        )
        require(
            empty_remainder == "",
            f"empty TSV must contain header only, got {empty_remainder!r}",
        )

        # Structure path through filesystem PDB inputs.
        pdbs = [root / "a.pdb", root / "b.pdb", root / "c.pdb"]
        for path in pdbs:
            write_simple_pdb(path, residue_count=3)
        streaming_pdb = root / "streaming_pdb.tsv"
        summary_pdb = hiko.all_vs_all.to_tsv_from_pdb(
            [str(path) for path in pdbs], streaming_pdb
        )
        _check_streaming_summary(
            summary_pdb, expected_pair_count=3, output_path=streaming_pdb
        )
        collected_pdb = hiko.all_vs_all.from_pdb([str(path) for path in pdbs])
        _check_streaming_matches_buffered(
            streaming_pdb, collected_pdb.records, label="pdb"
        )

        # Coords route shares the same C++ engine path.
        streaming_coords = root / "streaming_coords.tsv"
        summary_coords = hiko.all_vs_all.to_tsv_from_coords(
            [str(path) for path in pdbs], streaming_coords
        )
        _check_streaming_summary(
            summary_coords, expected_pair_count=3, output_path=streaming_coords
        )
        collected_coords = hiko.all_vs_all.from_coords([str(path) for path in pdbs])
        _check_streaming_matches_buffered(
            streaming_coords, collected_coords.records, label="coords"
        )

        # The dispatcher should pick the structure path for filesystem inputs.
        streaming_dispatch = root / "streaming_dispatch.tsv"
        summary_dispatch = hiko.all_vs_all.to_tsv(
            [str(path) for path in pdbs], streaming_dispatch
        )
        _check_streaming_summary(
            summary_dispatch, expected_pair_count=3, output_path=streaming_dispatch
        )
        _check_streaming_matches_buffered(
            streaming_dispatch, collected_pdb.records, label="to_tsv() dispatcher"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

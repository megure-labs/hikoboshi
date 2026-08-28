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


def write_pdb(path: Path) -> None:
    path.write_text(
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
        "END\n",
        encoding="ascii",
    )


def require_pairs(stdout: str, expected_pairs: list[tuple[str, str]]) -> None:
    lines = stdout.strip().splitlines()
    if len(lines) != len(expected_pairs) + 1:
        raise SystemExit(f"expected header plus {len(expected_pairs)} records\n{stdout}")
    header = lines[0].split("\t")
    if header[:3] != ["query_index", "target_index", "pair_id"]:
        raise SystemExit(f"unexpected all-vs-all header: {lines[0]!r}")
    observed_pairs = [tuple(line.split("\t")[:2]) for line in lines[1:]]
    if observed_pairs != expected_pairs:
        raise SystemExit(f"unexpected pair order: {observed_pairs!r}")


def parse_summary(stdout: str) -> list[dict[str, str]]:
    lines = stdout.strip().splitlines()
    if not lines:
        raise SystemExit("all-vs-all summary was empty")
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def require_soft_schema(rows: list[dict[str, str]], label: str) -> None:
    if not rows:
        raise SystemExit(f"{label} emitted no rows")
    required = (
        "soft_sw_score",
        "sw_per_query_len",
        "sw_per_target_len",
        "sw_per_aligned",
    )
    for column in required:
        if column not in rows[0]:
            raise SystemExit(f"{label} missing soft score column {column}")
        if rows[0][column] == "NA":
            raise SystemExit(f"{label} column {column} was not populated")


def run_all_vs_all(binary: Path, inputs: list[Path], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, "all-vs-all", "embeddings", *map(str, inputs), *extra],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_structure_all_vs_all(
    binary: Path, mode: str, inputs: list[Path], *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, "all-vs-all", mode, *map(str, inputs), *extra],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        help_result = subprocess.run(
            [binary, "all-vs-all", "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if help_result.returncode != 0:
            raise SystemExit(help_result.stdout + help_result.stderr)
        if "Structure inputs may be files or directories" not in help_result.stdout:
            raise SystemExit("all-vs-all help did not document directory inputs")

        inputs = [root / "a.npy", root / "b.npy", root / "c.npy"]
        for index, path in enumerate(inputs, 1):
            write_npy(path, [[float(index)]])
        summary = root / "all_vs_all.tsv"

        result = run_all_vs_all(binary, inputs, "--summary", str(summary))
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)

        require_pairs(result.stdout, [("0", "1"), ("0", "2"), ("1", "2")])
        hard_header = result.stdout.splitlines()[0].split("\t")
        for column in (
            "soft_sw_score",
            "sw_per_query_len",
            "sw_per_target_len",
            "sw_per_aligned",
        ):
            if column in hard_header:
                raise SystemExit(f"hard all-vs-all unexpectedly emitted {column}")
        if summary.read_text() != result.stdout:
            raise SystemExit("all-vs-all summary file must match stdout summary")

        soft_summary = root / "all_vs_all_soft.tsv"
        soft = run_all_vs_all(
            binary, inputs, "--mode", "soft", "--summary", str(soft_summary)
        )
        if soft.returncode != 0:
            raise SystemExit(soft.stdout + soft.stderr)
        require_pairs(soft.stdout, [("0", "1"), ("0", "2"), ("1", "2")])
        require_soft_schema(parse_summary(soft.stdout), "soft all-vs-all")
        if soft_summary.read_text() != soft.stdout:
            raise SystemExit("soft all-vs-all summary file must match stdout")

        bad = subprocess.run(
            [binary, "all-vs-all", "embeddings", str(inputs[0]), str(inputs[1]), "--score-only"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if bad.returncode == 0:
            raise SystemExit("reserved all-vs-all score-only option must fail")
        if "score-only" not in bad.stderr or "cosine scoring" not in bad.stderr:
            raise SystemExit("reserved all-vs-all scoring diagnostic missing")

        threaded_inputs = [root / f"threaded_{index}.npy" for index in range(10)]
        for index, path in enumerate(threaded_inputs, 1):
            write_npy(path, [[float(index)]])

        serial = run_all_vs_all(binary, threaded_inputs, "--threads", "1")
        if serial.returncode != 0:
            raise SystemExit(serial.stdout + serial.stderr)
        auto = run_all_vs_all(binary, threaded_inputs, "--threads=0")
        if auto.returncode != 0:
            raise SystemExit(auto.stdout + auto.stderr)
        threaded = run_all_vs_all(binary, threaded_inputs, "--threads=4")
        if threaded.returncode != 0:
            raise SystemExit(threaded.stdout + threaded.stderr)
        if serial.stdout != auto.stdout:
            raise SystemExit("auto all-vs-all threads changed output")
        if serial.stdout != threaded.stdout:
            raise SystemExit("threaded all-vs-all output changed from serial")

        for value in ("-1", "many"):
            bad_threads = run_all_vs_all(
                binary, inputs[:2], "--threads", value
            )
            if bad_threads.returncode == 0:
                raise SystemExit(f"all-vs-all --threads {value} must fail")
            if "threads must be a non-negative integer" not in bad_threads.stderr:
                raise SystemExit(
                    f"all-vs-all --threads {value} diagnostic missing\n"
                    f"{bad_threads.stderr}"
                )

        pdbs = [root / "a.pdb", root / "b.pdb"]
        for path in pdbs:
            write_pdb(path)
        for mode in ("pdb", "coords"):
            result = run_structure_all_vs_all(binary, mode, pdbs)
            if result.returncode != 0:
                raise SystemExit(result.stdout + result.stderr)
            require_pairs(result.stdout, [("0", "1")])

        structure_dir = root / "structure_dir"
        structure_dir.mkdir()
        directory_inputs = [
            structure_dir / "a.ent",
            structure_dir / "b.PDB",
            structure_dir / "c.pdb",
        ]
        for path in directory_inputs:
            write_pdb(path)
        (structure_dir / "ignore.txt").write_text("not a structure\n", encoding="ascii")

        positional = run_structure_all_vs_all(binary, "structure", directory_inputs)
        if positional.returncode != 0:
            raise SystemExit(positional.stdout + positional.stderr)
        directory = run_structure_all_vs_all(binary, "structure", [structure_dir])
        if directory.returncode != 0:
            raise SystemExit(directory.stdout + directory.stderr)
        if directory.stdout != positional.stdout:
            raise SystemExit(
                "directory all-vs-all output differed from sorted positional input\n"
                f"positional:\n{positional.stdout}\n"
                f"directory:\n{directory.stdout}"
            )
        require_pairs(directory.stdout, [("0", "1"), ("0", "2"), ("1", "2")])

        empty_dir = root / "empty_structures"
        empty_dir.mkdir()
        (empty_dir / "README.txt").write_text("no structures here\n", encoding="ascii")
        empty = run_structure_all_vs_all(binary, "structure", [empty_dir])
        if empty.returncode == 0:
            raise SystemExit("empty structure directory must fail")
        if "all-vs-all structure directory contained no PDB or mmCIF files" not in empty.stderr:
            raise SystemExit(
                "empty structure directory diagnostic missing\n" + empty.stderr
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

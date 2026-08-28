#!/usr/bin/env python3
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def require_line(text: str, expected: str) -> None:
    if expected not in text.splitlines():
        raise SystemExit(f"missing line {expected!r}\n{text}")


def run_cli(binary: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


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


def require_package_error(result: subprocess.CompletedProcess[str]) -> None:
    require(result.returncode != 0, "package rejection command must fail")
    combined = result.stdout + result.stderr
    require(
        "external package paths are not supported" in combined,
        f"missing external package diagnostic\n{combined}",
    )
    require(
        "unknown Hikoboshi package id" not in combined,
        f"path-like package must not fall through to unknown-name lookup\n{combined}",
    )


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"

    for command in ("encode", "pairwise", "all-vs-all"):
        result = run_cli(binary, command, "--help")
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        require(
            "--package NAME" in result.stdout,
            f"{command} help is missing --package NAME\n{result.stdout}",
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        query = root / "query.npy"
        target = root / "target.npy"
        pdb = root / "input.pdb"
        write_npy(query, [[1.0, 0.0], [0.0, 1.0]])
        write_npy(target, [[1.0, 0.0], [0.0, 1.0]])
        write_pdb(pdb)

        result = run_cli(
            binary,
            "pairwise",
            "embeddings",
            str(query),
            str(target),
            "--package=hikoboshi-mpnn-d64",
        )
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        require_line(result.stdout, "raw_sw_score\t2")

        result = run_cli(
            binary,
            "all-vs-all",
            "embeddings",
            str(query),
            str(target),
            "--package",
            "mpnn64",
        )
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        require(
            any(line.startswith("0\t1\t") for line in result.stdout.splitlines()),
            f"all-vs-all alias run did not emit the expected pair\n{result.stdout}",
        )

        result = run_cli(
            binary,
            "encode",
            "pdb",
            str(pdb),
            "--package",
            "mpnn-64",
        )
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        require_line(result.stdout, "command\tencode")
        require_line(result.stdout, "input_mode\tpdb")

        result = run_cli(
            binary,
            "pairwise",
            "embeddings",
            str(query),
            str(target),
            "--package",
            "unknown-package",
        )
        require(result.returncode != 0, "unknown package must fail")
        combined = result.stdout + result.stderr
        for expected in (
            "unknown Hikoboshi package id",
            "available compiled package IDs/aliases",
            "hikoboshi-mpnn-d64",
            "mpnn64",
            "mpnn-64",
        ):
            require(expected in combined, f"missing {expected!r}\n{combined}")

        for package_value in (
            "packages/hikoboshi-mpnn-d64",
            r"packages\hikoboshi-mpnn-d64",
            "hikoboshi-mpnn-d64.json",
        ):
            require_package_error(
                run_cli(
                    binary,
                    "pairwise",
                    "embeddings",
                    str(query),
                    str(target),
                    "--package",
                    package_value,
                )
            )

        package_path = str(root / "packages" / "hikoboshi-mpnn-d64")
        for command in (
            ("encode", "pdb", str(pdb)),
            ("pairwise", "embeddings", str(query), str(target)),
            ("all-vs-all", "embeddings", str(query), str(target)),
        ):
            require_package_error(
                run_cli(binary, *command, "--package-path", package_path)
            )
            require_package_error(
                run_cli(binary, *command, f"--package-path={package_path}")
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

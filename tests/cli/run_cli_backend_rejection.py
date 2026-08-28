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
        "END\n",
        encoding="ascii",
    )


RESERVED_BACKENDS = {
    "sse4": "future x86 SSE4 SIMD builds",
    "avx2": "future x86 AVX2 SIMD builds",
    "avx512": "future x86 AVX-512 SIMD builds",
    "neon": "future ARM NEON SIMD builds",
    "sve": "future ARM SVE SIMD builds",
    "cuda": "future NVIDIA CUDA GPU builds",
    "hip": "future AMD HIP GPU builds",
    "metal": "future Apple Metal GPU builds",
    "vulkan": "future Vulkan GPU builds",
    "opencl": "future OpenCL GPU builds",
    "fat": "future multi-backend build bundles",
}


def require_reserved_backend_error(
    result: subprocess.CompletedProcess[str],
    backend: str,
    expected_detail: str,
) -> None:
    require(result.returncode != 0, f"{backend} backend command must fail")
    combined = result.stdout + result.stderr
    for expected in (
        f"backend '{backend}'",
        "reserved",
        expected_detail,
        "Hikoboshi 0.1.0 accepts only auto or scalar",
    ):
        require(expected in combined, f"missing {expected!r}\n{combined}")
    require("raw_sw_score" not in result.stdout, "reserved backend must not run")


def main() -> int:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        query = root / "query.npy"
        target = root / "target.npy"
        pdb = root / "input.pdb"
        write_npy(query, [[1.0, 0.0], [0.0, 1.0]])
        write_npy(target, [[1.0, 0.0], [0.0, 1.0]])
        write_pdb(pdb)

        for accepted in ("auto", "scalar"):
            result = run_cli(
                binary,
                "pairwise",
                "embeddings",
                str(query),
                str(target),
                "--backend",
                accepted,
            )
            if result.returncode != 0:
                raise SystemExit(result.stdout + result.stderr)
            require(
                "raw_sw_score\t2" in result.stdout.splitlines(),
                f"accepted backend {accepted} did not run pairwise\n{result.stdout}",
            )

        commands = (
            ("encode", "pdb", str(pdb)),
            ("pairwise", "embeddings", str(query), str(target)),
            ("all-vs-all", "embeddings", str(query), str(target)),
        )
        for backend, expected_detail in RESERVED_BACKENDS.items():
            for command in commands:
                require_reserved_backend_error(
                    run_cli(binary, *command, "--backend", backend),
                    backend,
                    expected_detail,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

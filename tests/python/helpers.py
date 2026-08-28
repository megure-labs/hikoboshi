from __future__ import annotations

from array import array
from pathlib import Path
from typing import NoReturn

_ALA2_BODY = (
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
)
_ALA3_BODY = (
    _ALA2_BODY
    + "ATOM     11  N   ALA A   3       8.000   0.000   0.000  1.00  0.00           N\n"
    + "ATOM     12  CA  ALA A   3       9.000   0.000   0.000  1.00  0.00           C\n"
    + "ATOM     13  C   ALA A   3      10.000   0.000   0.000  1.00  0.00           C\n"
    + "ATOM     14  O   ALA A   3      11.000   0.000   0.000  1.00  0.00           O\n"
    + "ATOM     15  CB  ALA A   3       9.500   1.500   0.000  1.00  0.00           C\n"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def skip(message: str) -> NoReturn:
    print(message)
    raise SystemExit(77)


def matrix(rows: list[list[float]]) -> memoryview:
    flat = array("f", [value for row in rows for value in row])
    return memoryview(flat).cast("B").cast("f", shape=(len(rows), len(rows[0])))


def scalar_matrix(value: float) -> memoryview:
    flat = array("f", [value])
    return memoryview(flat).cast("B").cast("f", shape=(1, 1))


def write_simple_pdb(path: Path, *, residue_count: int = 2) -> None:
    if residue_count == 2:
        body = _ALA2_BODY
    elif residue_count == 3:
        body = _ALA3_BODY
    else:
        raise ValueError(f"unsupported residue_count: {residue_count}")
    path.write_text(body + "END\n", encoding="ascii")


def write_two_model_pdb(path: Path) -> None:
    path.write_text(
        f"MODEL        1\n{_ALA2_BODY}ENDMDL\n"
        f"MODEL        2\n{_ALA2_BODY}ENDMDL\nEND\n",
        encoding="ascii",
    )


def require_backend_availability(
    payload: object,
    *,
    compiled: bool,
    available: bool,
    label: str = "backend",
) -> None:
    require(isinstance(payload, dict), f"{label} availability must be a dict")
    require(payload.get("compiled") is compiled, f"{label} compiled flag mismatch")
    require(
        payload.get("runtime_available") is available,
        f"{label} runtime availability mismatch",
    )
    require(isinstance(payload.get("reason"), str), f"{label} reason must be a string")
    if not compiled:
        require(payload["reason"], f"{label} reserved reason must be non-empty")


def require_sha256(value: object, label: str) -> None:
    require(isinstance(value, str), f"{label} must be a string")
    require(len(value) == 64, f"{label} must be a SHA-256 hex digest")
    require(all(char in "0123456789abcdef" for char in value), f"{label} must be hex")


__all__ = [
    "matrix",
    "require",
    "require_backend_availability",
    "require_sha256",
    "scalar_matrix",
    "skip",
    "write_simple_pdb",
    "write_two_model_pdb",
]

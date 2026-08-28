from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import hikoboshi as hiko

from helpers import require, write_simple_pdb


def require_encoded(result: hiko.EncodeResult, *, label: str) -> None:
    embeddings: Any = result.embeddings
    require(len(embeddings) == 2, f"{label} residue count mismatch")
    require(len(embeddings[0]) == 64, f"{label} embedding dimension mismatch")
    require(result.metadata is not None, f"{label} metadata missing")
    require(result.metadata["residue_count"] == 2, f"{label} metadata count mismatch")
    require(result.metadata["dimension"] == 64, f"{label} metadata dimension mismatch")
    require(result.metadata["residue_codes"] == "AA", f"{label} residue codes mismatch")


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "query.pdb"
    write_simple_pdb(path)

    require_encoded(
        hiko.encode.from_structure(str(path), package="hikoboshi-mpnn-d64"),
        label="structure encode",
    )
    require_encoded(
        hiko.encode.from_coords(str(path), package="hikoboshi-mpnn-d64"),
        label="coords encode",
    )

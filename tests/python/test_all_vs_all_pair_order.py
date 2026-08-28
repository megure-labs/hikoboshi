from __future__ import annotations

import tempfile
from pathlib import Path

import hikoboshi as hiko

from helpers import require, scalar_matrix, write_simple_pdb


def expected_pairs(count: int, include_self: bool) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(count):
        start = i if include_self else i + 1
        for j in range(start, count):
            pairs.append((i, j))
    return pairs


for count in range(4):
    inputs = [scalar_matrix(float(index + 1)) for index in range(count)]
    for include_self in (False, True):
        result = hiko.all_vs_all.from_embeddings(inputs, include_self=include_self)
        observed = [(record.query_index, record.target_index) for record in result.records]
        require(
            observed == expected_pairs(count, include_self),
            f"pair order mismatch for count={count} include_self={include_self}",
        )

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    paths = [root / "a.pdb", root / "b.pdb"]
    for path in paths:
        write_simple_pdb(path, residue_count=3)
    for route in (hiko.all_vs_all.from_structure, hiko.all_vs_all.from_coords):
        result = route([str(path) for path in paths])
        observed = [(record.query_index, record.target_index) for record in result.records]
        require(observed == [(0, 1)], f"{route!r} returned unexpected pair order")

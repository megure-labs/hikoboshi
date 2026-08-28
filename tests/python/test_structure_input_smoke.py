from __future__ import annotations

import tempfile
from pathlib import Path

import hikoboshi as hiko
from hikoboshi._arrays import load_core

from helpers import require, write_simple_pdb


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    query = root / "query.pdb"
    target = root / "target.pdb"
    write_simple_pdb(query)
    write_simple_pdb(target)

    structure_result = hiko.pairwise.from_structure(
        str(query),
        str(target),
        package="hikoboshi-mpnn-d64",
        chain_index=None,
        pdb_model_index=None,
    )
    require(
        structure_result.path.aligned_pairs > 0,
        "pairwise.from_structure returned an empty alignment",
    )

    coords_result = hiko.pairwise.from_coords(
        str(query),
        str(target),
        package="hikoboshi-mpnn-d64",
    )
    require(
        coords_result.path.aligned_pairs > 0,
        "pairwise.from_coords returned an empty alignment",
    )

    core = load_core()
    positional_metadata = core.load_structure_metadata(str(query), None, None)
    require(
        positional_metadata["residue_count"] == 2,
        "load_structure_metadata positional None selectors changed residue count",
    )
    keyword_metadata = core.load_structure_metadata(
        str(query),
        chain_index=None,
        pdb_model_index=None,
    )
    require(
        keyword_metadata["selected_chain_id"] == "A",
        "load_structure_metadata keyword None selectors changed chain selection",
    )

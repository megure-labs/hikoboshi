from __future__ import annotations

import hikoboshi as hiko

from helpers import matrix, require


query = matrix([[1.0, 0.0], [0.0, 1.0]])
target = matrix([[1.0, 0.0], [0.0, 1.0]])

without_metadata = hiko.pairwise.from_embeddings(query, target)
require(without_metadata.metrics.identity is None, "identity requires sequence metadata")
require(without_metadata.metrics.rmsd is None, "embedding-only RMSD must be unavailable")
require(
    without_metadata.metrics.invalid_reasons["rmsd"] == "missing_structure_metadata",
    "embedding-only structural metric reason mismatch",
)

with_metadata = hiko.pairwise.from_embeddings(
    query,
    target,
    query_metadata={"input_id": "query", "residue_codes": "AC"},
    target_metadata={"input_id": "target", "residue_codes": "AC"},
)
require(with_metadata.metrics.identity == 1.0, "sequence metadata should enable identity")
require(with_metadata.query_metadata["input_id"] == "query", "query metadata not preserved")
require(with_metadata.target_metadata["input_id"] == "target", "target metadata not preserved")

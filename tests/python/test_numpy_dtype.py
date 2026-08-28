from __future__ import annotations

import hikoboshi as hiko
from hikoboshi._arrays import load_core

from helpers import matrix, require


query = matrix([[1.0, 0.0], [0.0, 1.0]])
target = matrix([[1.0, 0.0], [0.0, 1.0]])

core = load_core()
require(core._accepts_float32_2d(query), "float32 2D buffer must be accepted")
info = core._buffer_info(query)
require(info["shape"] == (2, 2), "buffer shape mismatch")
require(info["float32"], "buffer dtype must be float32")

# Pin to hard mode: the literal score golden 2.0 is the hard-SW raw_sw_score
# for the identity 2x2 case. Hikoboshi 0.1.0 defaults to hard SW; this test
# continues to validate the hard-SW dtype + metadata round-trip.
result = hiko.pairwise.from_embeddings(
    query,
    target,
    query_metadata={"residue_codes": "AC"},
    target_metadata={"residue_codes": "AC"},
    mode="hard",
)
require(result.metrics.raw_sw_score == 2.0, "embedding pairwise raw score mismatch")
require(result.metrics.identity == 1.0, "metadata-backed identity mismatch")

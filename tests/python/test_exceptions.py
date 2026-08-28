from __future__ import annotations

import hikoboshi as hiko
from hikoboshi._arrays import load_core

from helpers import matrix, require


try:
    hiko.pairwise.from_embeddings(matrix([[1.0, 0.0]]), matrix([[1.0, 0.0, 0.0]]))
except hiko.InvalidArgumentError as exc:
    require(exc.code == "invalid_argument", "exception code mismatch")
    require("dimensions differ" in exc.detail, "exception detail not preserved")
else:
    raise SystemExit("dimension mismatch should raise InvalidArgumentError")

core = load_core()
payload = core.pairwise_from_embeddings(
    matrix([[1.0, 0.0], [0.0, 1.0]]),
    matrix([[1.0, 0.0], [0.0, 1.0]]),
    None,
    None,
    -2.0,
    -0.15,
)
warnings = payload.get("warnings", [])
require(
    any(warning.get("code") == "gap_defaults_overridden" for warning in warnings),
    "gap override warning missing from pybind payload",
)

try:
    hiko.encode.from_structure("missing.pdb")
except hiko.UnavailableError as exc:
    require(exc.code == "unavailable", "structure IO code mismatch")
    require("structure file not readable" in exc.detail, "structure IO detail missing")
else:
    raise SystemExit("missing structure should raise UnavailableError")

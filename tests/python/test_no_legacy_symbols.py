from __future__ import annotations

import importlib

import hikoboshi as hiko

from helpers import require


LEGACY_TOP_LEVEL = {
    "batch_pairwise",
    "compute_distances",
    "distance_matrix",
    "msa",
    "MSA",
    "tree",
    "build_tree",
    "guide_tree",
    "similarity",
    "score",
    "_align_cpp",
}

LEGACY_ALL_VS_ALL = {
    "score",
    "scores",
    "score_only",
    "compute_distances",
    "distance_matrix",
}

RESERVED_SUBSTITUTION_TOP_LEVEL = {
    "substitution",
    "substitution_matrix",
    "substitution_matrices",
    "sequence_pairwise",
}

RESERVED_SUBSTITUTION_PAIRWISE = {
    "from_sequence_tokens",
    "from_sequences",
    "substitution_matrix",
    "substitution_lookup",
}

RESERVED_SUBSTITUTION_ALL_VS_ALL = {
    "from_sequence_tokens",
    "from_sequences",
    "substitution_matrix",
    "substitution_lookup",
}


for name in LEGACY_TOP_LEVEL:
    require(not hasattr(hiko, name), f"legacy top-level symbol reappeared: {name}")

for name in LEGACY_ALL_VS_ALL:
    require(not hasattr(hiko.all_vs_all, name), f"legacy all_vs_all symbol reappeared: {name}")

for name in RESERVED_SUBSTITUTION_TOP_LEVEL:
    require(not hasattr(hiko, name), f"reserved substitution workflow exposed: {name}")

for name in RESERVED_SUBSTITUTION_PAIRWISE:
    require(not hasattr(hiko.pairwise, name), f"reserved pairwise substitution workflow exposed: {name}")

for name in RESERVED_SUBSTITUTION_ALL_VS_ALL:
    require(not hasattr(hiko.all_vs_all, name), f"reserved all_vs_all substitution workflow exposed: {name}")

try:
    importlib.import_module("hikoboshi._align_cpp")
except ModuleNotFoundError:
    pass
else:
    raise SystemExit("historical hikoboshi._align_cpp extension must not import")

public_names = set(hiko.__all__)
for forbidden in LEGACY_TOP_LEVEL:
    require(forbidden not in public_names, f"legacy symbol exported through __all__: {forbidden}")

for forbidden in RESERVED_SUBSTITUTION_TOP_LEVEL:
    require(forbidden not in public_names, f"reserved substitution symbol exported through __all__: {forbidden}")

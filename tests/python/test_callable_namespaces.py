from __future__ import annotations

from inspect import signature

import hikoboshi as hiko

from helpers import require


def require_params(callable_object, names: tuple[str, ...]) -> None:
    params = signature(callable_object).parameters
    for name in names:
        require(name in params, f"{callable_object!r} missing parameter {name}")


def reject_params(callable_object, names: tuple[str, ...]) -> None:
    params = signature(callable_object).parameters
    for name in names:
        require(name not in params, f"{callable_object!r} must not expose {name}")


require(hiko.Tensor.__name__ == "Tensor", "hikoboshi.Tensor must be exposed lazily")
for namespace in (hiko.encode, hiko.pairwise, hiko.all_vs_all):
    require(callable(namespace), f"{namespace!r} must be callable")
    for name in ("from_structure", "from_pdb", "from_cif", "from_coords"):
        require(hasattr(namespace, name), f"missing callable namespace member {name}")
        require(callable(getattr(namespace, name)), f"{name} must be callable")

require(hasattr(hiko.pairwise, "from_embeddings"), "pairwise.from_embeddings missing")
require(callable(hiko.pairwise.from_embeddings), "pairwise.from_embeddings not callable")
require(hasattr(hiko.all_vs_all, "from_embeddings"), "all_vs_all.from_embeddings missing")
require(callable(hiko.all_vs_all.from_embeddings), "all_vs_all.from_embeddings not callable")
for route in (
    hiko.encode.from_structure,
    hiko.encode.from_pdb,
    hiko.encode.from_cif,
    hiko.pairwise.from_structure,
    hiko.pairwise.from_pdb,
    hiko.pairwise.from_cif,
    hiko.all_vs_all.from_structure,
    hiko.all_vs_all.from_pdb,
    hiko.all_vs_all.from_cif,
):
    require_params(route, ("package", "pdb_model_id", "pdb_model_index"))
    reject_params(route, ("model_id", "model_index"))
for route in (
    hiko.encode.from_coords,
    hiko.pairwise.from_embeddings,
    hiko.all_vs_all.from_coords,
    hiko.all_vs_all.from_embeddings,
):
    require_params(route, ("package",))
for route in (
    hiko.pairwise.from_structure,
    hiko.pairwise.from_pdb,
    hiko.pairwise.from_cif,
    hiko.pairwise.from_coords,
):
    require_params(route, ("package", "gap_open", "gap_extension"))
require(not hasattr(hiko, "similarity"), "top-level similarity must not be exposed")
require(not hasattr(hiko, "score"), "top-level score must not be exposed")
require(not hasattr(hiko, "load_package"), "external package loading must not be exposed")
require(not hasattr(hiko, "from_package"), "package selection workflow must not be exposed")
require(not hasattr(hiko.all_vs_all, "score_only"), "score-only all_vs_all must not exist")
require(not hasattr(hiko.pairwise, "score_only"), "score-only pairwise must not exist")

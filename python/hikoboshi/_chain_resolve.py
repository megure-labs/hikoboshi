from __future__ import annotations

import warnings
from typing import Any

from .exceptions import InvalidArgumentError

_DEPRECATED_PACKAGE_ALIASES = {
    "mpnn64": "hikoboshi-mpnn-d64",
    "mpnn-64": "hikoboshi-mpnn-d64",
    "hikoboshi-mpnn-64": "hikoboshi-mpnn-d64",
    "esm2-8m": "hikoboshi-esm2-8m",
    "esm2_8m": "hikoboshi-esm2-8m",
    "hikoboshi-esm2-8m": "hikoboshi-esm2-8m",
    "v_48_020": "proteinmpnn-v48-eps020",
    "hikoboshi-proteinmpnn-v48-020": "proteinmpnn-v48-eps020",
    "proteinmpnn-v48-020": "proteinmpnn-v48-eps020",
    "proteinmpnn": "proteinmpnn-v48-eps020",
}
_WARNED_PACKAGE_ALIASES: set[str] = set()


def reject_unknown_kwargs(kwargs: dict[str, Any], *, context: str) -> None:
    if not kwargs:
        return
    names = ", ".join(sorted(kwargs))
    raise InvalidArgumentError(f"unknown {context} keyword(s): {names}")


def resolve_deprecated_model_kwargs(
    kwargs: dict[str, Any],
    *,
    pdb_model_id: str | None,
    pdb_model_index: int | None,
    context: str,
) -> tuple[str | None, int | None]:
    if "model_id" in kwargs:
        if pdb_model_id is not None:
            raise InvalidArgumentError(
                f"choose either pdb_model_id or deprecated model_id for {context}, "
                "not both"
            )
        pdb_model_id = kwargs.pop("model_id")
        warnings.warn(
            "model_id is deprecated; use pdb_model_id",
            DeprecationWarning,
            stacklevel=3,
        )
    if "model_index" in kwargs:
        if pdb_model_index is not None:
            raise InvalidArgumentError(
                "choose either pdb_model_index or deprecated model_index for "
                f"{context}, not both"
            )
        pdb_model_index = kwargs.pop("model_index")
        warnings.warn(
            "model_index is deprecated; use pdb_model_index",
            DeprecationWarning,
            stacklevel=3,
        )
    return pdb_model_id, pdb_model_index


def warn_deprecated_package_alias(package: str | None) -> None:
    if package is None:
        return
    lookup = package.casefold()
    canonical = _DEPRECATED_PACKAGE_ALIASES.get(lookup)
    if canonical is None or package == canonical or lookup in _WARNED_PACKAGE_ALIASES:
        return
    _WARNED_PACKAGE_ALIASES.add(lookup)
    warnings.warn(
        f"package alias {package!r} is deprecated; use {canonical!r}",
        DeprecationWarning,
        stacklevel=3,
    )


def chain_selection(
    *,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
) -> dict[str, Any]:
    if chain_id is not None and chain_index is not None:
        raise InvalidArgumentError("choose either chain_id or chain_index, not both")
    if pdb_model_id is not None and pdb_model_index is not None:
        raise InvalidArgumentError(
            "choose either pdb_model_id or pdb_model_index, not both"
        )
    selection: dict[str, Any] = {}
    if chain_id is not None:
        selection["chain_id"] = chain_id
    if chain_index is not None:
        selection["chain_index"] = chain_index
    if pdb_model_id is not None:
        selection["pdb_model_id"] = pdb_model_id
    if pdb_model_index is not None:
        selection["pdb_model_index"] = pdb_model_index
    return selection


__all__ = [
    "chain_selection",
    "reject_unknown_kwargs",
    "resolve_deprecated_model_kwargs",
    "warn_deprecated_package_alias",
]

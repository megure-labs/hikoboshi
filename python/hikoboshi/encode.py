from __future__ import annotations

from typing import Any

from ._arrays import load_core
from ._chain_resolve import (
    chain_selection,
    reject_unknown_kwargs,
    resolve_deprecated_model_kwargs,
    warn_deprecated_package_alias,
)
from .exceptions import raise_from_core
from .results import EncodeResult


class EncodeNamespace:
    def __call__(self, structure: Any, **kwargs: Any) -> EncodeResult:
        return self.from_structure(structure, **kwargs)

    def from_structure(
        self,
        structure: Any,
        *,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        package: str | None = None,
        **kwargs: Any,
    ) -> EncodeResult:
        pdb_model_id, pdb_model_index = resolve_deprecated_model_kwargs(
            kwargs,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            context="encode.from_structure",
        )
        reject_unknown_kwargs(kwargs, context="encode.from_structure")
        warn_deprecated_package_alias(package)
        core = load_core()
        selection = chain_selection(
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
        )
        try:
            payload = core.encode_from_structure(
                structure,
                package=package,
                **selection,
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return EncodeResult(payload.get("embeddings"), payload.get("metadata"))

    def from_pdb(
        self,
        structure: Any,
        *,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        package: str | None = None,
        **kwargs: Any,
    ) -> EncodeResult:
        return self.from_structure(
            structure,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            package=package,
            **kwargs,
        )

    def from_cif(
        self,
        structure: Any,
        *,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        package: str | None = None,
        **kwargs: Any,
    ) -> EncodeResult:
        return self.from_structure(
            structure,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            package=package,
            **kwargs,
        )

    def from_coords(
        self,
        coords: Any,
        *,
        package: str | None = None,
        **kwargs: Any,
    ) -> EncodeResult:
        reject_unknown_kwargs(kwargs, context="encode.from_coords")
        warn_deprecated_package_alias(package)
        core = load_core()
        try:
            payload = core.encode_from_coords(coords, package=package)
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return EncodeResult(payload.get("embeddings"), payload.get("metadata"))


encode = EncodeNamespace()

__all__ = ["EncodeNamespace", "encode"]

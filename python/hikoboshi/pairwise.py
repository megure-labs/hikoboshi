from __future__ import annotations

from typing import Any

from ._arrays import as_float32_2d, load_core, normalize_metadata
from ._chain_resolve import (
    chain_selection,
    reject_unknown_kwargs,
    resolve_deprecated_model_kwargs,
    warn_deprecated_package_alias,
)
from ._input_detect import detect_pairwise_kind
from .exceptions import raise_from_core
from .results import PairwiseResult


def _gap_kwargs(gap_open: float | None, gap_extension: float | None) -> dict[str, float]:
    kwargs: dict[str, float] = {}
    if gap_open is not None:
        kwargs["gap_open"] = float(gap_open)
    if gap_extension is not None:
        kwargs["gap_extension"] = float(gap_extension)
    return kwargs


class PairwiseNamespace:
    def __call__(self, query: Any, target: Any, **kwargs: Any) -> PairwiseResult:
        if detect_pairwise_kind(query, target) == "embedding":
            return self.from_embeddings(query, target, **kwargs)
        return self.from_structure(query, target, **kwargs)

    def from_embeddings(
        self,
        query_embeddings: Any,
        target_embeddings: Any,
        *,
        query_metadata: dict[str, Any] | None = None,
        target_metadata: dict[str, Any] | None = None,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
    ) -> PairwiseResult:
        query = as_float32_2d(query_embeddings, name="query_embeddings")
        target = as_float32_2d(target_embeddings, name="target_embeddings")
        query_meta = normalize_metadata(query_metadata, int(query.shape[0]))
        target_meta = normalize_metadata(target_metadata, int(target.shape[0]))
        warn_deprecated_package_alias(package)
        core = load_core()
        try:
            payload = core.pairwise_from_embeddings(
                query,
                target,
                query_metadata=query_meta,
                target_metadata=target_meta,
                package=package,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return PairwiseResult.from_core(
            payload,
            query_metadata=query_meta,
            target_metadata=target_meta,
        )

    def from_structure(
        self,
        query: Any,
        target: Any,
        *,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> PairwiseResult:
        pdb_model_id, pdb_model_index = resolve_deprecated_model_kwargs(
            kwargs,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            context="pairwise.from_structure",
        )
        reject_unknown_kwargs(kwargs, context="pairwise.from_structure")
        warn_deprecated_package_alias(package)
        core = load_core()
        selection = chain_selection(
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
        )
        try:
            payload = core.pairwise_from_structure(
                query,
                target,
                package=package,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
                **selection,
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return PairwiseResult.from_core(payload)

    def from_pdb(
        self,
        query: Any,
        target: Any,
        *,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> PairwiseResult:
        return self.from_structure(
            query,
            target,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            gap_open=gap_open,
            gap_extension=gap_extension,
            package=package,
            mode=mode,
            temperature=temperature,
            **kwargs,
        )

    def from_cif(
        self,
        query: Any,
        target: Any,
        *,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> PairwiseResult:
        return self.from_structure(
            query,
            target,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            gap_open=gap_open,
            gap_extension=gap_extension,
            package=package,
            mode=mode,
            temperature=temperature,
            **kwargs,
        )

    def from_coords(
        self,
        query_coords: Any,
        target_coords: Any,
        *,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> PairwiseResult:
        reject_unknown_kwargs(kwargs, context="pairwise.from_coords")
        warn_deprecated_package_alias(package)
        core = load_core()
        try:
            payload = core.pairwise_from_coords(
                query_coords,
                target_coords,
                package=package,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return PairwiseResult.from_core(payload)


pairwise = PairwiseNamespace()

__all__ = ["PairwiseNamespace", "pairwise"]

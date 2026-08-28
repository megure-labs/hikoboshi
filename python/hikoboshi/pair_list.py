from __future__ import annotations

from typing import Any, Sequence

from ._arrays import as_float32_2d, load_core, normalize_metadata
from ._chain_resolve import (
    chain_selection,
    reject_unknown_kwargs,
    resolve_deprecated_model_kwargs,
    warn_deprecated_package_alias,
)
from .exceptions import InvalidArgumentError, raise_from_core
from .results import PairwiseResult

_MAX_THREAD_COUNT = 2**32 - 1
PairList = Sequence[tuple[str, str]]


def _normalize_thread_count(thread_count: int, *, name: str) -> int:
    if isinstance(thread_count, bool) or not isinstance(thread_count, int):
        raise InvalidArgumentError(f"{name} must be a non-negative integer")
    if thread_count < 0:
        raise InvalidArgumentError(f"{name} must be a non-negative integer")
    if thread_count > _MAX_THREAD_COUNT:
        raise InvalidArgumentError(f"{name} is outside the supported range")
    return thread_count


def _resolve_threads(threads: int, thread_count: int | None) -> int:
    if thread_count is not None:
        if threads != 0:
            raise InvalidArgumentError("use either threads or thread_count, not both")
        return _normalize_thread_count(thread_count, name="thread_count")
    return _normalize_thread_count(threads, name="threads")


def _results_from_core(payload: Sequence[dict[str, Any]]) -> list[PairwiseResult]:
    return [PairwiseResult.from_core(item) for item in payload]


def _gap_kwargs(gap_open: float | None, gap_extension: float | None) -> dict[str, float]:
    kwargs: dict[str, float] = {}
    if gap_open is not None:
        kwargs["gap_open"] = float(gap_open)
    if gap_extension is not None:
        kwargs["gap_extension"] = float(gap_extension)
    return kwargs


def pair_list_from_sequence(
    pairs: PairList,
    fasta_path: Any,
    *,
    package: str | None = None,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    threads: int = 0,
    thread_count: int | None = None,
    mode: str = "hard",
    temperature: float = 1.0,
) -> list[PairwiseResult]:
    kwargs: dict[str, Any] = {
        "package": package,
        "threads": _resolve_threads(threads, thread_count),
        "mode": mode,
        "temperature": float(temperature),
    }
    kwargs.update(_gap_kwargs(gap_open, gap_extension))
    warn_deprecated_package_alias(package)
    core = load_core()
    try:
        payload = core.pair_list_from_sequence(pairs, fasta_path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
        raise_from_core(exc)
    return _results_from_core(payload)


def pair_list_from_structure(
    pairs: PairList,
    pdb_dir: Any,
    *,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    threads: int = 0,
    thread_count: int | None = None,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
    mode: str = "hard",
    temperature: float = 1.0,
    **kwargs: Any,
) -> list[PairwiseResult]:
    pdb_model_id, pdb_model_index = resolve_deprecated_model_kwargs(
        kwargs,
        pdb_model_id=pdb_model_id,
        pdb_model_index=pdb_model_index,
        context="pair_list_from_structure",
    )
    reject_unknown_kwargs(kwargs, context="pair_list_from_structure")
    selection = chain_selection(
        chain_id=chain_id,
        chain_index=chain_index,
        pdb_model_id=pdb_model_id,
        pdb_model_index=pdb_model_index,
    )
    warn_deprecated_package_alias(package)
    core = load_core()
    try:
        payload = core.pair_list_from_structure(
            pairs,
            pdb_dir,
            package=package,
            threads=_resolve_threads(threads, thread_count),
            mode=mode,
            temperature=float(temperature),
            **_gap_kwargs(gap_open, gap_extension),
            **selection,
        )
    except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
        raise_from_core(exc)
    return _results_from_core(payload)


def pair_list_from_coords(
    pairs: PairList,
    coords: Sequence[Any],
    *,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    threads: int = 0,
    thread_count: int | None = None,
    mode: str = "hard",
    temperature: float = 1.0,
    **kwargs: Any,
) -> list[PairwiseResult]:
    reject_unknown_kwargs(kwargs, context="pair_list_from_coords")
    warn_deprecated_package_alias(package)
    core = load_core()
    try:
        payload = core.pair_list_from_coords(
            pairs,
            coords,
            package=package,
            threads=_resolve_threads(threads, thread_count),
            mode=mode,
            temperature=float(temperature),
            **_gap_kwargs(gap_open, gap_extension),
        )
    except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
        raise_from_core(exc)
    return _results_from_core(payload)


def pair_list_from_embeddings(
    pairs: PairList,
    embeddings: Sequence[Any],
    *,
    metadata: Sequence[dict[str, Any] | None] | None = None,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    threads: int = 0,
    thread_count: int | None = None,
    mode: str = "hard",
    temperature: float = 1.0,
) -> list[PairwiseResult]:
    arrays = [
        as_float32_2d(array, name=f"embeddings[{index}]")
        for index, array in enumerate(embeddings)
    ]
    if metadata is None:
        normalized_metadata = tuple(None for _ in arrays)
    else:
        if len(metadata) != len(arrays):
            raise InvalidArgumentError("metadata length must match embeddings")
        normalized_metadata = tuple(
            normalize_metadata(item, int(array.shape[0]))
            for item, array in zip(metadata, arrays)
        )
    warn_deprecated_package_alias(package)
    core = load_core()
    try:
        payload = core.pair_list_from_embeddings(
            pairs,
            arrays,
            metadata=normalized_metadata,
            package=package,
            threads=_resolve_threads(threads, thread_count),
            mode=mode,
            temperature=float(temperature),
            **_gap_kwargs(gap_open, gap_extension),
        )
    except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
        raise_from_core(exc)
    return _results_from_core(payload)


__all__ = [
    "PairList",
    "pair_list_from_embeddings",
    "pair_list_from_sequence",
    "pair_list_from_structure",
    "pair_list_from_coords",
]

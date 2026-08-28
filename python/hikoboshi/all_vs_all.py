from __future__ import annotations

import os
from typing import Any, Sequence

from ._arrays import as_float32_2d, load_core, normalize_metadata
from ._chain_resolve import (
    chain_selection,
    reject_unknown_kwargs,
    resolve_deprecated_model_kwargs,
    warn_deprecated_package_alias,
)
from ._input_detect import detect_input_kind
from .exceptions import InvalidArgumentError, raise_from_core
from .results import AllVsAllResult, AllVsAllStreamingSummary


_MAX_THREAD_COUNT = 2**32 - 1


def _normalize_thread_count(thread_count: int) -> int:
    if isinstance(thread_count, bool) or not isinstance(thread_count, int):
        raise InvalidArgumentError("thread_count must be a non-negative integer")
    if thread_count < 0:
        raise InvalidArgumentError("thread_count must be a non-negative integer")
    if thread_count > _MAX_THREAD_COUNT:
        raise InvalidArgumentError("thread_count is outside the supported range")
    return thread_count


def _normalize_output_path(output_path: Any) -> str:
    if output_path is None:
        raise InvalidArgumentError("output_path is required")
    text = os.fspath(output_path)
    if isinstance(text, bytes):
        text = text.decode()
    if not text:
        raise InvalidArgumentError("output_path must not be empty")
    return text


def _gap_kwargs(gap_open: float | None, gap_extension: float | None) -> dict[str, float]:
    kwargs: dict[str, float] = {}
    if gap_open is not None:
        kwargs["gap_open"] = float(gap_open)
    if gap_extension is not None:
        kwargs["gap_extension"] = float(gap_extension)
    return kwargs


class AllVsAllNamespace:
    def __call__(self, inputs: Sequence[Any], **kwargs: Any) -> AllVsAllResult:
        first_kind = detect_input_kind(inputs[0]) if inputs else "embedding"
        if first_kind == "embedding":
            return self.from_embeddings(inputs, **kwargs)
        return self.from_structure(inputs, **kwargs)

    def from_embeddings(
        self,
        embeddings: Sequence[Any],
        *,
        metadata: Sequence[dict[str, Any] | None] | None = None,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        mode: str = "hard",
        temperature: float = 1.0,
    ) -> AllVsAllResult:
        normalized_thread_count = _normalize_thread_count(thread_count)
        arrays = [
            as_float32_2d(array, name=f"embeddings[{index}]")
            for index, array in enumerate(embeddings)
        ]
        if metadata is None:
            normalized_metadata = tuple(None for _ in arrays)
        else:
            if len(metadata) != len(arrays):
                from .exceptions import InvalidArgumentError

                raise InvalidArgumentError("metadata length must match embeddings")
            normalized_metadata = tuple(
                normalize_metadata(item, int(array.shape[0]))
                for item, array in zip(metadata, arrays)
            )
        warn_deprecated_package_alias(package)
        core = load_core()
        try:
            payload = core.all_vs_all_from_embeddings(
                arrays,
                metadata=normalized_metadata,
                include_self=bool(include_self),
                package=package,
                thread_count=normalized_thread_count,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return AllVsAllResult.from_core(payload, metadata=normalized_metadata)

    def from_structure(
        self,
        inputs: Sequence[Any],
        *,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AllVsAllResult:
        pdb_model_id, pdb_model_index = resolve_deprecated_model_kwargs(
            kwargs,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            context="all_vs_all.from_structure",
        )
        reject_unknown_kwargs(kwargs, context="all_vs_all.from_structure")
        normalized_thread_count = _normalize_thread_count(thread_count)
        warn_deprecated_package_alias(package)
        core = load_core()
        selection = chain_selection(
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
        )
        try:
            payload = core.all_vs_all_from_structure(
                inputs,
                include_self=bool(include_self),
                package=package,
                thread_count=normalized_thread_count,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
                **selection,
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return AllVsAllResult.from_core(payload)

    def from_pdb(
        self,
        inputs: Sequence[Any],
        *,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AllVsAllResult:
        return self.from_structure(
            inputs,
            include_self=include_self,
            gap_open=gap_open,
            gap_extension=gap_extension,
            package=package,
            thread_count=thread_count,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            mode=mode,
            temperature=temperature,
            **kwargs,
        )

    def from_cif(
        self,
        inputs: Sequence[Any],
        *,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AllVsAllResult:
        return self.from_structure(
            inputs,
            include_self=include_self,
            gap_open=gap_open,
            gap_extension=gap_extension,
            package=package,
            thread_count=thread_count,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            mode=mode,
            temperature=temperature,
            **kwargs,
        )

    def from_coords(
        self,
        coords: Sequence[Any],
        *,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AllVsAllResult:
        reject_unknown_kwargs(kwargs, context="all_vs_all.from_coords")
        normalized_thread_count = _normalize_thread_count(thread_count)
        warn_deprecated_package_alias(package)
        core = load_core()
        try:
            payload = core.all_vs_all_from_coords(
                coords,
                include_self=bool(include_self),
                package=package,
                thread_count=normalized_thread_count,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return AllVsAllResult.from_core(payload)

    def to_tsv(
        self,
        inputs: Sequence[Any],
        output_path: Any,
        **kwargs: Any,
    ) -> AllVsAllStreamingSummary:
        first_kind = detect_input_kind(inputs[0]) if inputs else "embedding"
        if first_kind == "embedding":
            return self.to_tsv_from_embeddings(inputs, output_path, **kwargs)
        return self.to_tsv_from_structure(inputs, output_path, **kwargs)

    def to_tsv_from_embeddings(
        self,
        embeddings: Sequence[Any],
        output_path: Any,
        *,
        metadata: Sequence[dict[str, Any] | None] | None = None,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        mode: str = "hard",
        temperature: float = 1.0,
    ) -> AllVsAllStreamingSummary:
        normalized_path = _normalize_output_path(output_path)
        normalized_thread_count = _normalize_thread_count(thread_count)
        arrays = [
            as_float32_2d(array, name=f"embeddings[{index}]")
            for index, array in enumerate(embeddings)
        ]
        if metadata is None:
            normalized_metadata: tuple[Any, ...] = tuple(None for _ in arrays)
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
            payload = core.all_vs_all_to_tsv_from_embeddings(
                arrays,
                normalized_path,
                metadata=normalized_metadata,
                include_self=bool(include_self),
                package=package,
                thread_count=normalized_thread_count,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return AllVsAllStreamingSummary.from_core(payload)

    def to_tsv_from_structure(
        self,
        inputs: Sequence[Any],
        output_path: Any,
        *,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AllVsAllStreamingSummary:
        pdb_model_id, pdb_model_index = resolve_deprecated_model_kwargs(
            kwargs,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            context="all_vs_all.to_tsv_from_structure",
        )
        reject_unknown_kwargs(kwargs, context="all_vs_all.to_tsv_from_structure")
        normalized_path = _normalize_output_path(output_path)
        normalized_thread_count = _normalize_thread_count(thread_count)
        warn_deprecated_package_alias(package)
        core = load_core()
        selection = chain_selection(
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
        )
        try:
            payload = core.all_vs_all_to_tsv_from_structure(
                inputs,
                normalized_path,
                include_self=bool(include_self),
                package=package,
                thread_count=normalized_thread_count,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
                **selection,
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return AllVsAllStreamingSummary.from_core(payload)

    def to_tsv_from_pdb(
        self,
        inputs: Sequence[Any],
        output_path: Any,
        **kwargs: Any,
    ) -> AllVsAllStreamingSummary:
        return self.to_tsv_from_structure(inputs, output_path, **kwargs)

    def to_tsv_from_cif(
        self,
        inputs: Sequence[Any],
        output_path: Any,
        **kwargs: Any,
    ) -> AllVsAllStreamingSummary:
        return self.to_tsv_from_structure(inputs, output_path, **kwargs)

    def to_tsv_from_coords(
        self,
        coords: Sequence[Any],
        output_path: Any,
        *,
        include_self: bool = False,
        gap_open: float | None = None,
        gap_extension: float | None = None,
        package: str | None = None,
        thread_count: int = 0,
        mode: str = "hard",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AllVsAllStreamingSummary:
        reject_unknown_kwargs(kwargs, context="all_vs_all.to_tsv_from_coords")
        normalized_path = _normalize_output_path(output_path)
        normalized_thread_count = _normalize_thread_count(thread_count)
        warn_deprecated_package_alias(package)
        core = load_core()
        try:
            payload = core.all_vs_all_to_tsv_from_coords(
                coords,
                normalized_path,
                include_self=bool(include_self),
                package=package,
                thread_count=normalized_thread_count,
                mode=mode,
                temperature=float(temperature),
                **_gap_kwargs(gap_open, gap_extension),
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return AllVsAllStreamingSummary.from_core(payload)


all_vs_all = AllVsAllNamespace()

__all__ = ["AllVsAllNamespace", "all_vs_all"]

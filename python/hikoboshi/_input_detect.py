from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Literal

InputKind = Literal["structure", "embedding", "coords", "unknown"]

_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
_EMBEDDING_SUFFIXES = {".npy", ".npz"}
_FLOAT32_FORMATS = {"f", "<f", "=f", "@f"}


def _buffer_input_kind(value: Any) -> InputKind:
    try:
        view = memoryview(value)
    except TypeError:
        return "unknown"
    if view.ndim == 2 and view.format in _FLOAT32_FORMATS:
        return "embedding"
    return "unknown"


def detect_input_kind(value: Any) -> InputKind:
    if isinstance(value, (str, PathLike)):
        suffix = Path(value).suffix.lower()
        if suffix in _STRUCTURE_SUFFIXES:
            return "structure"
        if suffix in _EMBEDDING_SUFFIXES:
            return "embedding"
        return "unknown"
    if hasattr(value, "__array__") or hasattr(value, "__array_interface__"):
        return "embedding"
    return _buffer_input_kind(value)


def detect_pairwise_kind(query: Any, target: Any) -> InputKind:
    query_kind = detect_input_kind(query)
    target_kind = detect_input_kind(target)
    if query_kind == target_kind:
        return query_kind
    return "unknown"


__all__ = ["InputKind", "detect_input_kind", "detect_pairwise_kind"]

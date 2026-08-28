from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from .exceptions import InvalidArgumentError

_FLOAT32_FORMATS = {"f", "<f", "=f", "@f"}


@lru_cache(maxsize=1)
def load_core() -> ModuleType:
    try:
        return importlib.import_module("hikoboshi._core")
    except ModuleNotFoundError as original:
        build_root = os.environ.get("HIKOBOSHI_BUILD_ROOT")
        if not build_root:
            raise original
        matches = sorted(Path(build_root).glob("_core*.so"))
        if len(matches) != 1:
            raise original
        spec = importlib.util.spec_from_file_location("hikoboshi._core", matches[0])
        if spec is None or spec.loader is None:
            raise original
        module = importlib.util.module_from_spec(spec)
        sys.modules["hikoboshi._core"] = module
        spec.loader.exec_module(module)
        return module


def _numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise InvalidArgumentError("NumPy is required for array inputs") from exc
    return np


def as_float32_2d(value: Any, *, name: str) -> Any:
    try:
        np = _numpy()
    except InvalidArgumentError:
        view = memoryview(value)
        if view.ndim != 2 or view.format not in _FLOAT32_FORMATS:
            raise InvalidArgumentError(f"{name} must be a 2D float32 array")
        if view.shape[0] == 0 or view.shape[1] == 0:
            raise InvalidArgumentError(f"{name} must be non-empty")
        if not view.contiguous:
            raise InvalidArgumentError(f"{name} must be contiguous")
        return view
    else:
        array = np.asarray(value, dtype=np.float32, order="C")
        if array.ndim != 2:
            raise InvalidArgumentError(f"{name} must be a 2D float32 array")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise InvalidArgumentError(f"{name} must be non-empty")
        return np.ascontiguousarray(array, dtype=np.float32)


def as_float32_coords(value: Any, *, name: str) -> Any:
    try:
        np = _numpy()
    except InvalidArgumentError:
        view = memoryview(value)
        if view.ndim not in {2, 3} or view.format not in _FLOAT32_FORMATS:
            raise InvalidArgumentError(f"{name} must be a coordinate array")
        if not view.contiguous:
            raise InvalidArgumentError(f"{name} must be contiguous")
        return view
    else:
        array = np.asarray(value, dtype=np.float32, order="C")
        if array.ndim not in {2, 3}:
            raise InvalidArgumentError(f"{name} must be a coordinate array")
        return np.ascontiguousarray(array, dtype=np.float32)


def normalize_metadata(metadata: dict[str, Any] | None, residue_count: int) -> dict[str, Any] | None:
    if metadata is None:
        return None
    normalized = dict(metadata)
    residue_codes = normalized.get("residue_codes")
    if residue_codes is not None:
        if isinstance(residue_codes, str):
            if len(residue_codes) != residue_count:
                raise InvalidArgumentError("residue_codes length must match embeddings")
        else:
            codes = "".join(str(code)[0] for code in residue_codes)
            if len(codes) != residue_count:
                raise InvalidArgumentError("residue_codes length must match embeddings")
            normalized["residue_codes"] = codes
    return normalized


__all__ = [
    "as_float32_2d",
    "as_float32_coords",
    "load_core",
    "normalize_metadata",
]

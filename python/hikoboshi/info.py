from __future__ import annotations

from typing import Any

from ._arrays import load_core
from .exceptions import raise_from_core


def version_info() -> dict[str, Any]:
    core = load_core()
    try:
        return dict(core.version_info())
    except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
        raise_from_core(exc)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _threading_info() -> dict[str, Any]:
    return {"compiled": True, "default_mode": "auto"}


def info() -> dict[str, Any]:
    core = load_core()
    try:
        backend_capabilities = _plain(dict(core.backend_capabilities()))
        models = _plain(dict(core.compiled_models_info()))
        threading = _threading_info()
        return {
            "version": _plain(dict(core.version_info())),
            "backends": _plain(backend_capabilities),
            "backend_capabilities": backend_capabilities,
            "threading": threading,
            "weights": _plain(dict(core.default_weights_info())),
            "models": models,
        }
    except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
        raise_from_core(exc)


__all__ = ["info", "version_info"]

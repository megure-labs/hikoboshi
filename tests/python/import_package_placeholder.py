from __future__ import annotations

import importlib
from pathlib import Path

module = importlib.import_module("hikoboshi")
module_path = Path(module.__file__).as_posix()

if not module_path.endswith("python/hikoboshi/__init__.py"):
    raise SystemExit(f"unexpected hikoboshi package path: {module_path}")

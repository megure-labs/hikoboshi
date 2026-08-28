from __future__ import annotations

import importlib.util
import os
from pathlib import Path

build_root = Path(os.environ["HIKOBOSHI_BUILD_ROOT"])
matches = sorted(build_root.glob("_core*.so"))

if len(matches) != 1:
    raise SystemExit(f"expected one _core extension, found {len(matches)}")

spec = importlib.util.spec_from_file_location("hikoboshi._core", matches[0])
if spec is None or spec.loader is None:
    raise SystemExit(f"could not load extension spec from {matches[0]}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

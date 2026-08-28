from __future__ import annotations

import os
from pathlib import Path

from hikoboshi._arrays import load_core

from helpers import require


build_root = Path(os.environ["HIKOBOSHI_BUILD_ROOT"])
matches = sorted(build_root.glob("_core*.so"))
require(len(matches) == 1, f"expected one _core extension, found {len(matches)}")
require("_align_cpp" not in matches[0].name, "extension filename uses legacy name")

module = load_core()
require(module.__name__ == "hikoboshi._core", "extension module name mismatch")
require(module.__version__ == "0.1.0", "extension version mismatch")
require(not hasattr(module, "_compute_distances_from_embeddings"), "legacy distance binding exists")

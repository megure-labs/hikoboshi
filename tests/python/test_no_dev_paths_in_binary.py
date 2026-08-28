#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from helpers import require


FORBIDDEN_MARKERS = (
    "hikoboshi-archive",
    "archive/hikoboshi/_core/src/hikoboshi/weights",
)


def strings(path: Path) -> str:
    result = subprocess.run(
        ["strings", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)
    return result.stdout


def require_no_forbidden_markers(path: Path) -> None:
    output = strings(path)
    for marker in FORBIDDEN_MARKERS:
        require(marker not in output, f"{path.name} contains forbidden marker {marker!r}")


def main() -> int:
    build_root = Path(os.environ["HIKOBOSHI_BUILD_ROOT"])
    core_extensions = sorted(build_root.glob("_core*.so"))
    require(len(core_extensions) == 1, f"expected one _core extension, found {len(core_extensions)}")

    targets = [core_extensions[0], build_root / "libhikoboshi_weights.a"]
    for target in targets:
        require(target.is_file(), f"missing binary scan target: {target}")
        require_no_forbidden_markers(target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

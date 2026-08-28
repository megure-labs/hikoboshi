from __future__ import annotations

import os
import subprocess
from pathlib import Path

binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
result = subprocess.run(
    [binary, "--version"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

if "hikoboshi 0.1.0 placeholder" not in result.stdout:
    raise SystemExit(f"unexpected CLI output: {result.stdout!r}")

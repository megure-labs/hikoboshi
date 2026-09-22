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

if result.stdout != "Hikoboshi 0.1.1\n":
    raise SystemExit(f"unexpected CLI output: {result.stdout!r}")

command_version = subprocess.run(
    [binary, "version"], check=True, text=True, capture_output=True
)
if command_version.stdout != result.stdout:
    raise SystemExit("--version and version disagree")

#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def run_info(*args: str) -> subprocess.CompletedProcess[str]:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
    return subprocess.run(
        [binary, "info", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_tsv(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("\t")
        require(bool(sep), f"line is not tab-separated: {line!r}")
        parsed[key] = value
    return parsed


def require_availability(
    fields: dict[str, str],
    prefix: str,
    *,
    compiled: str,
    runtime_available: str,
) -> None:
    require(fields[f"{prefix}.compiled"] == compiled, f"{prefix} compiled mismatch")
    require(
        fields[f"{prefix}.runtime_available"] == runtime_available,
        f"{prefix} runtime availability mismatch",
    )
    require(f"{prefix}.reason" in fields, f"{prefix} missing reason")
    if compiled == "no":
        require(fields[f"{prefix}.reason"], f"{prefix} reserved reason missing")


def main() -> int:
    result = run_info("backends")
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)
    fields = parse_tsv(result.stdout)

    require(fields["default_backend"] == "scalar", "default backend mismatch")
    require("backend_cpu_scalar_compiled" not in fields, "flat CPU key leaked")
    require("backend_default" not in fields, "flat default backend key leaked")

    require_availability(
        fields, "cpu.scalar", compiled="yes", runtime_available="yes"
    )
    for name in ("sse4", "avx2", "avx512", "neon", "sve"):
        require_availability(
            fields, f"cpu.{name}", compiled="no", runtime_available="no"
        )

    for name in ("cuda", "hip", "metal", "vulkan", "opencl"):
        prefix = f"gpu.{name}"
        require_availability(fields, prefix, compiled="no", runtime_available="no")
        require(fields[f"{prefix}.devices.count"] == "0", f"{prefix} devices leaked")

    require(fields["pipeline.pairwise_alignment"] == "yes", "pairwise flag mismatch")
    require(
        fields["pipeline.symmetric_all_vs_all"] == "yes",
        "all-vs-all flag mismatch",
    )
    require(fields["pipeline.structure_inputs"] == "yes", "structure flag mismatch")
    require(fields["pipeline.embedding_inputs"] == "yes", "embedding flag mismatch")
    require(fields["threading.compiled"] == "yes", "threading compiled mismatch")
    require(fields["threading.default_mode"] == "auto", "threading default mismatch")

    bare = run_info()
    if bare.returncode != 0:
        raise SystemExit(bare.stdout + bare.stderr)
    bare_lines = set(bare.stdout.splitlines())
    require(
        "backend_cpu_scalar_compiled\tyes" in bare_lines,
        "bare info lost legacy scalar backend summary",
    )
    require(
        "default_weights\thikoboshi-mpnn-d64" in bare_lines,
        "bare info lost default weights summary",
    )
    require(
        "threading_compiled\tyes" in bare_lines,
        "bare info lost threading summary",
    )
    require(
        "threading_default_mode\tauto" in bare_lines,
        "bare info lost threading default mode",
    )

    models = run_info("models")
    if models.returncode != 0:
        raise SystemExit(models.stdout + models.stderr)
    require(
        models.stdout.strip() == "models\tpackage registry listing lands in m15",
        "models placeholder mismatch",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


HIKOBOSHI_010_CPU_BACKENDS = ("scalar", "sse4", "avx2", "avx512", "neon", "sve")
HIKOBOSHI_010_GPU_BACKENDS = ("cuda", "hip", "metal", "vulkan", "opencl")
HIKOBOSHI_010_PIPELINE_FLAGS = {
    "pipeline.pairwise_alignment": "yes",
    "pipeline.symmetric_all_vs_all": "yes",
    "pipeline.structure_inputs": "yes",
    "pipeline.embedding_inputs": "yes",
}
HIKOBOSHI_010_THREADING = {
    "threading.compiled": "yes",
    "threading.default_mode": "auto",
}
HIKOBOSHI_010_MPNN64_MODEL = {
    "package_id": "hikoboshi-mpnn-d64",
    "aliases": ("mpnn64", "mpnn-64", "Hikoboshi-MPNN-64"),
    "family": "Hikoboshi-MPNN",
    "package_kind": "registered_architecture",
    "execution_mode": "registered_architecture",
    "architecture_id": "hikoboshi_mpnn_v1",
    "hidden_dimension": "64",
    "scoring.method": "raw_dot_v1",
    "gaps.family": "Hikoboshi 0.1.0 hard-SW",
    "gaps.gap_open": "-1.4",
    "gaps.gap_extension": "-0.15",
    "soft_gaps.family": "Hikoboshi 0.1.0 soft-SW",
    "soft_gaps.gap_open": "-3.21337",
    "soft_gaps.gap_extension": "-0.111704",
    "alignment.algorithm": "hard_local_affine_sw_v1",
    "availability.runtime_available": "yes",
}
HIKOBOSHI_010_ESM2_8M_MODEL = {
    "package_id": "hikoboshi-esm2-8m",
    "aliases": ("esm2-8m", "esm2_8m", "Hikoboshi-ESM2-8M"),
    "family": "Hikoboshi-ESM2",
    "package_kind": "registered_architecture",
    "execution_mode": "registered_architecture",
    "architecture_id": "hikoboshi_esm2_v1",
    "hidden_dimension": "320",
    "scoring.method": "raw_dot_v1",
    "gaps.family": "Hikoboshi 0.1.0 hard-SW",
    "gaps.gap_open": "-1.01982",
    "gaps.gap_extension": "0.225736",
    "soft_gaps.family": "Hikoboshi 0.1.0 soft-SW",
    "soft_gaps.gap_open": "-6.72805",
    "soft_gaps.gap_extension": "-0.0159468",
    "alignment.algorithm": "hard_local_affine_sw_v1",
    "availability.runtime_available": "yes",
}
HIKOBOSHI_010_MODELS = (HIKOBOSHI_010_MPNN64_MODEL, HIKOBOSHI_010_ESM2_8M_MODEL)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    binary = Path(os.environ["HIKOBOSHI_BUILD_ROOT"]) / "hikoboshi"
    return subprocess.run(
        [binary, *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_info(topic: str) -> subprocess.CompletedProcess[str]:
    return run_cli("info", topic)


def parse_tsv(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("\t")
        require(bool(sep), f"line is not tab-separated: {line!r}")
        parsed[key] = value
    return parsed


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise SystemExit(f"{label} failed\n{result.stdout}{result.stderr}")


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
    require(f"{prefix}.reason" in fields, f"{prefix} reason missing")
    if compiled == "yes" and runtime_available == "yes":
        require(fields[f"{prefix}.reason"] == "", f"{prefix} reason must be empty")
    else:
        require(fields[f"{prefix}.reason"], f"{prefix} reason must be non-empty")


def require_info_backends() -> None:
    result = run_info("backends")
    require_success(result, "hikoboshi info backends")
    fields = parse_tsv(result.stdout)

    require(fields["default_backend"] == "scalar", "default backend mismatch")
    require("backend_cpu_scalar_compiled" not in fields, "flat scalar key leaked")
    require("backend_default" not in fields, "flat default key leaked")

    require_availability(
        fields, "cpu.scalar", compiled="yes", runtime_available="yes"
    )
    for name in HIKOBOSHI_010_CPU_BACKENDS:
        if name != "scalar":
            require_availability(
                fields, f"cpu.{name}", compiled="no", runtime_available="no"
            )

    for name in HIKOBOSHI_010_GPU_BACKENDS:
        prefix = f"gpu.{name}"
        require_availability(fields, prefix, compiled="no", runtime_available="no")
        require(fields[f"{prefix}.devices.count"] == "0", f"{prefix} devices leaked")

    for key, expected in HIKOBOSHI_010_PIPELINE_FLAGS.items():
        require(fields[key] == expected, f"{key} mismatch")
    for key, expected in HIKOBOSHI_010_THREADING.items():
        require(fields[key] == expected, f"{key} mismatch")


def find_model_prefix(fields: dict[str, str], package_id: str) -> str:
    for key, value in fields.items():
        if key.endswith(".package_id") and value == package_id:
            return key[: -len(".package_id")]
    raise SystemExit(f"models listing missing entry for {package_id}")


def require_model_entry(fields: dict[str, str], spec: dict) -> None:
    prefix = find_model_prefix(fields, spec["package_id"])
    aliases = spec["aliases"]
    require(fields[f"{prefix}.aliases.count"] == str(len(aliases)),
            f"{spec['package_id']} alias count")
    for index, alias in enumerate(aliases):
        require(fields[f"{prefix}.aliases.{index}"] == alias,
                f"{spec['package_id']} alias {index}")
    for key, expected in spec.items():
        if key in ("aliases", "package_id", "availability.runtime_available"):
            continue
        require(fields[f"{prefix}.{key}"] == expected,
                f"{spec['package_id']} {key} mismatch")
    require(fields[f"{prefix}.version"], f"{spec['package_id']} version missing")
    require_availability(
        fields,
        f"{prefix}.availability",
        compiled="yes",
        runtime_available=spec["availability.runtime_available"],
    )


def require_info_models() -> None:
    result = run_info("models")
    require_success(result, "hikoboshi info models")
    require("package registry listing lands" not in result.stdout, "placeholder leaked")
    require("W_e.weight" not in result.stdout, "raw tensor name leaked")
    fields = parse_tsv(result.stdout)

    require(
        fields["models.count"] == str(len(HIKOBOSHI_010_MODELS)),
        "Hikoboshi 0.1.0 model count mismatch",
    )
    for spec in HIKOBOSHI_010_MODELS:
        require_model_entry(fields, spec)


def main() -> int:
    require_info_backends()
    require_info_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

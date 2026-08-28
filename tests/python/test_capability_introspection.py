from __future__ import annotations

from typing import Any

import hikoboshi as hiko

from helpers import require


HIKOBOSHI_010_VERSION = (0, 1, 0, "")
HIKOBOSHI_010_CPU_BACKENDS = ("scalar", "sse4", "avx2", "avx512", "neon", "sve")
HIKOBOSHI_010_GPU_BACKENDS = ("cuda", "hip", "metal", "vulkan", "opencl")
HIKOBOSHI_010_PIPELINE_FLAGS = {
    "pairwise_alignment": True,
    "symmetric_all_vs_all": True,
    "structure_inputs": True,
    "embedding_inputs": True,
}
HIKOBOSHI_010_THREADING = {"compiled": True, "default_mode": "auto"}
HIKOBOSHI_010_MPNN64_MODEL = {
    "package_id": "hikoboshi-mpnn-d64",
    "aliases": ["mpnn64", "mpnn-64", "Hikoboshi-MPNN-64"],
    "family": "Hikoboshi-MPNN",
    "package_kind": "registered_architecture",
    "execution_mode": "registered_architecture",
    "architecture_id": "hikoboshi_mpnn_v1",
    "hidden_dimension": 64,
    "scoring_method": "raw_dot_v1",
    "gap_family": "Hikoboshi 0.1.0 hard-SW",
    "soft_gap_family": "Hikoboshi 0.1.0 soft-SW",
    "alignment_algorithm": "hard_local_affine_sw_v1",
    "runtime_available": True,
}
HIKOBOSHI_010_ESM2_8M_MODEL = {
    "package_id": "hikoboshi-esm2-8m",
    "aliases": ["esm2-8m", "esm2_8m", "Hikoboshi-ESM2-8M"],
    "family": "Hikoboshi-ESM2",
    "package_kind": "registered_architecture",
    "execution_mode": "registered_architecture",
    "architecture_id": "hikoboshi_esm2_v1",
    "hidden_dimension": 320,
    "scoring_method": "raw_dot_v1",
    "gap_family": "Hikoboshi 0.1.0 hard-SW",
    "soft_gap_family": "Hikoboshi 0.1.0 soft-SW",
    "alignment_algorithm": "hard_local_affine_sw_v1",
    "runtime_available": True,
}
HIKOBOSHI_010_MODELS = (HIKOBOSHI_010_MPNN64_MODEL, HIKOBOSHI_010_ESM2_8M_MODEL)


def require_010_version() -> None:
    version = hiko.version_info()
    observed = (
        version["major"],
        version["minor"],
        version["patch"],
        version["label"],
    )
    require(
        observed == HIKOBOSHI_010_VERSION,
        "capability introspection expectations must be updated for this release",
    )


def require_availability(
    payload: object, *, compiled: bool, runtime_available: bool, label: str
) -> None:
    require(isinstance(payload, dict), f"{label} availability must be a dict")
    require(payload.get("compiled") is compiled, f"{label} compiled flag mismatch")
    require(
        payload.get("runtime_available") is runtime_available,
        f"{label} runtime availability mismatch",
    )
    require(isinstance(payload.get("reason"), str), f"{label} reason missing")
    if compiled and runtime_available:
        require(payload["reason"] == "", f"{label} available reason must be empty")
    else:
        require(payload["reason"], f"{label} reserved reason must be non-empty")


def require_backends(info: dict[str, Any]) -> None:
    backends = info["backends"]
    require(backends == info["backend_capabilities"], "backend aliases diverged")
    require(backends["default_backend"] == "scalar", "default backend mismatch")

    cpu = backends["cpu"]
    require(set(cpu) == set(HIKOBOSHI_010_CPU_BACKENDS), "CPU backend keys mismatch")
    require_availability(
        cpu["scalar"], compiled=True, runtime_available=True, label="cpu.scalar"
    )
    for name in HIKOBOSHI_010_CPU_BACKENDS:
        if name != "scalar":
            require_availability(
                cpu[name],
                compiled=False,
                runtime_available=False,
                label=f"cpu.{name}",
            )

    gpu = backends["gpu"]
    require(set(gpu) == set(HIKOBOSHI_010_GPU_BACKENDS), "GPU backend keys mismatch")
    for name in HIKOBOSHI_010_GPU_BACKENDS:
        payload = gpu[name]
        require_availability(
            payload, compiled=False, runtime_available=False, label=f"gpu.{name}"
        )
        require(payload["devices"] == [], f"gpu.{name} device list must be empty")

    require(
        backends["pipeline"] == HIKOBOSHI_010_PIPELINE_FLAGS,
        "pipeline capabilities mismatch",
    )
    require(info["threading"] == HIKOBOSHI_010_THREADING, "threading info mismatch")


def require_package_entry(package: dict[str, Any], spec: dict[str, Any], label: str) -> None:
    require(isinstance(package, dict), f"{label} entry must be a dict")
    require("tensors" not in package, f"{label} must not expose raw tensors")
    require(package["package_id"] == spec["package_id"], f"{label} package id")
    require(package["aliases"] == spec["aliases"], f"{label} aliases mismatch")
    require(package["family"] == spec["family"], f"{label} family mismatch")
    require(isinstance(package["version"], str) and package["version"], f"{label} version")
    require(package["package_kind"] == spec["package_kind"], f"{label} package kind")
    require(package["execution_mode"] == spec["execution_mode"], f"{label} execution mode")
    require(package["architecture_id"] == spec["architecture_id"], f"{label} architecture id")
    require(
        package["hidden_dimension"] == spec["hidden_dimension"],
        f"{label} hidden dimension",
    )

    scoring = package["scoring"]
    require(scoring["method"] == spec["scoring_method"], f"{label} scoring method")
    gaps = package["gaps"]
    require(gaps["family"] == spec["gap_family"], f"{label} gap family")
    soft_gaps = package["soft_gaps"]
    require(
        soft_gaps["family"] == spec["soft_gap_family"],
        f"{label} soft gap family",
    )
    require(
        package["alignment_algorithm"] == spec["alignment_algorithm"],
        f"{label} alignment algorithm",
    )
    availability = package["availability"]
    require_availability(
        availability,
        compiled=True,
        runtime_available=spec["runtime_available"],
        label=f"{label} availability",
    )


def require_models(info: dict[str, Any]) -> None:
    models = info["models"]
    require(isinstance(models, dict), "models must be a dict")
    require(
        models["count"] == len(HIKOBOSHI_010_MODELS),
        "Hikoboshi 0.1.0 must list compiled packages for every registered architecture",
    )
    packages = models["packages"]
    require(isinstance(packages, list), "models packages must be a list")
    require(len(packages) == models["count"], "models count/list mismatch")

    for spec in HIKOBOSHI_010_MODELS:
        match = next(
            (entry for entry in packages if entry.get("package_id") == spec["package_id"]),
            None,
        )
        require(match is not None, f"{spec['package_id']} package missing from models")
        require_package_entry(match, spec, spec["package_id"])

    # Hard and T=1 soft defaults are independently calibrated for each model.
    mpnn_match = next(
        entry for entry in packages
        if entry.get("package_id") == HIKOBOSHI_010_MPNN64_MODEL["package_id"]
    )
    mpnn_gaps = mpnn_match["gaps"]
    mpnn_soft_gaps = mpnn_match["soft_gaps"]
    require(abs(mpnn_gaps["gap_open"] - (-1.4)) < 1e-6, "MPNN-64 gap_open mismatch")
    require(
        abs(mpnn_gaps["gap_extension"] - (-0.15)) < 1e-6,
        "MPNN-64 gap extension mismatch",
    )
    require(
        abs(mpnn_soft_gaps["gap_open"] - (-3.21337)) < 1e-6,
        "MPNN-64 soft gap_open mismatch",
    )
    require(
        abs(mpnn_soft_gaps["gap_extension"] - (-0.111704)) < 1e-6,
        "MPNN-64 soft gap extension mismatch",
    )

    esm2_match = next(
        entry for entry in packages
        if entry.get("package_id") == HIKOBOSHI_010_ESM2_8M_MODEL["package_id"]
    )
    esm2_gaps = esm2_match["gaps"]
    esm2_soft_gaps = esm2_match["soft_gaps"]
    require(
        abs(esm2_gaps["gap_open"] - (-1.01982)) < 1e-6,
        "ESM2-8M gap_open mismatch",
    )
    require(
        abs(esm2_gaps["gap_extension"] - 0.225736) < 1e-6,
        "ESM2-8M gap extension mismatch",
    )
    require(
        abs(esm2_soft_gaps["gap_open"] - (-6.72805)) < 1e-6,
        "ESM2-8M soft gap_open mismatch",
    )
    require(
        abs(esm2_soft_gaps["gap_extension"] - (-0.0159468)) < 1e-8,
        "ESM2-8M soft gap extension mismatch",
    )
    require(
        len(
            {
                (mpnn_gaps["gap_open"], mpnn_gaps["gap_extension"]),
                (mpnn_soft_gaps["gap_open"], mpnn_soft_gaps["gap_extension"]),
                (esm2_gaps["gap_open"], esm2_gaps["gap_extension"]),
                (esm2_soft_gaps["gap_open"], esm2_soft_gaps["gap_extension"]),
            }
        )
        == 4,
        "the MPNN/ESM2 hard/T=1 gap pairs must remain distinct",
    )


def main() -> int:
    require_010_version()
    info = hiko.info()
    require_backends(info)
    require_models(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hikoboshi as hiko

from helpers import require, require_backend_availability


info = hiko.info()
capabilities = info["backend_capabilities"]
threading = info["threading"]

require(
    set(capabilities) == {"cpu", "gpu", "pipeline", "default_backend"},
    "backend_capabilities must use the nested public shape",
)
for old_key in (
    "auto_backend",
    "scalar_backend",
    "simd_backend",
    "gpu_backend",
    "pairwise_alignment",
    "symmetric_all_vs_all",
    "structure_inputs",
    "embedding_inputs",
):
    require(old_key not in capabilities, f"old flat capability key leaked: {old_key}")

require(capabilities["default_backend"] == "scalar", "default backend mismatch")
require(threading == {"compiled": True, "default_mode": "auto"}, "threading info mismatch")

cpu = capabilities["cpu"]
require(
    set(cpu) == {"scalar", "sse4", "avx2", "avx512", "neon", "sve"},
    "cpu backend keys mismatch",
)
require_backend_availability(cpu["scalar"], compiled=True, available=True)
for name in ("sse4", "avx2", "avx512", "neon", "sve"):
    require_backend_availability(cpu[name], compiled=False, available=False)

gpu = capabilities["gpu"]
require(
    set(gpu) == {"cuda", "hip", "metal", "vulkan", "opencl"},
    "gpu backend keys mismatch",
)
for name in ("cuda", "hip", "metal", "vulkan", "opencl"):
    require_backend_availability(gpu[name], compiled=False, available=False)
    require(gpu[name]["devices"] == [], "0.1.0 GPU device list must be empty")

pipeline = capabilities["pipeline"]
require(
    pipeline
    == {
        "pairwise_alignment": True,
        "symmetric_all_vs_all": True,
        "structure_inputs": True,
        "embedding_inputs": True,
    },
    "pipeline capability flags mismatch",
)

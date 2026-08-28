from __future__ import annotations

import hikoboshi as hiko

from helpers import require, require_backend_availability


info = hiko.info()

require("backends" in info, "info() must expose backends")
require("backend_capabilities" in info, "compatibility capability key missing")
require(info["backends"] == info["backend_capabilities"], "backends must mirror capabilities")
require(info["backends"] is not info["backend_capabilities"], "backends must be copy-safe")
require(info["version"]["product_name"] == "Hikoboshi", "version information missing")
require(info["weights"]["model_name"] == "hikoboshi-mpnn-d64", "weights information missing")
require(info["models"]["count"] == 2, "MPNN-64 + ESM2-8M models listing missing")
require(info["threading"] == {"compiled": True, "default_mode": "auto"}, "threading info mismatch")

backends = info["backends"]
require(
    set(backends) == {"cpu", "gpu", "pipeline", "default_backend"},
    "backends must use the nested public shape",
)
require(backends["default_backend"] == "scalar", "default backend mismatch")

cpu = backends["cpu"]
require(
    set(cpu) == {"scalar", "sse4", "avx2", "avx512", "neon", "sve"},
    "cpu backend keys mismatch",
)
require_backend_availability(cpu["scalar"], compiled=True, available=True)
for name in ("sse4", "avx2", "avx512", "neon", "sve"):
    require_backend_availability(cpu[name], compiled=False, available=False)

gpu = backends["gpu"]
require(
    set(gpu) == {"cuda", "hip", "metal", "vulkan", "opencl"},
    "gpu backend keys mismatch",
)
for name in ("cuda", "hip", "metal", "vulkan", "opencl"):
    require_backend_availability(gpu[name], compiled=False, available=False)
    require(gpu[name]["devices"] == [], "0.1.0 GPU device list must be empty")

require(
    backends["pipeline"] == {
        "pairwise_alignment": True,
        "symmetric_all_vs_all": True,
        "structure_inputs": True,
        "embedding_inputs": True,
    },
    "pipeline capability flags mismatch",
)

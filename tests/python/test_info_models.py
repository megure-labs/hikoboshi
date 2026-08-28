from __future__ import annotations

import hikoboshi as hiko

from helpers import require, require_sha256


PENDING_SENTINEL = "pending"


def require_string(value: object, label: str) -> None:
    require(isinstance(value, str) and value, f"{label} missing")


def require_pending_or_sha256(value: object, label: str) -> None:
    if value == PENDING_SENTINEL:
        return
    require_sha256(value, label)


info = hiko.info()
models = info["models"]

require(isinstance(models, dict), "models must be a dict")
require("status" not in models, "models placeholder status leaked")
require("message" not in models, "models placeholder message leaked")
require(models["count"] == 2, "model count mismatch")

packages = models["packages"]
require(isinstance(packages, list), "models packages must be a list")
require(len(packages) == 2, "MPNN-64 and ESM2-8M compiled packages must be listed")

mpnn_package = next(
    (entry for entry in packages if entry.get("package_id") == "hikoboshi-mpnn-d64"),
    None,
)
esm2_package = next(
    (entry for entry in packages if entry.get("package_id") == "hikoboshi-esm2-8m"),
    None,
)
require(mpnn_package is not None, "MPNN-64 package entry must appear")
require(esm2_package is not None, "ESM2-8M package entry must appear")

for entry in packages:
    require(isinstance(entry, dict), "model entry must be a dict")
    require("tensors" not in entry, "raw tensor internals must not be exposed")

require(
    mpnn_package["aliases"] == ["mpnn64", "mpnn-64", "Hikoboshi-MPNN-64"],
    "MPNN-64 aliases mismatch",
)
require(mpnn_package["family"] == "Hikoboshi-MPNN", "MPNN-64 family mismatch")
require_string(mpnn_package["version"], "MPNN-64 version")
require(mpnn_package["package_kind"] == "registered_architecture", "MPNN-64 package kind mismatch")
require(mpnn_package["execution_mode"] == "registered_architecture", "MPNN-64 execution mode mismatch")
require(mpnn_package["architecture_id"] == "hikoboshi_mpnn_v1", "MPNN-64 architecture id mismatch")
require(mpnn_package["hidden_dimension"] == 64, "MPNN-64 hidden dimension mismatch")

scoring = mpnn_package["scoring"]
require(scoring["method"] == "raw_dot_v1", "MPNN-64 scoring method mismatch")
require(scoring["similarity"] == "raw_dot_product", "MPNN-64 scoring similarity mismatch")
semantics = scoring["semantics"]
require(semantics["normalization"] == "none", "MPNN-64 normalization mismatch")
require(semantics["scale_family"] == "raw_dot", "MPNN-64 scale family mismatch")
require(semantics["higher_is_better"] is True, "MPNN-64 score ordering mismatch")
require(semantics["local_affine_additive"] is True, "MPNN-64 score additivity mismatch")

gaps = mpnn_package["gaps"]
require(gaps["family"] == "Hikoboshi 0.1.0 hard-SW", "MPNN-64 gap family mismatch")
require(gaps["model"] == "affine", "MPNN-64 gap model mismatch")
require(abs(gaps["gap_open"] - (-1.4)) < 1e-6, "MPNN-64 gap_open mismatch")
require(abs(gaps["gap_extension"] - (-0.15)) < 1e-6, "MPNN-64 gap extension mismatch")
require(
    gaps["convention"] == "gap_open_plus_k_minus_1_gap_extension",
    "MPNN-64 gap convention mismatch",
)
require(gaps["calibrated_for_score_method"] == "raw_dot_v1", "MPNN-64 gap scoring mismatch")
soft_gaps = mpnn_package["soft_gaps"]
require(soft_gaps["family"] == "Hikoboshi 0.1.0 soft-SW", "MPNN-64 soft gap family mismatch")
require(soft_gaps["model"] == "affine", "MPNN-64 soft gap model mismatch")
require(abs(soft_gaps["gap_open"] - (-3.21337)) < 1e-6, "MPNN-64 soft gap_open mismatch")
require(
    abs(soft_gaps["gap_extension"] - (-0.111704)) < 1e-6,
    "MPNN-64 soft gap extension mismatch",
)
require(
    soft_gaps["convention"] == "gap_open_plus_k_minus_1_gap_extension",
    "MPNN-64 soft gap convention mismatch",
)
require(
    soft_gaps["calibrated_for_score_method"] == "raw_dot_v1",
    "MPNN-64 soft gap scoring mismatch",
)
require(
    mpnn_package["alignment_algorithm"] == "hard_local_affine_sw_v1",
    "MPNN-64 alignment algorithm mismatch",
)

mpnn_checksum = mpnn_package["checksum"]
require(mpnn_checksum["algorithm"] == "sha256", "MPNN-64 checksum algorithm mismatch")
require_sha256(mpnn_checksum["package"], "MPNN-64 package checksum")
require_sha256(mpnn_checksum["source_checkpoint"], "MPNN-64 source checkpoint checksum")

mpnn_provenance = mpnn_package["provenance"]
for key in (
    "source_checkpoint",
    "generation_tool",
    "generation_tool_version",
    "generation_date",
    "validation_status",
    "status",
):
    require_string(mpnn_provenance.get(key), f"MPNN-64 provenance.{key}")

mpnn_availability = mpnn_package["availability"]
require(mpnn_availability["compiled"] is True, "MPNN-64 compiled flag mismatch")
require(mpnn_availability["runtime_available"] is True, "MPNN-64 availability flag mismatch")
require(isinstance(mpnn_availability["reason"], str), "MPNN-64 availability reason missing")

# ESM2-8M fully runtime-available now that the embedded weights blob
# from `esm2-8m-weights-package` ships real per-tensor SHA-256 records,
# calibrated gap defaults, and a non-empty source-checkpoint identity.
require(
    esm2_package["aliases"] == ["esm2-8m", "esm2_8m", "Hikoboshi-ESM2-8M"],
    "ESM2-8M aliases mismatch",
)
require(esm2_package["family"] == "Hikoboshi-ESM2", "ESM2-8M family mismatch")
require_string(esm2_package["version"], "ESM2-8M version")
require(esm2_package["package_kind"] == "registered_architecture", "ESM2-8M package kind mismatch")
require(esm2_package["execution_mode"] == "registered_architecture", "ESM2-8M execution mode mismatch")
require(esm2_package["architecture_id"] == "hikoboshi_esm2_v1", "ESM2-8M architecture id mismatch")
require(esm2_package["hidden_dimension"] == 320, "ESM2-8M hidden dimension mismatch")

esm2_scoring = esm2_package["scoring"]
require(esm2_scoring["method"] == "raw_dot_v1", "ESM2-8M scoring method mismatch")
require(esm2_scoring["similarity"] == "raw_dot_product", "ESM2-8M scoring similarity mismatch")
esm2_semantics = esm2_scoring["semantics"]
require(esm2_semantics["normalization"] == "none", "ESM2-8M normalization mismatch")
require(esm2_semantics["scale_family"] == "raw_dot", "ESM2-8M scale family mismatch")
esm2_gaps = esm2_package["gaps"]
require(esm2_gaps["family"] == "Hikoboshi 0.1.0 hard-SW", "ESM2-8M gap family mismatch")
require(abs(esm2_gaps["gap_open"] - (-1.01982)) < 1e-6, "ESM2-8M hard gap_open mismatch")
require(abs(esm2_gaps["gap_extension"] - 0.225736) < 1e-6, "ESM2-8M hard gap extension mismatch")
esm2_soft_gaps = esm2_package["soft_gaps"]
require(
    esm2_soft_gaps["family"] == "Hikoboshi 0.1.0 soft-SW",
    "ESM2-8M soft gap family mismatch",
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
    esm2_package["alignment_algorithm"] == "hard_local_affine_sw_v1",
    "ESM2-8M alignment algorithm mismatch",
)

esm2_checksum = esm2_package["checksum"]
require(esm2_checksum["algorithm"] == "sha256", "ESM2-8M checksum algorithm mismatch")
require_pending_or_sha256(esm2_checksum["package"], "ESM2-8M package checksum")
require_pending_or_sha256(esm2_checksum["source_checkpoint"], "ESM2-8M source checkpoint checksum")

esm2_availability = esm2_package["availability"]
require(esm2_availability["compiled"] is True, "ESM2-8M compiled flag mismatch")
require(
    esm2_availability["runtime_available"] is True,
    "ESM2-8M must report runtime_available=true once the embedded "
    "weights blob ships",
)
require(
    esm2_availability["reason"] == "",
    "ESM2-8M runtime-available record must not carry an unavailability "
    "reason",
)

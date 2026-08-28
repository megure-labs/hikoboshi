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


def require_sha256(value: str, label: str) -> None:
    require(len(value) == 64, f"{label} must be a SHA-256 hex digest")
    require(all(char in "0123456789abcdef" for char in value), f"{label} must be hex")


def require_backend_listing() -> None:
    result = run_info("backends")
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)
    fields = parse_tsv(result.stdout)

    require(fields["default_backend"] == "scalar", "default backend mismatch")
    require("backend_cpu_scalar_compiled" not in fields, "flat CPU key leaked")
    require("backend_default" not in fields, "flat default backend key leaked")

    require_availability(fields, "cpu.scalar", compiled="yes", runtime_available="yes")
    for name in ("sse4", "avx2", "avx512", "neon", "sve"):
        require_availability(fields, f"cpu.{name}", compiled="no", runtime_available="no")

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


def require_pending_or_sha256(value: str, label: str) -> None:
    if value == "pending":
        return
    require_sha256(value, label)


def find_model_prefix(fields: dict[str, str], package_id: str) -> str:
    for key, value in fields.items():
        if key.endswith(".package_id") and value == package_id:
            return key[: -len(".package_id")]
    raise SystemExit(f"models listing missing entry for {package_id}")


def require_mpnn64_listing(fields: dict[str, str]) -> None:
    prefix = find_model_prefix(fields, "hikoboshi-mpnn-d64")
    require(fields[f"{prefix}.aliases.count"] == "3", "MPNN-64 alias count mismatch")
    require(fields[f"{prefix}.aliases.0"] == "mpnn64", "MPNN-64 first alias mismatch")
    require(fields[f"{prefix}.aliases.1"] == "mpnn-64", "MPNN-64 second alias mismatch")
    require(
        fields[f"{prefix}.aliases.2"] == "Hikoboshi-MPNN-64",
        "MPNN-64 old canonical alias mismatch",
    )
    require(fields[f"{prefix}.family"] == "Hikoboshi-MPNN", "MPNN-64 family mismatch")
    require(fields[f"{prefix}.version"], "MPNN-64 version must be populated")
    require(
        fields[f"{prefix}.package_kind"] == "registered_architecture",
        "MPNN-64 package kind mismatch",
    )
    require(
        fields[f"{prefix}.execution_mode"] == "registered_architecture",
        "MPNN-64 execution mode mismatch",
    )
    require(
        fields[f"{prefix}.architecture_id"] == "hikoboshi_mpnn_v1",
        "MPNN-64 architecture id mismatch",
    )
    require(fields[f"{prefix}.hidden_dimension"] == "64", "MPNN-64 hidden dimension mismatch")
    require(fields[f"{prefix}.scoring.method"] == "raw_dot_v1", "MPNN-64 scoring mismatch")
    require(
        fields[f"{prefix}.scoring.similarity"] == "raw_dot_product",
        "MPNN-64 similarity mismatch",
    )
    require(fields[f"{prefix}.scoring.normalization"] == "none", "MPNN-64 normalization mismatch")
    require(fields[f"{prefix}.scoring.scale_family"] == "raw_dot", "MPNN-64 scale mismatch")
    require(fields[f"{prefix}.gaps.family"] == "Hikoboshi 0.1.0 hard-SW", "MPNN-64 gap family mismatch")
    require(fields[f"{prefix}.gaps.model"] == "affine", "MPNN-64 gap model mismatch")
    require(fields[f"{prefix}.gaps.gap_open"] == "-1.4", "MPNN-64 gap_open mismatch")
    require(fields[f"{prefix}.gaps.gap_extension"] == "-0.15", "MPNN-64 gap extension mismatch")
    require(
        fields[f"{prefix}.gaps.convention"] == "gap_open_plus_k_minus_1_gap_extension",
        "MPNN-64 gap convention mismatch",
    )
    require(
        fields[f"{prefix}.gaps.calibrated_for_score_method"] == "raw_dot_v1",
        "MPNN-64 gap scoring calibration mismatch",
    )
    require(fields[f"{prefix}.soft_gaps.family"] == "Hikoboshi 0.1.0 soft-SW", "MPNN-64 soft gap family mismatch")
    require(fields[f"{prefix}.soft_gaps.model"] == "affine", "MPNN-64 soft gap model mismatch")
    require(fields[f"{prefix}.soft_gaps.gap_open"] == "-3.21337", "MPNN-64 soft gap_open mismatch")
    require(fields[f"{prefix}.soft_gaps.gap_extension"] == "-0.111704", "MPNN-64 soft gap extension mismatch")
    require(
        fields[f"{prefix}.soft_gaps.convention"] == "gap_open_plus_k_minus_1_gap_extension",
        "MPNN-64 soft gap convention mismatch",
    )
    require(
        fields[f"{prefix}.soft_gaps.calibrated_for_score_method"] == "raw_dot_v1",
        "MPNN-64 soft gap scoring calibration mismatch",
    )
    require(
        fields[f"{prefix}.alignment.algorithm"] == "hard_local_affine_sw_v1",
        "MPNN-64 alignment algorithm mismatch",
    )
    require(fields[f"{prefix}.checksum.algorithm"] == "sha256", "MPNN-64 checksum algorithm mismatch")
    require_sha256(fields[f"{prefix}.checksum.package"], "MPNN-64 package checksum")
    require_sha256(
        fields[f"{prefix}.checksum.source_checkpoint"],
        "MPNN-64 source checkpoint checksum",
    )
    require(fields[f"{prefix}.provenance.source_checkpoint"], "MPNN-64 source provenance missing")
    require(fields[f"{prefix}.provenance.generation_tool"], "MPNN-64 generation tool missing")
    require(fields[f"{prefix}.provenance.generation_tool_version"], "MPNN-64 generation version missing")
    require(fields[f"{prefix}.provenance.generation_date"], "MPNN-64 generation date missing")
    require(fields[f"{prefix}.provenance.validation_status"], "MPNN-64 validation status missing")
    require(fields[f"{prefix}.provenance.status"], "MPNN-64 provenance status missing")
    require(fields[f"{prefix}.availability.compiled"] == "yes", "MPNN-64 compiled flag mismatch")
    require(
        fields[f"{prefix}.availability.runtime_available"] == "yes",
        "MPNN-64 runtime availability mismatch",
    )
    require(f"{prefix}.availability.reason" in fields, "MPNN-64 availability reason missing")


def require_esm2_8m_listing(fields: dict[str, str]) -> None:
    prefix = find_model_prefix(fields, "hikoboshi-esm2-8m")
    require(fields[f"{prefix}.aliases.count"] == "3", "ESM2-8M alias count mismatch")
    require(fields[f"{prefix}.aliases.0"] == "esm2-8m", "ESM2-8M first alias mismatch")
    require(fields[f"{prefix}.aliases.1"] == "esm2_8m", "ESM2-8M second alias mismatch")
    require(
        fields[f"{prefix}.aliases.2"] == "Hikoboshi-ESM2-8M",
        "ESM2-8M old canonical alias mismatch",
    )
    require(fields[f"{prefix}.family"] == "Hikoboshi-ESM2", "ESM2-8M family mismatch")
    require(fields[f"{prefix}.version"], "ESM2-8M version must be populated")
    require(
        fields[f"{prefix}.package_kind"] == "registered_architecture",
        "ESM2-8M package kind mismatch",
    )
    require(
        fields[f"{prefix}.execution_mode"] == "registered_architecture",
        "ESM2-8M execution mode mismatch",
    )
    require(
        fields[f"{prefix}.architecture_id"] == "hikoboshi_esm2_v1",
        "ESM2-8M architecture id mismatch",
    )
    require(fields[f"{prefix}.hidden_dimension"] == "320", "ESM2-8M hidden dimension mismatch")
    require(fields[f"{prefix}.scoring.method"] == "raw_dot_v1", "ESM2-8M scoring mismatch")
    require(
        fields[f"{prefix}.scoring.similarity"] == "raw_dot_product",
        "ESM2-8M similarity mismatch",
    )
    require(fields[f"{prefix}.scoring.normalization"] == "none", "ESM2-8M normalization mismatch")
    require(fields[f"{prefix}.scoring.scale_family"] == "raw_dot", "ESM2-8M scale mismatch")
    require(fields[f"{prefix}.gaps.family"] == "Hikoboshi 0.1.0 hard-SW", "ESM2-8M gap family mismatch")
    require(fields[f"{prefix}.gaps.gap_open"] == "-1.01982", "ESM2-8M hard gap_open mismatch")
    require(fields[f"{prefix}.gaps.gap_extension"] == "0.225736", "ESM2-8M hard gap extension mismatch")
    require(fields[f"{prefix}.soft_gaps.family"] == "Hikoboshi 0.1.0 soft-SW", "ESM2-8M soft gap family mismatch")
    require(fields[f"{prefix}.soft_gaps.gap_open"] == "-6.72805", "ESM2-8M soft gap_open mismatch")
    require(
        fields[f"{prefix}.soft_gaps.gap_extension"] == "-0.0159468",
        "ESM2-8M soft gap extension mismatch",
    )
    require(
        fields[f"{prefix}.alignment.algorithm"] == "hard_local_affine_sw_v1",
        "ESM2-8M alignment algorithm mismatch",
    )
    require(fields[f"{prefix}.checksum.algorithm"] == "sha256", "ESM2-8M checksum algorithm mismatch")
    require_pending_or_sha256(
        fields[f"{prefix}.checksum.package"], "ESM2-8M package checksum"
    )
    require_pending_or_sha256(
        fields[f"{prefix}.checksum.source_checkpoint"],
        "ESM2-8M source checkpoint checksum",
    )
    require(fields[f"{prefix}.availability.compiled"] == "yes", "ESM2-8M compiled flag mismatch")
    require(
        fields[f"{prefix}.availability.runtime_available"] == "yes",
        "ESM2-8M must report runtime_available=yes once the embedded "
        "weights blob ships",
    )
    require(
        not fields.get(f"{prefix}.availability.reason"),
        "ESM2-8M runtime-available listing must not carry an "
        "unavailability reason",
    )


def require_model_listing() -> None:
    result = run_info("models")
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)
    require("package registry listing lands in m15" not in result.stdout, "placeholder leaked")
    require("W_e.weight" not in result.stdout, "raw tensor name leaked")
    require("layers.0" not in result.stdout, "raw layer tensor name leaked")
    fields = parse_tsv(result.stdout)

    require(fields["models.count"] == "2", "model count must include MPNN-64 + ESM2-8M")
    require_mpnn64_listing(fields)
    require_esm2_8m_listing(fields)


def require_bare_info_summary() -> None:
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


def main() -> int:
    require_backend_listing()
    require_bare_info_summary()
    require_model_listing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

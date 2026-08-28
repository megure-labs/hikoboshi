from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _metric(payload: dict[str, Any] | None) -> float | None:
    if not payload or not payload.get("valid"):
        return None
    return float(payload["value"])


def _reason(payload: dict[str, Any] | None) -> str | None:
    if not payload or payload.get("valid"):
        return None
    return str(payload.get("reason", "unavailable"))


@dataclass(frozen=True)
class AlignmentStep:
    query_index: int
    target_index: int
    residue_score: float

    @classmethod
    def from_core(cls, payload: dict[str, Any]) -> "AlignmentStep":
        return cls(
            query_index=int(payload["query_index"]),
            target_index=int(payload["target_index"]),
            residue_score=float(payload["residue_score"]),
        )


@dataclass(frozen=True)
class AlignmentPath:
    steps: tuple[AlignmentStep, ...]
    aligned_pairs: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int

    @classmethod
    def from_core(cls, payload: dict[str, Any]) -> "AlignmentPath":
        return cls(
            steps=tuple(AlignmentStep.from_core(step) for step in payload["steps"]),
            aligned_pairs=int(payload["aligned_pairs"]),
            query_start=int(payload["query_start"]),
            query_end=int(payload["query_end"]),
            target_start=int(payload["target_start"]),
            target_end=int(payload["target_end"]),
        )


@dataclass(frozen=True)
class PairwiseMetrics:
    raw_sw_score: float
    soft_sw_score: float | None
    sw_per_query_len: float | None
    sw_per_target_len: float | None
    sw_per_aligned: float | None
    coverage_query: float | None
    coverage_target: float | None
    coverage_mean: float | None
    identity: float | None
    rmsd: float | None
    tm_score_query: float | None
    tm_score_target: float | None
    lddt: float | None
    lddt_byA: float | None
    lddt_byB: float | None
    lddt_aln: float | None
    coverage_byA: float | None
    coverage_byB: float | None
    ecs: float | None
    invalid_reasons: dict[str, str]

    @classmethod
    def from_core(cls, payload: dict[str, Any]) -> "PairwiseMetrics":
        names = (
            "soft_sw_score",
            "sw_per_query_len",
            "sw_per_target_len",
            "sw_per_aligned",
            "coverage_query",
            "coverage_target",
            "coverage_mean",
            "identity",
            "rmsd",
            "tm_score_query",
            "tm_score_target",
            "lddt",
            "lddt_byA",
            "lddt_byB",
            "lddt_aln",
            "coverage_byA",
            "coverage_byB",
            "ecs",
        )
        reasons = {
            name: reason
            for name in names
            if (reason := _reason(payload.get(name))) is not None
        }
        return cls(
            raw_sw_score=float(payload["raw_sw_score"]),
            soft_sw_score=_metric(payload.get("soft_sw_score")),
            sw_per_query_len=_metric(payload.get("sw_per_query_len")),
            sw_per_target_len=_metric(payload.get("sw_per_target_len")),
            sw_per_aligned=_metric(payload.get("sw_per_aligned")),
            coverage_query=_metric(payload.get("coverage_query")),
            coverage_target=_metric(payload.get("coverage_target")),
            coverage_mean=_metric(payload.get("coverage_mean")),
            identity=_metric(payload.get("identity")),
            rmsd=_metric(payload.get("rmsd")),
            tm_score_query=_metric(payload.get("tm_score_query")),
            tm_score_target=_metric(payload.get("tm_score_target")),
            lddt=_metric(payload.get("lddt")),
            lddt_byA=_metric(payload.get("lddt_byA")),
            lddt_byB=_metric(payload.get("lddt_byB")),
            lddt_aln=_metric(payload.get("lddt_aln")),
            coverage_byA=_metric(payload.get("coverage_byA")),
            coverage_byB=_metric(payload.get("coverage_byB")),
            ecs=_metric(payload.get("ecs")),
            invalid_reasons=reasons,
        )


@dataclass(frozen=True)
class PairwiseResult:
    path: AlignmentPath
    metrics: PairwiseMetrics
    query_metadata: dict[str, Any] | None = None
    target_metadata: dict[str, Any] | None = None

    @classmethod
    def from_core(
        cls,
        payload: dict[str, Any],
        *,
        query_metadata: dict[str, Any] | None = None,
        target_metadata: dict[str, Any] | None = None,
    ) -> "PairwiseResult":
        return cls(
            path=AlignmentPath.from_core(payload["path"]),
            metrics=PairwiseMetrics.from_core(payload["metrics"]),
            query_metadata=query_metadata,
            target_metadata=target_metadata,
        )


@dataclass(frozen=True)
class AllVsAllRecord:
    query_index: int
    target_index: int
    result: PairwiseResult

    @classmethod
    def from_core(
        cls,
        payload: dict[str, Any],
        *,
        metadata: tuple[dict[str, Any] | None, ...],
    ) -> "AllVsAllRecord":
        query_index = int(payload["query_index"])
        target_index = int(payload["target_index"])
        return cls(
            query_index=query_index,
            target_index=target_index,
            result=PairwiseResult.from_core(
                payload["result"],
                query_metadata=metadata[query_index] if query_index < len(metadata) else None,
                target_metadata=metadata[target_index] if target_index < len(metadata) else None,
            ),
        )


@dataclass(frozen=True)
class AllVsAllResult:
    records: tuple[AllVsAllRecord, ...]

    @classmethod
    def from_core(
        cls,
        payload: dict[str, Any],
        *,
        metadata: tuple[dict[str, Any] | None, ...] = (),
    ) -> "AllVsAllResult":
        return cls(
            records=tuple(
                AllVsAllRecord.from_core(record, metadata=metadata)
                for record in payload["records"]
            )
        )


@dataclass(frozen=True)
class EncodeResult:
    embeddings: Any
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class InverseFoldSequenceResult:
    sequence: str
    score: float
    recovery_vs_native: float | None
    seed: int
    decode_order: str

    @classmethod
    def from_core(cls, payload: dict[str, Any]) -> "InverseFoldSequenceResult":
        return cls(
            sequence=str(payload["sequence"]),
            score=float(payload["score"]),
            recovery_vs_native=_metric(payload.get("recovery_vs_native")),
            seed=int(payload["seed"]),
            decode_order=str(payload["decode_order"]),
        )


@dataclass(frozen=True)
class InverseFoldLogProbsArtifact:
    path: str
    shape: tuple[int, int, int]
    written: bool

    @classmethod
    def from_core(cls, payload: dict[str, Any] | None) -> "InverseFoldLogProbsArtifact":
        if not payload:
            return cls(path="", shape=(0, 0, 0), written=False)
        shape_payload = tuple(int(value) for value in payload.get("shape", (0, 0, 0)))
        if len(shape_payload) != 3:
            shape_payload = (0, 0, 0)
        return cls(
            path=str(payload.get("path", "")),
            shape=shape_payload,  # type: ignore[arg-type]
            written=bool(payload.get("written", False)),
        )


@dataclass(frozen=True)
class InverseFoldResult:
    sequences: tuple[InverseFoldSequenceResult, ...]
    logprobs: InverseFoldLogProbsArtifact

    @classmethod
    def from_core(cls, payload: dict[str, Any]) -> "InverseFoldResult":
        return cls(
            sequences=tuple(
                InverseFoldSequenceResult.from_core(item)
                for item in payload["sequences"]
            ),
            logprobs=InverseFoldLogProbsArtifact.from_core(payload.get("logprobs")),
        )


@dataclass(frozen=True)
class AllVsAllStreamingSummary:
    """Summary metadata returned by the streaming TSV all-vs-all entry point.

    Records are written to ``output_path`` as the C++ engine produces them, so
    the Python caller never holds an in-memory ``AllVsAllResult``. ``pair_count``
    is the total number of pair rows the engine attempted to write (the same
    formula Hikoboshi uses for symmetric all-vs-all enumeration), and
    ``wall_time_seconds`` is the time spent inside the streaming engine call,
    measured around the GIL-released native section.
    """

    pair_count: int
    wall_time_seconds: float
    output_path: str

    @classmethod
    def from_core(cls, payload: dict[str, Any]) -> "AllVsAllStreamingSummary":
        return cls(
            pair_count=int(payload["pair_count"]),
            wall_time_seconds=float(payload["wall_time_seconds"]),
            output_path=str(payload["output_path"]),
        )


__all__ = [
    "AlignmentStep",
    "AlignmentPath",
    "PairwiseMetrics",
    "PairwiseResult",
    "AllVsAllRecord",
    "AllVsAllResult",
    "AllVsAllStreamingSummary",
    "EncodeResult",
    "InverseFoldSequenceResult",
    "InverseFoldLogProbsArtifact",
    "InverseFoldResult",
]

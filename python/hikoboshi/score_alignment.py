from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ._arrays import load_core
from ._chain_resolve import (
    chain_selection,
    reject_unknown_kwargs,
    resolve_deprecated_model_kwargs,
)
from .exceptions import raise_from_core


def _metric(payload: dict[str, Any] | None) -> float | None:
    if not payload or not payload.get("valid"):
        return None
    return float(payload["value"])


def _reason(payload: dict[str, Any] | None) -> str | None:
    if not payload or payload.get("valid"):
        return None
    return str(payload.get("reason", "unavailable"))


@dataclass(frozen=True)
class ScoreAlignmentResult:
    """Metric panel returned for an externally-supplied alignment.

    Mirrors the metric fields of ``PairwiseMetrics`` (no ``raw_sw_score``,
    since no Smith-Waterman recurrence runs). When ``correspondences`` is
    the path Hikoboshi's ``pairwise`` workflow produced for the same
    structures, every field equals its ``PairwiseMetrics`` counterpart.
    """

    rmsd: float | None
    tm_score_query: float | None
    tm_score_target: float | None
    lddt: float | None
    lddt_byA: float | None
    lddt_byB: float | None
    lddt_aln: float | None
    identity: float | None
    coverage_query: float | None
    coverage_target: float | None
    coverage_mean: float | None
    coverage_byA: float | None
    coverage_byB: float | None
    ecs: float | None
    aligned_pairs: int
    invalid_reasons: dict[str, str]

    @classmethod
    def from_core(cls, payload: dict[str, Any]) -> "ScoreAlignmentResult":
        names = (
            "rmsd",
            "tm_score_query",
            "tm_score_target",
            "lddt",
            "lddt_byA",
            "lddt_byB",
            "lddt_aln",
            "identity",
            "coverage_query",
            "coverage_target",
            "coverage_mean",
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
            rmsd=_metric(payload.get("rmsd")),
            tm_score_query=_metric(payload.get("tm_score_query")),
            tm_score_target=_metric(payload.get("tm_score_target")),
            lddt=_metric(payload.get("lddt")),
            lddt_byA=_metric(payload.get("lddt_byA")),
            lddt_byB=_metric(payload.get("lddt_byB")),
            lddt_aln=_metric(payload.get("lddt_aln")),
            identity=_metric(payload.get("identity")),
            coverage_query=_metric(payload.get("coverage_query")),
            coverage_target=_metric(payload.get("coverage_target")),
            coverage_mean=_metric(payload.get("coverage_mean")),
            coverage_byA=_metric(payload.get("coverage_byA")),
            coverage_byB=_metric(payload.get("coverage_byB")),
            ecs=_metric(payload.get("ecs")),
            aligned_pairs=int(payload["aligned_pairs"]),
            invalid_reasons=reasons,
        )


def _normalize_correspondences(
    correspondences: Iterable[Any] | None,
) -> list[Any]:
    if correspondences is None:
        return []
    return list(correspondences)


class ScoreAlignmentNamespace:
    """Score externally-supplied correspondences with Hikoboshi's metric panel.

    ``correspondences`` is a sequence of ``(query_index, target_index)``
    pairs (or dict-shaped entries with ``query_index`` and
    ``target_index`` keys) using zero-based residue indices. Use
    ``hikoboshi.ALIGNMENT_GAP_SENTINEL`` (-1) on either side of a step to
    mark a gap. Out-of-range match indices raise
    ``InvalidArgumentError``.
    """

    def __call__(
        self,
        query: Any,
        target: Any,
        correspondences: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> ScoreAlignmentResult:
        return self.from_structure(
            query, target, correspondences=correspondences, **kwargs
        )

    def from_structure(
        self,
        query: Any,
        target: Any,
        *,
        correspondences: Sequence[Any] | None = None,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        **kwargs: Any,
    ) -> ScoreAlignmentResult:
        pdb_model_id, pdb_model_index = resolve_deprecated_model_kwargs(
            kwargs,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            context="score_alignment.from_structure",
        )
        reject_unknown_kwargs(kwargs, context="score_alignment.from_structure")
        selection = chain_selection(
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
        )
        steps = _normalize_correspondences(correspondences)
        core = load_core()
        try:
            payload = core.score_alignment_from_structure(
                query,
                target,
                correspondences=steps,
                **selection,
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return ScoreAlignmentResult.from_core(payload)

    def from_pdb(
        self,
        query: Any,
        target: Any,
        *,
        correspondences: Sequence[Any] | None = None,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        **kwargs: Any,
    ) -> ScoreAlignmentResult:
        return self.from_structure(
            query,
            target,
            correspondences=correspondences,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            **kwargs,
        )

    def from_cif(
        self,
        query: Any,
        target: Any,
        *,
        correspondences: Sequence[Any] | None = None,
        chain_id: str | None = None,
        chain_index: int | None = None,
        pdb_model_id: str | None = None,
        pdb_model_index: int | None = None,
        **kwargs: Any,
    ) -> ScoreAlignmentResult:
        return self.from_structure(
            query,
            target,
            correspondences=correspondences,
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            **kwargs,
        )


score_alignment = ScoreAlignmentNamespace()


__all__ = ["ScoreAlignmentNamespace", "ScoreAlignmentResult", "score_alignment"]

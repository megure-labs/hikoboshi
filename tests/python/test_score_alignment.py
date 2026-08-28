from __future__ import annotations

import tempfile
from pathlib import Path

import hikoboshi as hiko

from helpers import require, write_simple_pdb


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    query_path = root / "query.pdb"
    target_path = root / "target.pdb"
    write_simple_pdb(query_path, residue_count=3)
    write_simple_pdb(target_path, residue_count=3)

    # Bit-equality vs the pairwise pipeline: compute pairwise to get the
    # internal hard-SW path, then feed that path back through
    # score_alignment and require every metric field to be IEEE bit-equal
    # to the pairwise metrics.
    pairwise_result = hiko.pairwise.from_structure(
        str(query_path),
        str(target_path),
        package="hikoboshi-mpnn-d64",
        mode="hard",
    )
    pairwise_steps = [
        (step.query_index, step.target_index)
        for step in pairwise_result.path.steps
    ]
    scored = hiko.score_alignment(
        str(query_path),
        str(target_path),
        correspondences=pairwise_steps,
    )
    require(
        scored.aligned_pairs == pairwise_result.path.aligned_pairs,
        "score_alignment aligned_pairs must match pairwise.path.aligned_pairs",
    )
    metric_fields = (
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
    for field in metric_fields:
        scored_value = getattr(scored, field)
        pairwise_value = getattr(pairwise_result.metrics, field)
        require(
            scored_value == pairwise_value,
            f"score_alignment.{field} must bit-equal pairwise.metrics.{field} "
            f"(score={scored_value!r}, pairwise={pairwise_value!r})",
        )

    # Empty correspondences: aligned_pairs must be 0. Kabsch-driven metrics
    # (rmsd, tm_score_*) surface InsufficientAlignedPairs because Kabsch
    # needs >= 3 aligned pairs. Canonical (Mariani) lDDT and structure-length
    # coverage are defined as 0 over an empty path with non-zero reference
    # graphs, so they remain valid.
    empty = hiko.score_alignment(
        str(query_path),
        str(target_path),
        correspondences=[],
    )
    require(empty.aligned_pairs == 0, "empty correspondences must have aligned_pairs == 0")
    require(empty.rmsd is None, "empty correspondences must invalidate rmsd")
    require(
        empty.tm_score_query is None and empty.tm_score_target is None,
        "empty correspondences must invalidate the Kabsch-driven TM-scores",
    )
    require(
        empty.invalid_reasons.get("rmsd") == "insufficient_aligned_pairs",
        "empty correspondences must report InsufficientAlignedPairs for rmsd",
    )

    # Out-of-range index must raise InvalidArgumentError.
    try:
        hiko.score_alignment(
            str(query_path),
            str(target_path),
            correspondences=[(0, 99)],  # target has only 3 residues
        )
    except hiko.InvalidArgumentError as exc:
        require(
            "out of range" in exc.detail.lower(),
            "out-of-range index must surface a descriptive error message",
        )
    else:
        raise SystemExit("out-of-range index must raise InvalidArgumentError")

    # Gap sentinel on either side is accepted and contributes no aligned pair.
    gap_result = hiko.score_alignment(
        str(query_path),
        str(target_path),
        correspondences=[
            (0, 0),
            (1, hiko.ALIGNMENT_GAP_SENTINEL),
            (2, 2),
        ],
    )
    require(
        gap_result.aligned_pairs == 2,
        "gap-side sentinels must not count toward aligned_pairs",
    )

    # from_structure namespace and dict-shaped correspondences both work.
    dict_steps = [
        {"query_index": step.query_index, "target_index": step.target_index}
        for step in pairwise_result.path.steps
    ]
    dict_scored = hiko.score_alignment.from_structure(
        str(query_path),
        str(target_path),
        correspondences=dict_steps,
    )
    require(
        dict_scored.lddt == scored.lddt
        and dict_scored.aligned_pairs == scored.aligned_pairs,
        "dict-shaped correspondences must score identically to tuple-shaped",
    )

    # __call__ namespace alias is the same surface as from_structure.
    call_scored = hiko.score_alignment(
        str(query_path),
        str(target_path),
        pairwise_steps,
    )
    require(
        call_scored.lddt == scored.lddt,
        "score_alignment(...) positional surface must match score_alignment(...) kwarg form",
    )

    # The convenience aliases route through the same surface.
    pdb_alias = hiko.score_alignment.from_pdb(
        str(query_path),
        str(target_path),
        correspondences=pairwise_steps,
    )
    require(
        pdb_alias.lddt == scored.lddt,
        "from_pdb alias must route through the same scoring entry point",
    )

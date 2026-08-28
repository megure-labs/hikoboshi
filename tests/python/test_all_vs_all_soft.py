from __future__ import annotations

import math
import os
import tempfile

import hikoboshi as hiko

from helpers import matrix, require

MPNN64_SOFT_GAP_OPEN = -3.21337
MPNN64_SOFT_GAP_EXTENSION = -0.111704


def three_distinct_embeddings():
    a = matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b = matrix([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]])
    c = matrix([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    return [a, b, c]


def gap_sensitive_embeddings():
    query = matrix(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    target = matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return [query, target]


soft_result = hiko.all_vs_all.from_embeddings(
    three_distinct_embeddings(),
    mode="soft",
    temperature=1.0,
)
require(
    len(soft_result.records) == 3,
    "soft all-vs-all must emit three i<j pairs for n=3",
)
expected_pairs = [(0, 1), (0, 2), (1, 2)]
for index, record in enumerate(soft_result.records):
    require(
        (record.query_index, record.target_index) == expected_pairs[index],
        "soft all-vs-all pair order mismatch",
    )
    require(
        math.isfinite(record.result.metrics.raw_sw_score),
        "soft all-vs-all score must be finite",
    )
    require(
        record.result.metrics.soft_sw_score
        == record.result.metrics.raw_sw_score,
        "soft all-vs-all raw score must mirror soft_sw_score",
    )

default_soft_gap = hiko.all_vs_all.from_embeddings(
    gap_sensitive_embeddings(),
    mode="soft",
)
explicit_soft_gap = hiko.all_vs_all.from_embeddings(
    gap_sensitive_embeddings(),
    mode="soft",
    gap_open=MPNN64_SOFT_GAP_OPEN,
    gap_extension=MPNN64_SOFT_GAP_EXTENSION,
)
require(
    math.isclose(
        default_soft_gap.records[0].result.metrics.raw_sw_score,
        explicit_soft_gap.records[0].result.metrics.raw_sw_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ),
    "default soft all-vs-all score must use calibrated soft gaps",
)

default_hard = hiko.all_vs_all.from_embeddings(three_distinct_embeddings())
explicit_hard = hiko.all_vs_all.from_embeddings(
    three_distinct_embeddings(),
    mode="hard",
    temperature=42.0,
)
require(
    len(default_hard.records) == len(explicit_hard.records),
    "default and explicit hard all-vs-all pair counts must match",
)
for default_record, explicit_record in zip(
    default_hard.records,
    explicit_hard.records,
):
    require(
        default_record.result.metrics.raw_sw_score
        == explicit_record.result.metrics.raw_sw_score,
        "hard all-vs-all must ignore temperature",
    )
    require(
        default_record.result.path.aligned_pairs
        == explicit_record.result.path.aligned_pairs,
        "hard all-vs-all path must ignore temperature",
    )
    require(
        default_record.result.metrics.soft_sw_score is None,
        "hard all-vs-all must not populate soft_sw_score",
    )

both = hiko.all_vs_all.from_embeddings(three_distinct_embeddings(), mode="both")
for hard_record, soft_record, both_record in zip(
    default_hard.records,
    soft_result.records,
    both.records,
):
    require(
        both_record.result.metrics.raw_sw_score
        == hard_record.result.metrics.raw_sw_score,
        "both all-vs-all raw score must keep the hard score",
    )
    require(
        both_record.result.path.aligned_pairs
        == hard_record.result.path.aligned_pairs,
        "both all-vs-all path must keep the hard path",
    )
    require(
        both_record.result.metrics.soft_sw_score is not None,
        "both all-vs-all must populate soft_sw_score",
    )
    require(
        math.isclose(
            both_record.result.metrics.soft_sw_score,
            soft_record.result.metrics.raw_sw_score,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "both all-vs-all soft score must match explicit soft mode",
    )

soft_again = hiko.all_vs_all.from_embeddings(
    three_distinct_embeddings(),
    mode="soft",
)
for first, second in zip(soft_result.records, soft_again.records):
    require(
        first.result.metrics.raw_sw_score
        == second.result.metrics.raw_sw_score,
        "soft all-vs-all scores must be deterministic",
    )
    require(
        first.result.path.aligned_pairs == second.result.path.aligned_pairs,
        "soft all-vs-all consensus alignments must be deterministic",
    )

for invalid_mode in ("bogus",):
    try:
        hiko.all_vs_all.from_embeddings(
            three_distinct_embeddings(),
            mode=invalid_mode,
        )
    except hiko.InvalidArgumentError:
        pass
    else:
        raise SystemExit("invalid all-vs-all mode should raise")

try:
    hiko.all_vs_all.from_embeddings(
        three_distinct_embeddings(),
        mode="soft",
        temperature=0.0,
    )
except hiko.InvalidArgumentError:
    pass
else:
    raise SystemExit("soft all-vs-all with temperature=0 should raise")

with tempfile.TemporaryDirectory() as tmp:
    soft_path = os.path.join(tmp, "soft.tsv")
    hiko.all_vs_all.to_tsv_from_embeddings(
        three_distinct_embeddings(),
        soft_path,
        mode="soft",
    )
    with open(soft_path, "r", encoding="ascii") as handle:
        soft_lines = handle.read().splitlines()
    header = soft_lines[0].split("\t")
    for column in (
        "soft_sw_score",
        "sw_per_query_len",
        "sw_per_target_len",
        "sw_per_aligned",
    ):
        require(column in header, f"soft TSV must include {column}")
    default_path = os.path.join(tmp, "default.tsv")
    hard_path = os.path.join(tmp, "hard.tsv")
    hiko.all_vs_all.to_tsv_from_embeddings(
        three_distinct_embeddings(),
        default_path,
    )
    hiko.all_vs_all.to_tsv_from_embeddings(
        three_distinct_embeddings(),
        hard_path,
        mode="hard",
    )
    with open(default_path, "rb") as default_handle, open(
        hard_path,
        "rb",
    ) as hard_handle:
        require(
            default_handle.read() == hard_handle.read(),
            "default streaming TSV must be byte-identical to hard mode",
        )

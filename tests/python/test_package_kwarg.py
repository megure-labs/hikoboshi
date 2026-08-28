from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable
import warnings

import hikoboshi as hiko

from helpers import matrix, require, write_two_model_pdb


def require_invalid_argument(
    call: Callable[[], Any],
    *,
    expected: tuple[str, ...],
) -> None:
    try:
        call()
    except hiko.InvalidArgumentError as exc:
        for text in expected:
            require(text in exc.detail, f"missing {text!r} in {exc.detail!r}")
    else:
        raise SystemExit("expected InvalidArgumentError")


def require_deprecation(call: Callable[[], Any], *, expected: str) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = call()
    require(
        any(
            issubclass(item.category, DeprecationWarning)
            and expected in str(item.message)
            for item in caught
        ),
        f"missing DeprecationWarning containing {expected!r}",
    )
    return result


def require_encoded(result: hiko.EncodeResult, *, message: str) -> None:
    require(len(result.embeddings) == 2, message)
    require(result.metadata is not None, "encode metadata missing")
    require(result.metadata["dimension"] == 64, "encode dimension mismatch")


query = matrix([[1.0, 0.0], [0.0, 1.0]])
target = matrix([[1.0, 0.0], [0.0, 1.0]])

# Pin to hard mode: the literal score golden 2.0 is the hard-SW raw_sw_score
# for a 2x2 identity-similarity matrix. Hikoboshi 0.1.0 defaults to hard
# Smith-Waterman; this test continues to validate hard-SW package routing.
default = hiko.pairwise.from_embeddings(query, target, mode="hard")
require(default.metrics.raw_sw_score == 2.0, "default package score mismatch")

canonical = hiko.pairwise.from_embeddings(
    query,
    target,
    package="hikoboshi-mpnn-d64",
    mode="hard",
)
require(canonical.metrics.raw_sw_score == 2.0, "canonical package score mismatch")

old_name = require_deprecation(
    lambda: hiko.pairwise.from_embeddings(
        query,
        target,
        package="Hikoboshi-MPNN-64",
        mode="hard",
    ),
    expected="package alias 'Hikoboshi-MPNN-64' is deprecated",
)
require(old_name.metrics.raw_sw_score == 2.0, "old package alias score mismatch")

alias = require_deprecation(
    lambda: hiko.pairwise.from_embeddings(
        query,
        target,
        package="mpnn64",
        mode="hard",
    ),
    expected="package alias 'mpnn64' is deprecated",
)
require(alias.metrics.raw_sw_score == 2.0, "package alias score mismatch")

all_vs_all = require_deprecation(
    lambda: hiko.all_vs_all.from_embeddings(
        [query, target],
        package="mpnn-64",
    ),
    expected="package alias 'mpnn-64' is deprecated",
)
require(
    [(record.query_index, record.target_index) for record in all_vs_all.records]
    == [(0, 1)],
    "all-vs-all package alias pair order mismatch",
)

default_all_vs_all = hiko.all_vs_all.from_embeddings([query, target])
require(
    [(record.query_index, record.target_index) for record in default_all_vs_all.records]
    == [(0, 1)],
    "default all-vs-all package pair order mismatch",
)

require_invalid_argument(
    lambda: hiko.pairwise.from_embeddings(query, target, package="unknown-package"),
    expected=(
        "unknown Hikoboshi package id",
        "available compiled package IDs/aliases",
        "hikoboshi-mpnn-d64",
        "mpnn64",
        "mpnn-64",
    ),
)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    first = root / "first.pdb"
    second = root / "second.pdb"
    write_two_model_pdb(first)
    write_two_model_pdb(second)
    paths = [str(first), str(second)]

    require_encoded(
        hiko.encode.from_structure(str(first), package="hikoboshi-mpnn-d64"),
        message="canonical structure encode did not run",
    )
    require_encoded(
        hiko.encode.from_structure(str(first)),
        message="default structure encode did not run",
    )
    require_invalid_argument(
        lambda: hiko.encode.from_structure(str(first), package="unknown-package"),
        expected=("unknown Hikoboshi package id", "hikoboshi-mpnn-d64"),
    )

    by_index = hiko.all_vs_all.from_pdb(
        paths,
        package="hikoboshi-mpnn-d64",
        pdb_model_index=2,
    )
    require(len(by_index.records) == 1, "pdb_model_index route did not run")

    by_id = hiko.all_vs_all.from_pdb(
        paths,
        package="hikoboshi-mpnn-d64",
        pdb_model_id="2",
    )
    require(len(by_id.records) == 1, "pdb_model_id route did not run")

    legacy_by_index = require_deprecation(
        lambda: hiko.all_vs_all.from_pdb(
            paths,
            package="hikoboshi-mpnn-d64",
            model_index=2,
        ),
        expected="model_index is deprecated",
    )
    require(len(legacy_by_index.records) == 1, "legacy model_index route did not run")

    legacy_by_id = require_deprecation(
        lambda: hiko.all_vs_all.from_pdb(
            paths,
            package="hikoboshi-mpnn-d64",
            model_id="2",
        ),
        expected="model_id is deprecated",
    )
    require(len(legacy_by_id.records) == 1, "legacy model_id route did not run")

    require_invalid_argument(
        lambda: hiko.all_vs_all.from_pdb(paths, pdb_model_index=99),
        expected=("model",),
    )
    require_invalid_argument(
        lambda: hiko.all_vs_all.from_pdb(
            paths,
            pdb_model_index=2,
            model_index=2,
        ),
        expected=("pdb_model_index", "model_index"),
    )
    require_invalid_argument(
        lambda: hiko.all_vs_all.from_pdb(
            paths,
            pdb_model_id="2",
            pdb_model_index=2,
        ),
        expected=("pdb_model_id", "pdb_model_index"),
    )

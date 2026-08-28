#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import hikoboshi as hiko

PAIR_COUNT = 1000
UNIQUE_PROTEINS = 200
THRESHOLD = 0.25


def _pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index in range(PAIR_COUNT):
        query = f"p{(index * 37) % UNIQUE_PROTEINS:03d}.pdb"
        target = f"p{(index * 53 + 17) % UNIQUE_PROTEINS:03d}.pdb"
        pairs.append((query, target))
    return pairs


def _write_simple_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       1.500   1.500   0.000  1.00  0.00           C\n"
        "ATOM      6  N   ALA A   2       4.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      7  CA  ALA A   2       5.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      8  C   ALA A   2       6.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      9  O   ALA A   2       7.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM     10  CB  ALA A   2       5.500   1.500   0.000  1.00  0.00           C\n"
        "END\n",
        encoding="ascii",
    )


def _output_path() -> Path:
    env_path = os.environ.get("HIKOBOSHI_PAIR_LIST_PERF_JSON")
    if env_path:
        return Path(env_path)
    source_root = Path(
        os.environ.get("HIKOBOSHI_SOURCE_ROOT", Path(__file__).resolve().parents[2])
    )
    return source_root / "bench" / "pair_list_perf_smoke.json"


def main() -> int:
    pairs = _pairs()

    with tempfile.TemporaryDirectory() as tmp:
        pdb_dir = Path(tmp) / "pdbs"
        pdb_dir.mkdir()
        paths: dict[str, Path] = {}
        for index in range(UNIQUE_PROTEINS):
            path = pdb_dir / f"p{index:03d}.pdb"
            _write_simple_pdb(path)
            paths[path.name] = path

        # Warm the extension, package registry, and code paths outside the timed
        # regions. The measured comparison is then one pair-list call versus
        # the same 1000 pairs as independent pairwise calls.
        hiko.pairwise.from_structure(
            paths[pairs[0][0]],
            paths[pairs[0][1]],
            package="hikoboshi-mpnn-d64",
            mode="hard",
        )

        pairwise_start = time.perf_counter()
        for query, target in pairs:
            hiko.pairwise.from_structure(
                paths[query],
                paths[target],
                package="hikoboshi-mpnn-d64",
                mode="hard",
            )
        pairwise_wall = time.perf_counter() - pairwise_start

        pair_list_start = time.perf_counter()
        pair_list_results = hiko.pair_list_from_structure(
            pairs,
            pdb_dir,
            package="hikoboshi-mpnn-d64",
            mode="hard",
        )
        pair_list_wall = time.perf_counter() - pair_list_start

    if len(pair_list_results) != PAIR_COUNT:
        raise SystemExit(
            f"pair-list returned {len(pair_list_results)} results, want {PAIR_COUNT}"
        )
    if pairwise_wall <= 0.0:
        raise SystemExit("pairwise wall time must be positive")
    ratio = pair_list_wall / pairwise_wall
    record = {
        "schema_version": 1,
        "fixture": {
            "pair_count": PAIR_COUNT,
            "unique_proteins": UNIQUE_PROTEINS,
            "route": "structure",
            "residue_count": 2,
            "package": "hikoboshi-mpnn-d64",
            "mode": "hard",
        },
        "pair_list_wall_seconds": pair_list_wall,
        "n_pairwise_wall_seconds": pairwise_wall,
        "ratio": ratio,
        "threshold": THRESHOLD,
        "pass": ratio < THRESHOLD,
    }
    output = _output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    if ratio >= THRESHOLD:
        raise SystemExit(
            "pair-list perf smoke failed: "
            f"ratio={ratio:.6f}, threshold={THRESHOLD:.6f}, "
            f"pair_list={pair_list_wall:.6f}s, pairwise={pairwise_wall:.6f}s"
        )
    print(
        "pair_list_perf_smoke: ok "
        f"pairs={PAIR_COUNT} unique={UNIQUE_PROTEINS} "
        f"ratio={ratio:.6f} pair_list={pair_list_wall:.6f}s "
        f"pairwise={pairwise_wall:.6f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

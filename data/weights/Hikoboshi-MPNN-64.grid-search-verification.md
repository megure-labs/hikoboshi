# Hikoboshi-MPNN-64 Hard-SW Gap Grid-Search Verification

Date verified: 2026-04-29

Verifier: casey

Human attestation: casey confirmed in the Hikoboshi development chat on
2026-04-29 that the recovered hard-SW grid search used the same model family,
same scoring mode, same hard local affine Smith-Waterman recurrence, and same
affine-gap convention that Hikoboshi 0.1.0 ships.

## Verified Runtime Defaults

Hikoboshi 0.1.0 ships these hard-SW defaults for the compiled
`Hikoboshi-MPNN-64` package:

```text
gap_open = -1.40000
gap_ext  = -0.150000
```

The recovered fine-grid best row records the exact source values:

```text
gap_open  = -1.399999999999999
gap_ext   = -0.15000000000000008
precision = 0.6339086883983029
recall    = 0.6510185040894891
f1        = 0.6420386534701755
```

## Evidence Reviewed

Recovered local artifacts were present in multiple remediation/recovery trees.
The representative evidence set used for this verification was:

- `remediation_verify_20260424/repo/scripts/grid_search_fine.py`
- `remediation_verify_20260424/repo/scripts/grid_search_fine_results.csv`
- `remediation_verify_20260424/repo/scripts/grid_search_gap_penalties.py`
- `remediation_verify_20260424/repo/scripts/grid_search_stratified.log`
- `remediation_verify_20260424/repo/scripts/hard_sw_kernel.cpp`

Representative SHA-256 values:

```text
grid_search_fine.py          947f609d2723bc6f4649b03645ee6226de3c5416fb8a513ce58acd20c34e62c3
grid_search_fine_results.csv 1e099107844acccb3b89b87996f3382d7adbbd6afb8c1f2daa3271c559a6561b
hard_sw_kernel.cpp           9c74754ef6daa2fe877e53ce3abd0dfe0be219670009f5de00f245e32da0bbdf
```

The exact best row above appeared identically in the recovered
`grid_search_fine_results.csv` copies inspected across the available recovery
trees.

## Cross-References Confirmed

Model family:
The grid search consumed the historical `mpnn_64_epoch_50.h5` embedding
artifact. Hikoboshi 0.1.0 embeds the corresponding `Hikoboshi-MPNN-64` package
bytes, recorded in `Hikoboshi-MPNN-64.provenance.json` as
`archive-embedded-header-0348104439a78dae`.

Scoring mode:
The grid-search scripts compute the similarity matrix as raw embedding dot
product, `emb1 @ emb2.T`. No cosine normalization is applied. This matches the
`raw_dot_v1` scoring family in the compiled package manifest.

Alignment recurrence:
The recovered kernel implements hard local affine Smith-Waterman with states
`M`, `I`, and `D`, a local restart in `M`, and a global best-cell scan over all
three states.

Gap convention:
The recurrence opens a gap with `gap_open` and extends an already-open gap with
`gap_ext`, so a gap of length `k` has cost:

```text
gap_open + (k - 1) * gap_ext
```

This is also the convention used by the orihime/dp affine implementations that
informed the original training and grid-search path.

Validation data:
The recovered grid-search scripts read validation pairs from the historical
`SCOPe40-test` validation dataset and use metadata index mapping to look up the
corresponding embedding rows.

Historical path-name audit:
The historical embedding artifact path contained the string `data-leakage`.
The reviewed scripts use that path as an embedding artifact location while the
validation pairs come from the separate `SCOPe40-test` validation inputs. The
verifier signed off that this is historical path naming rather than validation
contamination.

## Result

The hard-SW gap defaults in the `Hikoboshi-MPNN-64` manifest are verified for
Hikoboshi 0.1.0 release-cut preparation:

- `hard_local_affine_sw_v1`
- `raw_dot_v1`
- `gap_open_plus_k_minus_1_gap_ext`
- `gap_open = -1.40000`
- `gap_ext = -0.150000`

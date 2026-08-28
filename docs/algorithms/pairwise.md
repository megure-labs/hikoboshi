# Pairwise Alignment

Hikoboshi pairwise alignment compares two protein inputs and returns one local
alignment path with summary metrics. The public routes are:

- CLI: `hikoboshi pairwise [structure|pdb|cif|coords|embeddings|sequence] <query> <target>`
- Python: `hikoboshi.pairwise(...)`
- Python explicit routes: `from_structure`, `from_pdb`, `from_cif`,
  `from_coords`, `from_embeddings`, and `from_sequence`

## Inputs

Structure routes load PDB or mmCIF/CIF files, select one polymer chain, normalize
the backbone atom set, and encode the selected residues with the compiled
`Hikoboshi-MPNN-64` package.

Coordinate routes accept canonical backbone coordinates. Embedding routes accept
precomputed residue embeddings and skip structure encoding.

Sequence routes accept amino-acid strings or FASTA-backed inputs and encode
them with the compiled `Hikoboshi-ESM2-8M` package. Sequence results can write
aligned FASTA but do not report coordinate-dependent structural metrics.

## Scoring

For residue embeddings `A` and `B`, Hikoboshi builds a raw dot-product score
matrix:

```text
S_AB[p, q] = dot(A[p], B[q])
```

The score matrix is not cosine-normalized in `0.1.0`.

## Alignment

Hard mode uses local affine Smith-Waterman and returns the discrete alignment
path. Soft mode evaluates the differentiable partition-function path; `both`
runs both branches and retains the hard path as the primary alignment. Soft
and both modes are approximately 6–10× slower than hard mode at the same
problem size.

Each package and alignment mode has its own calibrated pair. Public values are
the six-significant-figure projections recorded in the package provenance:

| Package | Mode | Temperature | `gap_open` | `gap_extension` |
| --- | --- | ---: | ---: | ---: |
| `hikoboshi-mpnn-d64` | hard | — | `-1.40000` | `-0.150000` |
| `hikoboshi-mpnn-d64` | soft | `1.00000` | `-3.21337` | `-0.111704` |
| `hikoboshi-esm2-8m` | hard | — | `-1.01982` | `+0.225736` |
| `hikoboshi-esm2-8m` | soft | `1.00000` | `-6.72805` | `-0.0159468` |

The positive ESM2 hard-mode extension is intentional: it is the terminal value
of the recorded near-zero-temperature hard-SW anneal, not the ESM2 T=1 soft
value. Package validation pins all four pairs independently.

A gap of length `k` costs:

```text
gap_open + (k - 1) * gap_extension
```

Package descriptors carry separate calibrated hard and soft gap values, and
explicit request values override them. The public result contains an
ordered `AlignmentPath`; gap steps use `-1` for the missing side. The raw
Smith-Waterman score is reported separately from per-step residue-pair scores.

## Outputs

Pairwise summaries include:

- raw Smith-Waterman score
- aligned residue-pair count
- query, target, and mean coverage
- identity when sequence metadata is available
- RMSD, TM-score, and lDDT when observed CA coordinate metadata is available
- ECS when its required metadata is available

Unavailable metrics are explicit. The CLI prints `NA`; Python result objects
use `None` and expose invalid-metric reasons.

For structure inputs, the CLI can write alignment FASTA and superposed PDB:

```bash
hikoboshi pairwise pdb query.pdb target.pdb \
  --fasta alignment.fasta \
  --pdb superposed.pdb \
  --summary pairwise.tsv
```

Embedding-only inputs can report score, path, and coverage. FASTA, PDB, and
structure-derived metrics require sequence or structure metadata.

## Python Example

```python
from array import array
import hikoboshi as hkbs

def matrix(rows):
    flat = array("f", [value for row in rows for value in row])
    return memoryview(flat).cast("B").cast("f", shape=(len(rows), len(rows[0])))

result = hkbs.pairwise.from_embeddings(
    matrix([[1.0, 0.0], [0.0, 1.0]]),
    matrix([[1.0, 0.0], [0.0, 1.0]]),
    query_metadata={"residue_codes": "AC"},
    target_metadata={"residue_codes": "AC"},
)

print(result.metrics.raw_sw_score)
print(result.metrics.identity)
```

## Scope

Hikoboshi `0.1.0` does not expose global or semiglobal alignment selection,
tree building, MSA workflows, database search, GPU backends, or SIMD release
backends. Soft mode returns its score and thresholded consensus alignment.

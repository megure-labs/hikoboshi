# Symmetric All-Vs-All Enumeration

Hikoboshi all-vs-all runs the pairwise alignment pipeline over a set of inputs and
emits pair records. It is symmetric pair enumeration, not clustering, database
search, guide-tree construction, or MSA preparation.

Public routes:

- CLI: `hikoboshi all-vs-all <structure|pdb|cif|coords|embeddings|sequence> <inputs...>`
- Python: `hikoboshi.all_vs_all(...)`
- Python explicit routes: `from_structure`, `from_pdb`, `from_cif`,
  `from_coords`, `from_embeddings`, and `from_sequence`

## Pair Order

By default, Hikoboshi emits records for `i < j`:

```text
(0,1), (0,2), ..., (0,N-1), (1,2), ...
```

With `include_self` in Python or `--include-self` in the CLI, Hikoboshi emits
`i <= j`:

```text
(0,0), (0,1), ..., (1,1), ...
```

No default full matrix is produced. The C++ API has a streaming sink boundary,
and the Python and CLI adapters expose collected records or TSV rows.

Pair order is stable at every thread count. Threaded workers may compute
different ranges concurrently, but Hikoboshi delivers records to the public sink,
CLI summary, and Python result in the same lexicographic order as the serial
path.

## Threading

All-vs-all threading is coarse-grained scheduling across inputs and pair
records. Hikoboshi does not thread a single pairwise alignment.

The CLI option is `--threads N`:

```bash
hikoboshi all-vs-all embeddings a.npy b.npy c.npy --summary all.tsv --threads 8
```

Python all-vs-all routes use `thread_count=`:

```python
result = hkbs.all_vs_all.from_embeddings(inputs, thread_count=8)
```

Thread-count values have the same meaning in the CLI and Python:

- `0` selects automatic mode and is the default.
- `1` forces the serial path.
- `N > 1` requests `N` worker slots.

Automatic mode uses the host hardware thread count as the request, then applies
the same eligibility and memory checks as an explicit request. Small jobs may
therefore run serially even when auto mode or a larger value is selected.

Embedding- and sequence-based all-vs-all can parallelize the pair phase when there are at
least `45` emitted pairs. Structure and coordinate all-vs-all can also
parallelize the encoding phase when there are at least `4` inputs, subject to
the available workspace budget. If the structure encoding workspace would be
too large, Hikoboshi reduces the effective worker count or uses the serial path
instead of overcommitting memory.

## Outputs

The CLI summary is TSV with one row per pair. For structure inputs, `--output-dir`
writes per-pair artifacts:

```text
output/
  alignments/
    <pair-id>.fasta
  pdb/
    <pair-id>.pdb
```

Embedding-only all-vs-all reports score, path, and embedding-length coverage.
FASTA/PDB artifacts and structure-derived metrics require sequence or structure
metadata.

## CLI Example

```bash
hikoboshi all-vs-all embeddings a.npy b.npy c.npy --summary all-vs-all.tsv
```

For structure inputs:

```bash
hikoboshi all-vs-all pdb a.pdb b.pdb c.pdb --output-dir output
```

## Python Example

```python
from array import array
import hikoboshi as hkbs

def scalar_embedding(value):
    flat = array("f", [value])
    return memoryview(flat).cast("B").cast("f", shape=(1, 1))

result = hkbs.all_vs_all.from_embeddings(
    [scalar_embedding(1.0), scalar_embedding(2.0), scalar_embedding(3.0)]
)

for record in result.records:
    print(record.query_index, record.target_index, record.result.metrics.raw_sw_score)
```

## Scope

All-vs-all uses the same hard, soft, or combined local affine Smith-Waterman
semantics as the pairwise route. Hard mode remains the default and the only
mode with a discrete primary path. Hikoboshi `0.1.0` does not define
distance-matrix, guide-tree, clustering, MSA, database-search, GPU, or SIMD
release behavior.

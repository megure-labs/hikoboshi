# Pair-List Alignment

Hikoboshi pair-list alignment runs the pairwise pipeline over a caller-supplied
list of named `(query, target)` pairs. Use it when you know exactly which pairs
you want and those pairs reuse proteins across the list.

## Choosing A Mode

| Mode | Use When | Pair Set | Encoding Work | Output Order |
| --- | --- | --- | --- | --- |
| `pairwise` | You need one alignment. | One query and one target. | Encodes the two inputs for that call. | Single result. |
| `pair-list` | You need a chosen list of named pairs. | Exactly the TSV or Python pair list. | Encodes each unique named protein once. | Input pair order. |
| `all-vs-all` | You need every pair in a set. | Generated symmetric combinations. | Encodes every input once. | Lexicographic `(i, j)` order. |

Pair-list is not database search: it does not rank hits, apply top-K filters, or
generate pairs that were not supplied.

## CLI

The CLI command is:

```bash
hikoboshi pair-list --pairs pairs.tsv --fasta proteins.fa --summary out.tsv
```

For structure inputs, pass a directory containing PDB or mmCIF files instead of
`--fasta`:

```bash
hikoboshi pair-list --pairs pairs.tsv pdb_dir --summary out.tsv
```

Options:

- `--pairs FILE.tsv`: required two-column TSV with `query_id<TAB>target_id`.
- `--fasta FILE.fa`: named sequence FASTA source.
- `--summary PATH`: write the TSV summary to a file as well as stdout.
- `--package NAME`: compiled package ID or alias. Sequence pair-list defaults
  to `esm2-8m`; structure pair-list defaults to `Hikoboshi-MPNN-64`.
- `--parity-mode strict|fast`: sequence package parity selector.

The pair TSV parser skips blank lines and lines whose first non-space character
is `#`. Each remaining line must contain exactly two tab-separated fields.
Protein IDs are case-sensitive. Duplicate input pairs are preserved and produce
duplicate output rows.

Example TSV:

```text
# query_id	target_id
alpha	beta
gamma	alpha
alpha	beta
```

The summary schema matches `hikoboshi all-vs-all`: one row per pair with
`query_index`, `target_index`, `pair_id`, score, path, metric, and artifact
columns. For pair-list, rows are emitted in the same order as the TSV.

## Python

The public Python entry points return `list[hikoboshi.PairwiseResult]`, one result
per input pair in input order:

```python
import hikoboshi as hkbs

pairs = [("alpha", "beta"), ("gamma", "alpha")]
results = hkbs.pair_list_from_sequence(pairs, "proteins.fa")

for result in results:
    print(result.metrics.raw_sw_score, result.path.aligned_pairs)
```

Structure mode uses a PDB/mmCIF directory whose loaded structure IDs are the
filenames:

```python
results = hkbs.pair_list_from_structure(
    [("query.pdb", "target.pdb")],
    "pdb_dir",
)
```

The lower-level routes `pair_list_from_embeddings` and `pair_list_from_coords`
mirror the Python all-vs-all input families. Embedding pair-list metadata must
provide `input_id` or `source_id` per embedding so the string pairs can be
resolved.

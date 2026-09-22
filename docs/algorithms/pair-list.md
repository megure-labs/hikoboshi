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

## Bounded result storage

The CLI releases each alignment result after writing its artifacts and rendering
its summary row. Parallel pair execution stages at most 1,024 records, reduced
further to target 64 MiB of reserved record/traceback storage. One exceptionally
large record may exceed that target. Input pair names, resolved indices, loaded
structures, embeddings and per-worker alignment scratch remain resident; this
is a bound on result staging, not a constant-memory guarantee for the entire job.

Summary rows are spooled to one anonymous temporary file and published to stdout
and `--summary` only after successful alignment and artifact generation. Allow
temporary disk space for one copy of the rendered TSV, including with `--summary`.
Temporary files close automatically. Per-pair artifacts are written as results
arrive and may remain after a later failure, as with partial artifact write
failures in the collecting workflow.

C++ callers can consume results directly without the CLI's summary spool:

```cpp
struct Sink final : hikoboshi::api::PairwiseResultSink {
  hikoboshi::universal::Status receive(
      const hikoboshi::api::PairwiseResultRecord& record) override {
    // Consume record here; copy it only if it must outlive this callback.
    return hikoboshi::universal::ok_status();
  }
};
Sink sink;
auto status = engine.pair_list(request, sink);
```

The overloads accept structure, coordinate, embedding and sequence pair-list
requests. Callbacks run serially on the calling thread, in caller pair order,
with original source indices and duplicate/reverse pairs preserved. Returning a
non-OK status stops emission and propagates that status. Input-ID validation and
encoding complete before any result callback; a later alignment or sink failure
can leave an emitted prefix. A sink must not re-enter the same engine while its
request is active.

`Engine::collect_pair_list` and Python's list-returning functions remain
collecting interfaces: their returned result storage grows with pair count.
Use the native CLI or the C++ sink interface for bounded result staging.

## Reuse encodings across jobs

Use `--embedding-cache DIR` for persistent structure embeddings while retaining
geometry and artifacts. See [cache workflow and identity](embedding-cache.md).

## Referenced structure loading

The structure CLI enumerates and sorts all supported directory entries, resolves
all requested IDs, then parses only referenced files. Result source indices
still refer to that full sorted directory, including unreferenced entries.
Duplicate/reversed/self pairs retain input order and one encoding per referenced
protein. A name is case-sensitive and follows the same filename rule as the
structure loader, including its historical slash/backslash handling.

Missing or ambiguous requested IDs fail before parsing, choosing the first bad
query/target ID in pair-list order. Once IDs resolve, multiple parsing failures
select the earliest referenced file in sorted-directory order, independently of
worker count. Unreferenced malformed files and unreferenced ambiguous IDs are
ignored. An empty pair list parses no structures and emits a header; directory
discovery/readability and the existing supported-file checks still apply.
All-vs-all continues to parse every input. These rules also apply with an
embedding cache, because structure inputs are resolved before cache lookup.

## Summary reuse

With `--summary PATH`, each record is formatted once into one anonymous temporary
TSV spool. After successful processing the same bytes are replayed to stdout and
the summary file. Temporary TSV storage is one output's size; both requested
destinations still receive the complete result. The file is opened after stdout
publication, retaining the existing behavior on an unwritable summary path.

All-vs-all streaming reuses formatted metric strings and callback values for
its output destinations. The collected structure-artifact route also spools one
rendering when a summary file is requested. Formatting retains column order,
precision, NA values, hard/soft schemas and artifact paths.

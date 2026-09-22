# Reuse structure embeddings across CLI jobs

Structure pair-list and structure/coordinate all-vs-all support an optional local
embedding cache. Give consecutive jobs the same `--embedding-cache DIR`:

```bash
hikoboshi pair-list --pairs first.tsv pdbs/ --threads 16 --embedding-cache embeddings/
hikoboshi pair-list --pairs next.tsv pdbs/ --threads 16 --embedding-cache embeddings/
hikoboshi all-vs-all structure pdbs/ --threads 16 --embedding-cache embeddings/
```

The first job encodes missing proteins and saves their embeddings. Later jobs
load valid embeddings and skip both their inference and encoder scratch
allocation. Newly referenced or changed proteins are encoded as needed. The
cache is off by default. It currently supports the MPNN structure/coordinate
batch routes; sequence and embedding-input routes reject the option.

Structures are still parsed, and coordinates and residue metadata remain live.
Geometry metrics, alignment paths, scores and requested FASTA/PDB artifacts use
the same pipeline as uncached execution. This cache does not remove input
loading or pair alignment costs. Reuse within one job already works; a cold
cache cannot eliminate that job's first encoding pass.

Counters appear on stderr as `embedding-cache: hits=... misses=... writes=...
rejected=...`. TSV stdout and `--summary` retain their schemas. A rejected
entry counts as a miss and is recomputed; directory/read/write failures stop the
job with an explicit error. Successfully published entries may remain after a
later job failure.

## Identity and storage

An entry is bound to the running executable's SHA-256 (including embedded model
code/weights and compiler targeting), the canonical package/checksum, the GEMM
environment selector and build default, local OS/kernel/architecture/hostname,
and the exact normalized coordinates and atom provenance consumed by the
encoder. Different file paths with identical encoder inputs can reuse entries.
Live residue metadata supplies sequence identity and output labels; it is not
replaced by cached metadata. Changing pairs, gap scores or hard/soft alignment
mode does not invalidate an encoding. Fast and strict GEMM selections do.

A rebuilt or byte-modified executable gets a new namespace, even if its model
or calculations are unchanged. The CLI identifies the executable through the
operating system, not its command name or file timestamp. Cache portability
across binaries or machines is deliberately not promised. Linux and macOS
support this workflow; other platforms report an unsupported identity/write
operation. No claim of NIST validation is made for the independently implemented
SHA-256 content checksum; its mathematical specification is
[FIPS 180-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf).

Files contain a versioned header, input/encoder digest, payload shape, payload
SHA-256 and native float32 embeddings. Readers require exact shape/length,
matching identity/checksum and finite values. Truncated, stale or corrupt
entries are rejected. Same-directory temporary files and atomic rename let
separate processes share a cache; two cold jobs may redundantly encode the same
protein, but readers never consume a partially published entry. A process crash
can leave `.pending-*` files, which readers ignore. Publication is atomic;
there is no power-loss persistence guarantee or automatic cleanup/eviction.

At width 64, payload storage is 256 bytes per residue, plus 80 bytes per entry
and filesystem overhead. Old namespaces remain until explicitly removed.
Delete cache directories when no jobs are using them to reclaim disk space;
future jobs will recompute entries. Treat a cache as derived local data, not a
model or structure archive, and do not modify it while a job runs.

## C++ integration

`EngineConfig::structure_embedding_cache` accepts a borrowed, thread-safe
`universal::StructureEmbeddingCache` for structure/coordinate pair-list and
all-vs-all. Its owner must outlive the synchronous call and bind the cache to
all encoder weights, settings and numerical execution identity. The engine
retains its in-memory contract; an IO adapter owns persistence. Callbacks may
run concurrently, cannot retain borrowed spans, and must fill every output
value on a hit. A miss leaves the span unspecified until inference overwrites
it. Other engine routes ignore this batch cache.

`io::DiskStructureEmbeddingCache` provides the local disk implementation. Its
`prepare` identity is caller-supplied; custom C++ users must not reuse a cache
with different weights or execution settings under the same identity. The CLI
constructs this identity automatically.

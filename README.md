<h1 align="center">Hikoboshi</h1>

<p align="center">
  <strong>One native suite for search, alignment, MSA, and the rest of bioinformatics</strong><br>
  Pairwise and all-vs-all alignment today; database search, multiple sequence alignment, phylogenetics, design, and more on one C++ foundation.
</p>

<p align="center">
  <a href="https://github.com/megure-labs/hikoboshi/actions/workflows/ci.yml"><img src="https://github.com/megure-labs/hikoboshi/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://isocpp.org/"><img src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg" alt="C++17"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

Hikoboshi is Megure Labs' native bioinformatics suite. Its goal is to replace
the fragmented collection of tools used for biological sequence and structure
analysis with one CLI, Python API, and C++ core. Version 0.1.0 starts with
learned sequence and structure encoders, pairwise and all-vs-all alignment,
scoring, artifact generation, and ProteinMPNN inverse folding. The same core
will support database search, multiple sequence alignment, phylogenetics,
clustering, and more.

The model packages and tokenizers are embedded and verified at build time, so
the compiled core has no PyTorch or TensorFlow runtime dependency and performs
no model download on first use. The command, Python package, and C++ namespace
are all named `hikoboshi`; Python examples use the conventional short alias
`hkbs`.

> **API stability:** Hikoboshi is pre-1.0 software. Releases before `1.0.0` are
> not guaranteed to be backward compatible; pin an exact version when a stable
> CLI, Python, or C++ interface is required.

## What runs end to end in 0.1.0

| Input or task | Native pipeline | Result |
| --- | --- | --- |
| Protein structure | PDB/mmCIF → normalized backbone → `Hikoboshi-MPNN-64` → residue embeddings → alignment | Path, score, coverage, identity, RMSD, TM-score, lDDT, FASTA, superposed PDB |
| Protein sequence | FASTA/raw sequence → `Hikoboshi-ESM2-8M` → residue embeddings → alignment | Path, score, coverage, aligned FASTA |
| Coordinates | Canonical `[L, 5, 3]` backbone → MPNN → alignment | The structure route without file parsing |
| Embeddings | Caller-provided `[L, D]` matrices → alignment | Path, score, and coverage without re-encoding |
| Protein design | Backbone → ProteinMPNN `v_48_020` → autoregressive decoding | Designed sequences and optional per-position log probabilities |

The same engine powers one pair, a caller-selected pair list, or every
`i < j` pair in a collection. Pair-list and all-vs-all workloads reuse
encodings instead of recomputing a protein for every comparison.

## Build and install

**PyPI and Conda packages are coming soon.** Hikoboshi `0.1.0` is currently a
source release built with Meson. For the native CLI and C++ library:

```bash
git clone https://github.com/megure-labs/hikoboshi.git
cd hikoboshi
meson setup builddir -Dbuildtype=release -Dhikoboshi_python_api=false
meson compile -C builddir -j 8
meson test -C builddir --print-errorlogs
./builddir/hikoboshi info
```

For the Python package and compiled extension:

```bash
python -m pip install .
```

The native `hikoboshi` executable does not embed or link CPython. Meson and the
checked model-blob generator use Python while building from source, but Python
is not required to run the resulting CLI.

The Python package has no required third-party runtime packages. Install the
optional NumPy adapter with:

```bash
python -m pip install '.[numpy]'
```

Requirements:

- Python 3.10 or newer to run Meson and the deterministic model-blob generator,
  and for the optional Python API;
- Meson 1.1 or newer and `meson-python>=0.15`; and
- a C++17 compiler.

For editable installs, direct build-tree imports, and build options, see
[Building from source](docs/source-build.md).

## Quick start

Align two structures with the embedded MPNN encoder and write every available
artifact:

```bash
./builddir/hikoboshi pairwise structure query.pdb target.pdb \
  --package hikoboshi-mpnn-d64 \
  --mode hard \
  --summary alignment.tsv \
  --fasta alignment.fasta \
  --pdb superposed.pdb
```

Align two sequences with the embedded ESM-2 encoder:

```bash
./builddir/hikoboshi pairwise sequence ACDEFGHIK ACDEYGHIK \
  --package hikoboshi-esm2-8m \
  --summary sequence-alignment.tsv
```

Design sequences for a backbone with the embedded ProteinMPNN package:

```bash
./builddir/hikoboshi design \
  --pdb backbone.pdb \
  --out designs.fasta \
  --num-seqs 8 \
  --sampling-temp 0.1 \
  --seed 7
```

## A concrete nine-protein demo

The repository includes nine Protein Data Bank structures: five globins, two
lysozymes, and two unrelated controls. The demo runs both the structure/MPNN
and sequence/ESM-2 routes over four selected pairs and all 36 unique pairs:

```bash
./demo/run-demo.sh
```

The checked structure-route output separates related proteins from the
negative control without any external service:

| Pair | Relationship | Aligned residues | Mean coverage | RMSD | TM-score (query / target) | lDDT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1MBO / 1MYT | related myoglobins | 146 | 0.977 | 1.25 Å | 0.902 / 0.943 | 0.848 |
| 1MBO / 1MBA | divergent globins | 143 | 0.957 | 2.00 Å | 0.808 / 0.841 | 0.710 |
| 1REX / 2LYZ | human / hen lysozyme | 129 | 0.996 | 0.70 Å | 0.967 / 0.975 | 0.947 |
| 1MBO / 2LYZ | unrelated control | 109 | 0.773 | 15.22 Å | 0.092 / 0.095 | 0.217 |

The demo also writes per-pair aligned FASTA files and superposed PDBs under
`demo/results/`. See [the demo guide](demo/README.md) for the complete input
set, expected family blocks, output tables, and PDB attribution.

## Workload modes

| Mode | Use it when | Encoding behavior | Output order |
| --- | --- | --- | --- |
| `pairwise` | You need one comparison | Encodes the two inputs for that call | One result |
| `pair-list` | You know the exact named pairs | Encodes each unique protein once | Caller-supplied pair order |
| `all-vs-all` | You need every symmetric pair | Reuses inputs across `N(N-1)/2` comparisons | Stable lexicographic `(i, j)` order |

Examples:

```bash
# A chosen TSV list of sequence pairs
hikoboshi pair-list --pairs pairs.tsv --fasta proteins.fa \
  --package hikoboshi-esm2-8m --summary selected.tsv

# Every unique pair in a structure directory
hikoboshi all-vs-all structure structures/ \
  --package hikoboshi-mpnn-d64 --threads 8 \
  --summary all-vs-all.tsv --output-dir artifacts/
```

See the [pairwise](docs/algorithms/pairwise.md),
[pair-list](docs/algorithms/pair-list.md), and
[all-vs-all](docs/algorithms/all-vs-all.md) guides for input contracts and
output schemas.

## Python API

```python
import hikoboshi as hkbs

result = hkbs.pairwise.from_structure(
    "query.pdb",
    "target.pdb",
    package="hikoboshi-mpnn-d64",
    mode="hard",
)

print(result.metrics.raw_sw_score)
print(result.metrics.rmsd)
print(result.metrics.tm_score_query)
print(result.path.aligned_pairs)
```

Python exposes callable namespaces for `encode`, `pairwise`, `all_vs_all`,
`score_alignment`, and `inverse_fold`, plus explicit pair-list functions for
structure, sequence, coordinate, and embedding inputs. Native tensors support
the Python buffer protocol and DLPack; NumPy conversion is optional.

## C++ API

Public headers install under `<hikoboshi/...>`. The in-memory
`hikoboshi::api::Engine` accepts normalized structures, coordinates, tokenized
sequences, or residue embeddings and returns typed `Result<T>` values. File
parsing and artifact writing live in the adapters, so C++ callers can use the
engine without routing data through Python or the CLI.

The principal surfaces are:

- `<hikoboshi/api/engine.hpp>` for encode, pairwise, pair-list, all-vs-all, and
  inverse-fold requests;
- `<hikoboshi/api/requests.hpp>` and `<hikoboshi/api/results.hpp>` for public data
  contracts;
- `<hikoboshi/api/all_vs_all.hpp>` for streaming sinks; and
- `<hikoboshi/universal/package.hpp>` for compiled model-package resolution.

## Alignment and metrics

Hard mode runs local affine Smith–Waterman and returns the discrete alignment
path. Soft mode evaluates a differentiable partition-function alignment. The
`both` mode runs both and keeps the hard path as the primary alignment.
Soft and both modes are approximately 6–10× slower than hard mode at the same
problem size.

Hard and T=1 soft alignment use four independently calibrated gap pairs:

| Package | Mode | `gap_open` | `gap_extension` |
| --- | --- | ---: | ---: |
| `hikoboshi-mpnn-d64` | hard | `-1.40000` | `-0.150000` |
| `hikoboshi-mpnn-d64` | soft, T=1 | `-3.21337` | `-0.111704` |
| `hikoboshi-esm2-8m` | hard | `-1.01982` | `+0.225736` |
| `hikoboshi-esm2-8m` | soft, T=1 | `-6.72805` | `-0.0159468` |

The ESM2 hard extension is intentionally positive and comes from the recorded
near-zero-temperature anneal. See the [pairwise guide](docs/algorithms/pairwise.md)
and package provenance for the source values and hashes.

Depending on the input metadata, a result can include:

- raw Smith–Waterman score and aligned-pair count;
- query, target, and mean coverage;
- sequence identity;
- RMSD and query/target TM-score;
- lDDT and its directional/alignment variants;
- aligned FASTA; and
- a superposed target PDB for structure inputs.

Unavailable metrics are never fabricated: the CLI writes `NA`, while Python
uses `None` and retains the invalid-metric reason.

## Embedded model packages

| Package | Purpose | Public provenance |
| --- | --- | --- |
| `hikoboshi-mpnn-d64` | 64-dimensional structure embeddings | Checked tensor manifest and baked scalar payload |
| `hikoboshi-esm2-8m` | Sequence embeddings from the ESM-2 8M family | Checked safetensors payload, manifest, source digest, and transformation record |
| `proteinmpnn-v48-eps020` | ProteinMPNN inverse folding | Checked safetensors payload and upstream checkpoint provenance |

Large C++ byte arrays are reproducibly generated during the build only after
the checked artifacts pass their SHA-256 verification. Model, fixture, and
dataset terms are recorded separately in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); detailed manifests live in
`data/weights/` and `weights/`.

## Threading and reproducibility

All-vs-all and pair-list parallelism runs across independent pair workloads,
not inside one alignment. `--threads 0` selects automatic sizing, `1` forces
serial execution, and values above one request that many worker slots. Memory
planning may reduce the worker count rather than overcommit structure-encoder
workspaces.

Results retain stable pair order across thread counts. Strict and fast scalar
GEMM parity modes are available; strict mode targets the reference reduction
contract, while fast mode targets the documented numerical tolerance:

```bash
HIKOBOSHI_GEMM_PARITY_MODE=strict hikoboshi pairwise structure query.pdb target.pdb
```

For scheduler thresholds, workspace behavior, and benchmark methodology, see
[Performance](docs/algorithms/performance.md).

## Release scope

Version `0.1.0` is the portable scalar CPU release. CUDA, HIP, Metal, Vulkan,
OpenCL, and CPU SIMD backend names are reserved in the capability model but
are not buildable release backends yet. This first cut implements native
encoding, pairwise alignment, selected-pair and all-pairs workloads, scoring,
artifact generation, and inverse folding. Database search, MSA, clustering,
guide-tree construction, phylogenetics, and other suite surfaces are not in
`0.1.0` yet; they are the expansion path for the same runtime, not separate
projects.

## Contributing, license, and provenance

Admission is actor-neutral. Human-written and agent-written changes are subject
to the same provenance, clean-room, licensing, validation, and admission
requirements. Maintainer authority can authorize work, but it does not exempt a
change from those requirements.

External upstream contributions are temporarily closed until Megure Labs
deploys the production Kaname verifier. Forking and downstream modification
remain permitted under Apache-2.0. During the current pre-verifier bootstrap,
only explicitly authorized Megure-maintainer changes are admitted and no Kaname
history is claimed. Once the repository activates its forward-only enforcement
base, every later change will require a complete Kaname-compatible history.
Read [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md),
[the clean-room policy](CLEAN_ROOM.md), and
[the provenance policy](PROVENANCE_POLICY.md).

Hikoboshi is licensed under Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). Embedded model artifacts and PDB fixtures retain their
separate permissive terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The repository is a fresh release cut owned by Megure Labs and attributed to
Casey Mogilevsky. Development records for this release are retained privately
and are available on request. See [PROVENANCE.md](PROVENANCE.md).

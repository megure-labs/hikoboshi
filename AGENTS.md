# Agent instructions

These instructions apply to the entire repository.

## Project

Hikoboshi is one native suite for search, alignment, MSA, and the rest of
bioinformatics. Version 0.1.0 ships the suite's sequence/structure
encoding, pairwise and all-vs-all alignment, scoring, artifact-generation, and
inverse-folding foundation; database search, MSA, phylogenetics, clustering,
and other suite surfaces come later. Its command, Python package, and C++
namespace are all named `hikoboshi`; public Python examples conventionally use
`import hikoboshi as hiko`.

## Scope: actor-neutral

Provenance, clean-room, licensing, and admission requirements attach to a
contribution, not to the kind of actor that produced it. Human-written and
agent-written changes are admitted under identical requirements. After
enforcement, both require a complete Kaname-compatible trace and an entry under
`.provenance/changes/`. Being human is not an exemption; human authors and
agents may carry different legal responsibilities, but the repository
admission boundary is the same. During bootstrap, a maintainer-authorized
change without a trace must disclose its actual authority, validation,
authorship, and material assistance and must not fabricate a trace.

Imperatives below state repository-work and admission conditions for everyone
preparing a change, including maintainers and tools acting on their behalf.

## Contribution admission

External upstream contributions are temporarily closed while Megure Labs builds
and deploys the production Kaname verifier. Forking and downstream modification
remain permitted by Apache-2.0, but an external person or agent must not submit
code, tests, documentation, data, models, generated artifacts, or patches for
upstream admission during this closure.

These admission rules apply equally to human-written and agent-written changes
and to repository policy itself.

Until the repository's base commit contains
`.provenance/KANAME_ENFORCEMENT_BASE`, a Megure Labs maintainer may explicitly
authorize and execute an internal bootstrap change without a Kaname trace. Do
not create a `.provenance/changes/` record or claim Kaname provenance for such a
change. Preserve the actual authority, validation, and authorship disclosures in
the pull request instead. External contributions remain closed during bootstrap.
This bootstrap rule applies equally to human-written and agent-written
maintainer changes. Maintainer authority permits the bootstrap change; it does
not exempt the change from disclosure, clean-room, licensing, or validation
requirements.

Once that marker is present in the base commit, repository changes may be
proposed upstream only when a Megure Labs maintainer initiates and executes the
work through a Megure-controlled Kaname workflow and retains its complete
Kaname-compatible history. Git history, a final diff, a public provenance
record, a retrospective summary, or a contributor's self-attestation is not
that history. Do not replay an externally produced patch and describe the
replay as provenance for its creation.

After enforcement, if the complete history is absent or cannot be verified,
stop and report that the change is not eligible for review or merge.

The complete-history requirement applies equally to human-written and
agent-written changes and to repository policy itself. The exact bootstrap
boundary, history, and admission contract is defined in
`PROVENANCE_POLICY.md`.

## Clean-room implementation

`CLEAN_ROOM.md` is the normative clean-room procedure. The rules below are its
repository-wide admission boundary, not a substitute for its role separation,
access controls, evidence capture, and mechanical wall verification.

- Do not retrieve, copy, closely paraphrase, translate, or reconstruct source
  code from third-party implementations. This includes source surfaced by web
  searches, training-memory recall, package caches, unrelated local checkouts,
  decompilers, or other agents.
- Implement from public specifications, standards, papers, mathematical
  definitions, API documentation, test vectors, and independently observed
  behavior. General programming knowledge and documented APIs are allowed.
- You may freely edit, refactor, and reuse code already in this repository.
- You may use code supplied by a maintainer only when the maintainer identifies
  it as Megure-owned or explicitly approves its compatible license and source.
- Do not introduce GPL, AGPL, noncommercial, source-available, or unknown-license
  material. Permissively licensed code still requires maintainer approval and
  preservation of all required notices.
- If you have already inspected an external implementation relevant to the
  requested work, stop and disclose the source before implementing. The
  tainted attempt is not admissible. A maintainer must decide whether to begin
  a separate clean-room implementation under `CLEAN_ROOM.md`.

This policy governs importing external implementation material. It does not
prevent normal agent-assisted work on this repository or research using
papers, specifications, documentation, and test vectors.

## Models, data, licensing, and provenance

- New project code is contributed under Apache-2.0. Do not remove or alter
  `LICENSE`, `NOTICE`, third-party terms, attribution, or provenance records.
- Existing ESM-2, ProteinMPNN, and PDB artifacts are approved only as described
  in `THIRD_PARTY_NOTICES.md`. Do not add or replace code, weights, checkpoints,
  datasets, or fixtures without recording their source, checksum, license, and
  transformation history.
- After `.provenance/KANAME_ENFORCEMENT_BASE` is present in the base commit,
  every proposed change must have a complete Kaname-compatible trace and one
  new public record under `.provenance/changes/`, as specified by
  `PROVENANCE_POLICY.md`. Before that boundary, maintainer-authorized
  bootstrap changes must disclose that no Kaname trace exists and must not
  fabricate one. Never invent run ids, evidence, digests, review results, or
  closure state.
- Generated `*_blob.cpp` files for large checked model artifacts are build
  outputs. Change the checked artifact, metadata header, and generator together;
  do not hand-edit generated byte arrays.
- Never commit credentials, private Kaname traces, internal task graphs,
  unrelated machine paths, or private checkout metadata.
- Do not rewrite public history or force-push unless a maintainer explicitly
  asks for it.

## Validation

Run the narrow relevant tests while working. Before handing off a
release-facing change, run:

```bash
meson setup builddir -Dhikoboshi_python_api=false
meson compile -C builddir -j 8
meson test -C builddir --print-errorlogs

meson setup build-python -Dhikoboshi_python_api=true
meson compile -C build-python -j 8
meson test -C build-python --print-errorlogs
```

Also run `git diff --check`. See `docs/source-build.md` for Python installation
and build-tree usage. Public CI validates scalar CPU builds on Linux x86-64 and
current Apple Silicon (macOS 26). GPU execution is not performed on public
runners; any future GPU-affecting change must record applicable NVIDIA and/or
AMD validation from Megure-controlled hardware in its Kaname trace.

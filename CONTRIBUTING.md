# Contributing to Hikoboshi

The requirements in this file attach to each proposed change, not to whether
its author is human or an agent. Human contributors, maintainers, and coding
agents are subject to the same clean-room, licensing, provenance, validation,
and admission rules; no actor is exempt. Authority determines who may initiate
or approve work, but it does not waive requirements for the work itself.

## External contributions are temporarily closed

Megure Labs is not currently accepting external upstream contributions while
the production Kaname verifier is being built and deployed. Please do not open
a pull request or send a patch by issue, email, or another channel. This
closure applies to code, tests, documentation, dependencies, automation, data,
models, and generated artifacts, regardless of whether humans, coding agents,
or both produced them.

You may still fork and modify Hikoboshi under Apache-2.0. Security
vulnerabilities should continue to be reported privately as described in
`SECURITY.md`. A maintainer may independently act on an externally reported
idea, but the external patch may not be imported or retrospectively described
as having a complete history. After Kaname enforcement is activated, the
independent implementation must begin inside Megure's Kaname workflow.

External contributions will reopen only after the Kaname verifier is deployed
as a required, trusted GitHub check and the maintainers publish an updated
policy here.

## Maintainer development workflow during bootstrap

Bootstrap is active while the pull request's base commit does not contain
`.provenance/KANAME_ENFORCEMENT_BASE`.

Bootstrap is not an exemption for maintainers or human authors. It means only
that the enforcement trace does not yet exist: every authorized bootstrap
change must disclose its actual authority, validation, authorship, external
materials, and material human or agent assistance, and must not fabricate a
trace.

1. Obtain explicit authorization from a Megure Labs maintainer before editing.
2. Work on a focused branch and include tests for changed behavior.
3. Keep the `hikoboshi` CLI, Python package, and C++ namespace compatible unless
   the proposed change explicitly includes a migration.
4. Build and test the relevant paths locally.
5. Disclose the actual maintainer authority, validation, external materials,
   and material human or agent assistance in the pull request.
6. Do not add a `.provenance/changes/` record or claim Kaname history when no
   Kaname trace exists.

## Maintainer development workflow after enforcement

After enforcement, every change, whether human-written or agent-written, must
have a complete Kaname-compatible trace and an entry under
`.provenance/changes/`.

1. Initiate the change as an authorized Kaname goal, task, ticket, or packet
   before editing.
2. Work on a focused branch and include tests for changed behavior.
3. Keep the `hikoboshi` CLI, Python package, and C++ namespace compatible unless
   the proposed change explicitly includes a migration.
4. Build and test the relevant paths locally.
5. Seal the complete Kaname-compatible history and add its public change record at
   `.provenance/changes/<change-id>.json`.
6. Validate the record against the exact pull-request base with
   `python tools/validate_change_provenance.py check --base origin/main --head HEAD`.
7. Open a pull request using the repository template.

```bash
python -m pip install meson ninja
meson setup builddir -Dhikoboshi_python_api=false
meson compile -C builddir -j 8
meson test -C builddir --print-errorlogs

meson setup build-python -Dhikoboshi_python_api=true
meson compile -C build-python -j 8
meson test -C build-python --print-errorlogs
```

See `docs/source-build.md` for editable Python installs and optional NumPy
support.

Public CI builds and runs the scalar CPU implementation on Linux x86-64 and
current Apple Silicon (macOS 26). GPU execution is not currently performed on
public GitHub runners. Any future GPU-affecting change must include applicable
offline NVIDIA and/or AMD results in its retained Kaname trace and public
validation summary.

## Clean-room and AI-assisted work

The clean-room rule applies to the contribution regardless of whether a
maintainer writes it directly or uses coding agents.

Maintainer-controlled coding agents may implement, edit, test, and review work
during an explicitly authorized bootstrap change or, after enforcement, inside
an authorized Kaname run. All such work must follow `AGENTS.md` and the normative
three-role process in [`CLEAN_ROOM.md`](CLEAN_ROOM.md). In particular, do not
base a change on third-party implementation source. Public papers,
specifications, standards, API documentation, mathematical definitions, test
vectors, and black-box behavior are acceptable references.

In the pull request, disclose:

- which coding agents or AI tools materially contributed;
- every external paper, specification, API document, dataset, model, fixture,
  checkpoint, or code artifact consulted; and
- whether any external implementation source was viewed.

If external implementation source was viewed, stop the tainted attempt and
disclose the source before submitting code. A maintainer may authorize a fresh
implementation only through the isolated specification, implementation, and
audit roles defined in `CLEAN_ROOM.md`.

The complete history and approval contract is documented in
[`PROVENANCE_POLICY.md`](PROVENANCE_POLICY.md). A public change record is only
an index into that history. It cannot make an external or retrospectively
reconstructed change eligible to merge.

## Models and data

Do not add or replace a model, checkpoint, dataset, PDB/mmCIF fixture, or
generated artifact without maintainer approval. A contribution must retain the
original weights file in Megure-controlled artifact custody and include its
canonical source, license, cryptographic checksum, transformation process, and
any required attribution. Update `THIRD_PARTY_NOTICES.md` and the relevant
provenance manifest in the same pull request.

## License

When external contributions reopen, submitting one will represent that you have
the right to submit it and license it to recipients under the Apache License
2.0. Unless explicitly stated otherwise by the maintainers, the project's
inbound and outbound license is Apache-2.0. Preserve all copyright, license,
attribution, and provenance notices.

Do not submit GPL, AGPL, noncommercial, source-available, or unknown-license
material. Do not add permissively licensed third-party code or data without
maintainer approval and the notices required by its license.

## Reporting security issues

Do not open a public issue for a vulnerability. Follow `SECURITY.md`.

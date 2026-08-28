# Clean-room implementation policy

Applies to all contributions regardless of who authored them (human or agent).

This document defines Hikoboshi's normative clean-room process for work that
could otherwise derive protected expression from a third-party implementation.
It is based on Megure Labs' three-role clean-room workflow. `AGENTS.md` states
the repository-wide boundary; this file defines how a clean-room attempt must
actually be isolated, captured, and reviewed.

Clean-room treatment does not remove licensing, notice, patent, data-use, or
model-license obligations. A clean-room implementation is admissible only when
the underlying behavior may lawfully be implemented and every applicable
obligation is satisfied.

## When this process is required

An implementation attempt is tainted when a person or coding agent working on
it has viewed relevant third-party implementation source. Stop that attempt,
identify the material and access that caused the contamination, preserve the
real record, and notify a Megure Labs maintainer. Do not sanitize the attempt by
rewriting its history, paraphrasing the source, replaying its final diff, or
asking another agent to reproduce it from a retrospective summary.

Only a maintainer may authorize a clean-room restart. The authorization must
name the behavior being reimplemented, approved factual references, prohibited
source material, role assignments, workspace boundaries, required evidence,
and acceptance tests before the restarted implementation begins.

## Three isolated roles

### 1. Source-reading specification role

The specification role may read the explicitly approved external
implementation source and primary public materials. It may write only a neutral
behavioral specification, test vectors representing externally observable or
standardized facts, and its own run evidence. It must not read Hikoboshi
implementation source relevant to the target.

The specification records behavior, inputs, outputs, invariants, error cases,
mathematical definitions, standard encodings, and black-box test vectors. It
must not reproduce or closely paraphrase code, comments, identifiers, file or
module organization, function decomposition, control-flow structure,
implementation-specific data structures, or source-specific pseudocode.

### 2. Isolated implementation role

The implementation role receives the approved behavioral specification,
approved primary standards or API documentation, authorized Hikoboshi source,
and nothing from the contamination zone. It must not access:

- the external implementation source or any copy, cache, excerpt, or derivative
  of it;
- specification-role prompts, transcripts, tool results, or working notes;
- the tainted implementation attempt or a patch derived from it; or
- same-domain local source that the maintainer excluded from the task bundle.

The implementation role works in a distinct workspace with filesystem and tool
permissions that enforce those limits. It derives its own organization,
identifiers, decomposition, and expression from the behavioral specification
and approved factual materials.

### 3. Independent expression-audit role

The audit role receives the approved behavioral specification, mechanical wall
reports, and its own run evidence. It must not access the external source,
Hikoboshi implementation source, the tainted attempt, specification-role
transcripts, implementation-role transcripts, or another agent's run logs.

The audit role examines the specification and wall evidence for leaked
identifiers, source-shaped organization, copied comments or pseudocode,
unnecessary implementation detail, missing access records, and other signs
that protected expression crossed the wall. It reports findings to the human
orchestrator; it does not communicate them directly to the implementation role
or rewrite the candidate.

## Permitted bridge

The reviewed behavioral specification is the only bridge from the
source-reading role to the implementation role. Human orchestration may route a
factual ambiguity back through a new specification revision, but must not pass
source excerpts, source-shaped suggestions, hidden context, or audit-agent
wording into implementation. Every bridge revision receives a content digest
and is fixed before the corresponding implementation attempt starts.

## Evidence and access controls

Each role must use a separate identity, workspace, prompt, and allowlist. The
complete Kaname-compatible record must preserve at least:

- maintainer authority, scope, role assignments, and approved/prohibited
  materials;
- exact prompts, messages, model/provider/harness identities, tool calls and
  results, stdout, stderr, and lifecycle events;
- every path, process, network, and external-material access required by the
  active evidence profile;
- every created or modified file with size and cryptographic digest;
- immutable digests for specification revisions, test vectors, the candidate
  diff, validations, findings, and decisions; and
- clean termination, capture-completeness accounting, independent review,
  adjudication, and human approval where required.

An agent's statement that it respected the wall is not evidence. Git history, a
final diff, output similarity, or a public change record cannot replace the
captured access history.

## Mechanical wall verification

Before review, the verifier must reconstruct every role's read, write, execute,
and network accesses from the retained event record and fail closed if it finds
a forbidden access, missing event interval, unknown tool result, unbound output,
role/workspace overlap, or digest mismatch. It must also verify that:

- the specification role wrote only the authorized specification and evidence
  surfaces;
- the implementation role never reached the contamination zone or its logs;
- the audit role never reached either source tree or another role's logs;
- the implementation started from the approved specification digest; and
- the reviewed candidate is byte-identical to the candidate bound by the trace.

The human orchestrator reviews the full contamination-zone record, resolves
audit findings without crossing the wall, and records the admission decision.
If the wall cannot be proved intact, discard the candidate and begin a newly
authorized attempt; do not repair the evidence retrospectively.

## Bootstrap and post-enforcement admission

During the explicit pre-verifier bootstrap period, a maintainer-authorized
change may proceed without a Kaname trace, but no one may claim that an
uncaptured clean-room process occurred. A bootstrap change that depends on a
clean-room restart must still use real role separation and disclose the
available evidence and its limitations.

After `.provenance/KANAME_ENFORCEMENT_BASE` is present in the pull request's
base, a clean-room change is merge-eligible only when the complete three-role
history and a passing mechanical wall result are retained by Kaname and bound
to the public change record required by `PROVENANCE_POLICY.md`.

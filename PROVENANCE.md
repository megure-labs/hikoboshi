# Provenance

Applies to all contributions regardless of who authored them (human or agent).

This public version of Hikoboshi is a release cut from Megure Labs' private
development tree, prepared for public distribution. Its release Git history
intentionally begins with one commit. The private development commit history
and Kaname execution records are not included because they contain sensitive
infrastructure metadata and proprietary unreleased code. The public Kaname
provenance record format is being stabilized and will apply prospectively after
the repository's explicit enforcement boundary; complete private traces will
remain in Megure Labs' access-controlled store.

Where release work was captured by Kaname, Megure Labs retains complete,
immutable append-only provenance traces, including task graphs; agent, model,
and harness assignments; review and adjudication records; commands; tool calls;
execution metadata; and the source-to-release mapping.

The repository is currently in an explicitly authorized pre-verifier bootstrap
period. Bootstrap changes are not retrospectively relabeled as Kaname work and
do not receive fabricated trace records. The tracked enforcement marker and the
forward-only boundary are defined in
[Change provenance and merge policy](PROVENANCE_POLICY.md).

To request the full provenance traces, email Megure Labs at
`casey@megure.ai`. Access is reviewed case by case because traces may contain
security-sensitive infrastructure metadata and private development context.

Public model provenance is recorded in `data/weights/`, `weights/*.json`, and
`THIRD_PARTY_NOTICES.md`. The large embedded C++ arrays for checked model
artifacts are reproducibly generated at build time after verifying the source
artifact's SHA-256 digest.

The absence of the private traces from this repository does not alter the
Apache-2.0 license covering the released project source or the separate terms
covering third-party model and structure data. Release artifacts should be
verified against the signed or annotated Git tag and checksums published with
the applicable GitHub release.

## Provenance after enforcement

Every pull request based on a commit containing
`.provenance/KANAME_ENFORCEMENT_BASE` must carry a content-bound public change
record and a complete retained Kaname-compatible trace. The required trace
contents, clean-room treatment, automated checks, and human approval gates are
normative in the change provenance and merge policy.

The public record commits to the private trace with SHA-256 digests; it does
not publish transcripts or sensitive machine metadata. GitHub validates the
record's structure and exact patch binding. A code owner separately verifies
the committed digests against Kaname's access-controlled ledger before
approval.

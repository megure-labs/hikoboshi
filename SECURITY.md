# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected security or memory-safety
vulnerability. Use GitHub's private vulnerability-reporting flow on the
`megure-labs/hikoboshi` repository. Include the affected version, platform,
compiler, a minimal reproducer, and any sanitizer or crash output.

Do not include credentials, private data, proprietary structures, or private
provenance traces in a report. We will investigate privately and coordinate a
fixed release and disclosure when appropriate.

Any resulting security fix remains subject to the same provenance and
admission requirements whether authored by a human or an agent.

## Supported versions

Until a newer public release exists, `0.1.0` is the supported release line.

## Scope

Native memory safety, malformed input handling, artifact or model integrity,
unsafe path handling, dependency confusion, and unsafe binary loading are in
scope. Numerical model quality and execution of explicitly supplied untrusted
Python code are not security boundaries.

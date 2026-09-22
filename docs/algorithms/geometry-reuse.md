# Geometry preparation reuse

Pairwise result assembly computes one observed-C-alpha Kabsch superposition for
RMSD and both TM normalizations. Both TM sums share transformed distances while
retaining their separate scales and original per-step arithmetic/order. Standalone
RMSD and TM APIs retain their validity and normalization contracts.

lDDT reuses reference contact distances in a thread-local four-entry FIFO cache.
Every lookup compares the current observed-C-alpha mask and coordinate bits with
an owned snapshot and checks the contact radius. Pointers, input names, alignment
paths and externally owned lifetimes are not cache identities. Changes to inputs
at the same address invalidate reuse. Alignment maps and directional counters
are recomputed for every comparison; the six lDDT definitions do not change.

Each entry accepts at most 4,096 residues and 32,768 reference contacts. Capacity
requests per thread are at most 2 MiB of contacts plus 256 KiB of snapshots and
small metadata. Alignment-map scratch retains its existing independent behavior.
Larger structures, dense contact lists and nonfinite radii use the original direct
traversal. Cache allocation failure also falls back to direct computation. Entries
are thread-local and disappear with the thread, so reuse depends on scheduling;
this is neither a persistent cache nor a global per-protein distance matrix.

A first encounter records identity and reserves bounded capacity without writing
contact lists. A second encounter populates contacts while computing the requested
metric; later matching calls skip reference distance computation. This admission
rule avoids filling contact lists for one-use structures. Model distances
and threshold tests still run for aligned contacts. Single-use workloads may gain
little and pay bounded bookkeeping/allocation overhead; benchmark results must
separate them from repeated-protein workloads.

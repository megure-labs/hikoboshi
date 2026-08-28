# Performance Methodology

Hikoboshi `0.1.0` is validated as a scalar CPU release. The release evidence
compares current wall-clock timings with a rebuilt scalar archive binary and
keeps numerical correctness tied to the hard local affine Smith-Waterman
behavior in this release.

These measurements are intended to describe release validation status, not to
promise portable throughput. Actual runtime depends on compiler, CPU, input
size, storage, and Python environment.

## Benchmark Method

The release validation uses two benchmark sizes:

- Small deterministic fixtures exercise the public operation surface quickly:
  pairwise embeddings, pairwise structures, all-vs-all, Hikoboshi-MPNN-64
  forward, similarity, hard-SW recurrence, and traceback.
- Real-protein scale fixtures use a 129-residue pairwise case and a
  10-structure all-vs-all case. This size is large enough to expose the
  structure encoding and Hikoboshi-MPNN-64 scalar cost that small fixtures can
  hide.

For performance gates, lower ratios are better. A ratio of `1.0x` means current
Hikoboshi matched the rebuilt archive binary wall-clock time for that operation
on the validation host. Ratios above `1.0x` are slower than archive; ratios
below `1.0x` are faster.

The most recent release evidence was captured on a 12-logical-CPU Intel Xeon
Gold 6338 Linux host with GCC 13.3 and Meson 1.3.2. Native compiles were capped
at 8 jobs during validation.

## Locked Ratio Summary

| Operation | Small gate | Small ratio | Scale gate | Scale ratio | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `pairwise_embeddings` | 1.6x | 0.8356x | 3.0x | 0.1422x | faster than archive in scale evidence |
| `pairwise_structures` | 2.2x | 0.7111x | 5.0x | 3.2284x | within gate, but slower on real-protein scale |
| `all_vs_all` | 4.5x | 1.4434x | 6.0x | 2.8744x | within gate, but slower on real-protein scale |
| `mpnn_forward` | 2.2x | 0.6823x | 5.0x | 3.4694x | main remaining scalar scale cost |
| `similarity` | 13.0x | 2.6000x | 20.0x | 1.0633x | within gate |
| `hard_sw_recurrence` | 3.5x | 0.8974x | 5.0x | 0.0199x | faster than archive in scale evidence |
| `traceback` | 0.7x | 0.1356x | 1.0x | 0.0001x | faster than archive in scale evidence |

The scale results are intentionally candid: structure pairwise, structure
all-vs-all, and direct Hikoboshi-MPNN-64 forward are accepted for `0.1.0` under
the locked release gates, but they remain the clearest scalar optimization
targets for future releases.

## Thread Scaling Summary

All-vs-all threading is coarse worker scheduling over structure encoding and
pair records. It does not thread a single pairwise alignment. The public
selectors are CLI `--threads N` and Python `thread_count=N`.

The thread-scaling evidence measures one all-vs-all call outside fixture
construction and reports the median of 5 wall-clock samples.

| Mode | Inputs | 1 thread us | 2 threads us | 4 threads us | 8 threads us | 8-thread speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| embeddings | 100 | 225,171 | 115,849 | 71,062 | 44,881 | 5.017x |
| structures | 100 | 508,012 | 256,026 | 130,309 | 75,916 | 6.692x |

`--threads 0` and `thread_count=0` select automatic mode. `1` forces the serial
path. Values greater than `1` request worker slots, but Hikoboshi may reduce the
effective count or use the serial path for small jobs or memory-heavy structure
encoding. Output order remains stable and lexicographic at every thread count.

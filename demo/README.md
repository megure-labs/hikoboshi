# Hikoboshi demo

This self-contained fixture is intended to make Hikoboshi's behavior visually
obvious without downloading a large benchmark.

The nine structures contain:

- five globins: `1HBS`, `1MBA`, `1MBO`, `1MYT`, and `2NRL`;
- two lysozymes: `1REX` and `2LYZ`;
- two unrelated controls: immunoglobulin `1IGY` and RNase H `1RNH`.

The all-vs-all result should therefore contain recognizable globin and
lysozyme blocks, plus weak cross-family matches. There are nine inputs and 36
unique `i < j` pairs.

The structures come from the Protein Data Bank under CC0; see
[ATTRIBUTION.md](ATTRIBUTION.md).

`pairwise/pairs-structure.tsv` and `pairwise/pairs-sequence.tsv` contain four
illustrative comparisons:

1. `1MBO` vs `1MYT` — related myoglobins;
2. `1MBO` vs `1MBA` — a more divergent globin comparison;
3. `1REX` vs `2LYZ` — human and hen lysozyme;
4. `1MBO` vs `2LYZ` — unrelated negative control.

Run everything from the repository root:

```bash
./demo/run-demo.sh
```

The script writes summaries and alignment artifacts under `demo/results/`.
It runs MPNN from structures and ESM2 from the chain-A sequences extracted
from those same PDB files. The ESM2 command exercises the built-in annealed
hard-SW defaults, `gap_open=-1.01982` and `gap_extension=+0.225736`.

The binary is also directly usable as:

```bash
./builddir/hikoboshi --help
./builddir/hikoboshi pairwise --help
./builddir/hikoboshi pair-list --help
./builddir/hikoboshi all-vs-all --help
```

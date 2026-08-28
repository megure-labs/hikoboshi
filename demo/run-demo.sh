#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
binary="${HIKOBOSHI_BINARY:-$repo_root/builddir/hikoboshi}"
demo_root="demo"
results="$demo_root/results"
structures="$demo_root/all-vs-all/pdbs"

cd "$repo_root"

if [[ ! -x "$binary" ]]; then
  echo "missing Hikoboshi/Hikoboshi binary: $binary" >&2
  exit 1
fi

mkdir -p "$results"

"$binary" pair-list \
  --pairs "$demo_root/pairwise/pairs-structure.tsv" \
  "$structures" \
  --package hikoboshi-mpnn-d64 \
  --mode hard \
  --threads 4 \
  --summary "$results/pairwise-mpnn.tsv" \
  --output-dir "$results/pairwise-mpnn" \
  >/dev/null

"$binary" pair-list \
  --pairs "$demo_root/pairwise/pairs-sequence.tsv" \
  --fasta "$demo_root/all-vs-all/sequences.fasta" \
  --package hikoboshi-esm2-8m \
  --mode hard \
  --threads 4 \
  --parity-mode fast \
  --summary "$results/pairwise-esm2.tsv" \
  --output-dir "$results/pairwise-esm2" \
  >/dev/null

"$binary" all-vs-all structure "$structures" \
  --package hikoboshi-mpnn-d64 \
  --mode hard \
  --threads 8 \
  --summary "$results/all-vs-all-mpnn.tsv" \
  --output-dir "$results/all-vs-all-mpnn" \
  >/dev/null

"$binary" pair-list \
  --pairs "$demo_root/all-vs-all/all-pairs-sequence.tsv" \
  --fasta "$demo_root/all-vs-all/sequences.fasta" \
  --package hikoboshi-esm2-8m \
  --mode hard \
  --threads 8 \
  --parity-mode fast \
  --summary "$results/all-vs-all-esm2.tsv" \
  --output-dir "$results/all-vs-all-esm2" \
  >/dev/null

printf 'Hikoboshi demo complete:\n'
for summary in "$results"/*.tsv; do
  rows=$(($(wc -l < "$summary") - 1))
  printf '  %-28s %d pairs\n' "$(basename "$summary")" "$rows"
done

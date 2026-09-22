# Opt-in phase profiling

`tools/prepare_phase_profile.py` creates a separate diagnostic source tree from a
committed revision. It does not edit the production checkout or alter normal
builds. Python 3.12 or newer is required. Run from the repository:

```sh
python3 tools/prepare_phase_profile.py --output /tmp/hikoboshi-profile-source
meson setup /tmp/hikoboshi-profile-build /tmp/hikoboshi-profile-source \
  --buildtype=release -Ddebug=true -Dcpp_args=-fno-omit-frame-pointer \
  -Dhikoboshi_python_api=false -Dhikoboshi_cpu_target=portable
meson compile -C /tmp/hikoboshi-profile-build -j 8
HIKOBOSHI_PROFILE=phase /tmp/hikoboshi-profile-build/hikoboshi_cli \
  pair-list --pairs pairs.tsv structures --threads 16 > results.tsv 2> diagnostics.log
```

Choose a new output directory; the generator refuses to overwrite any existing
path or write inside the input repository. It archives the selected Git commit
(`--revision`, default `HEAD`), ignoring uncommitted changes. Function-count
assertions reject incompatible revisions before creating the output directory.
`PHASE_PROFILE.json` records the commit, generator/archive hashes, original file
hashes, and instrumented file hashes. Keep this with the exact build options,
binary hash, command, input hashes and machine controls.

The diagnostic CLI emits one `HIKO_PROFILE_JSON` line on stderr at exit, including
error exits. Stdout remains the ordinary command output. `HIKOBOSHI_PROFILE=phase`
enables coarse scopes; `detail` additionally enables nested encoder, pair and
geometry scopes. With the variable absent, the diagnostic build emits no timing
line. Other values disable timing. The production build has no inserted scopes.

## Accounting

The primary complete phase breakdown is for **structure pair-list**:

- `cli_total`: elapsed time inside main, excluding process startup and reporting.
- `load_phase`: the structure batch loader, including worker joins.
- `encode_phase`: complete serial/parallel encoding, including workspace/cache
  work inside that phase and worker joins.
- `pair_phase`: pair dispatch, including preparation, worker joins and synchronous
  result callbacks.
- `summary_stage`: pair-list callbacks, including artifact generation, formatting
  and temporary TSV writes. This is **inside** `pair_phase`.
- `summary_publish`: replay of the temporary TSV to stdout/summary, with summary
  flush. Kernel writeback and downstream consumption are not measured.

For a completed structure pair-list run, a disjoint presentation is load,
encode, `pair_phase - summary_stage`, summary stage, summary publish, and
`cli_total - load_phase - encode_phase - pair_phase - summary_publish` as the
remaining setup/teardown. Do not call the residual parsing or allocation time:
it also contains package resolution, validation and uninstrumented work.

Each detail record reports call count and **summed inclusive wall seconds**.
Concurrent calls and nested scopes overlap; these are not CPU seconds and cannot
be added to obtain elapsed time. For example, common linear-helper scopes are inside message/edge-update
scopes, which are inside layers, which are inside each protein's encoder scope.
They also cover edge embedding. These helpers include surrounding bias/mask work
and exclude the separate FFN projection helper; `encoder_ffn` measures the whole
FFN. Linear-helper values therefore are not total GEMM time.
`encoder_linear_row` may call `encoder_linear_bulk`; do not sum those categories.
The message/FFN/edge-update scopes cover all layers together. `geometry_lddt`
includes contact lookup/reuse. `geometry_superposition` covers shared RMSD/TM.

Other routes expose whichever scopes they use; absence is not a measured zero.
All-vs-all summary work is included in pair dispatch or uninstrumented collected
output, not pair-list `summary_stage`. ESM encoding exposes a coarse encode phase
but no MPNN detail. Direct encode and pairwise expose detail without a batch
encode phase. This tool does not provide GPU, AWS NUMA or memory-bandwidth data.

## Measurement discipline

Use matching builds, inputs, GEMM mode, geometry settings and CPU affinity. Run
sequentially in alternating order, recording frequency/governor/power profile and
host occupancy. Compare an ordinary build to diagnostic-off, phase and detail
runs; retain their actual overhead and output parity. Diagnostic timers can alter
inlining and incur clock/atomic overhead, especially at fine granularity. Use
ordinary builds for accepted end-to-end speed comparisons and diagnostic runs
for attribution. Reused embedding caches change the workload and must be labeled.

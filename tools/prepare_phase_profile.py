#!/usr/bin/env python3
"""Prepare an isolated, opt-in diagnostic source tree from a Git revision.

Production sources are never edited. Every insertion must match its expected
function count; unsupported revisions fail before creating the output tree.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

# name, reporting level (1=phase; 2=detail). Detail values are inclusive,
# summed elapsed time across calls/threads, never CPU time or additive phases.
LABELS = [
    ('cli_total', 1), ('load_phase', 1), ('encode_phase', 1), ('pair_phase', 1),
    ('summary_stage', 1), ('summary_publish', 1),
    ('load_file', 2), ('resolve_pairs', 2), ('cache_init', 2),
    ('encoder_protein', 2), ('encoder_knn', 2), ('encoder_rbf', 2),
    ('encoder_edge_embedding', 2), ('encoder_layers', 2),
    ('encoder_message', 2), ('encoder_ffn', 2), ('encoder_edge_update', 2),
    ('encoder_linear_bulk', 2), ('encoder_linear_row', 2),
    ('pair_compute', 2), ('similarity', 2), ('hard_sw', 2), ('traceback', 2),
    ('geometry_superposition', 2), ('geometry_lddt', 2),
]
HEADER = r'''#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
namespace hikoboshi_phase_profile {
using Clock = std::chrono::steady_clock;
enum Id { @IDS@, count };
inline constexpr const char* names[] = { @NAMES@ };
inline constexpr unsigned levels[] = { @LEVELS@ };
inline const unsigned level = [] {
  const char* value = std::getenv("HIKOBOSHI_PROFILE");
  return value && std::strcmp(value, "detail") == 0 ? 2U :
         value && std::strcmp(value, "phase") == 0 ? 1U : 0U;
}();
struct alignas(64) Counter { std::atomic<std::uint64_t> ns{0}, calls{0}; };
inline Counter totals[count];
struct Scope {
  Counter* counter;
  Clock::time_point start{};
  explicit Scope(Id id) noexcept : counter(level >= levels[id] ? &totals[id] : nullptr) {
    if (counter) start = Clock::now();
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() {
    if (!counter) return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
    counter->ns.fetch_add(elapsed, std::memory_order_relaxed);
    counter->calls.fetch_add(1, std::memory_order_relaxed);
  }
};
struct Report {
  ~Report() {
    if (!level) return;
    std::fprintf(stderr, "HIKO_PROFILE_JSON {\"schema\":1,\"level\":\"%s\",\"timers\":{", level == 2 ? "detail" : "phase");
    bool first = true;
    for (unsigned i = 0; i < count; ++i) {
      const auto calls = totals[i].calls.load();
      if (!calls) continue;
      std::fprintf(stderr, "%s\"%s\":{\"calls\":%llu,\"summed_wall_seconds\":%.9f}",
          first ? "" : ",", names[i], static_cast<unsigned long long>(calls), totals[i].ns.load() * 1e-9);
      first = false;
    }
    std::fprintf(stderr, "}}\n");
  }
};
}
#define HPP_JOIN_(a,b) a##b
#define HPP_JOIN(a,b) HPP_JOIN_(a,b)
#define HPP_SCOPE(id) hikoboshi_phase_profile::Scope HPP_JOIN(hpp_scope_,__LINE__)(hikoboshi_phase_profile::id)
'''


def prepare(repo, revision, output):
    repo, output = repo.resolve(), output.resolve()
    if output.exists() or output == repo or repo in output.parents:
        raise ValueError('output must be a new directory outside the source repository')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', revision + '^{commit}'], text=True).strip()
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', '--format=tar', commit])
    changed = {}
    before = {}

    def edit(path, functions):
        text = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + path]).decode()
        before[path] = hashlib.sha256(text.encode()).hexdigest()
        for name, label, expected in functions:
            pattern = r'(^[\w:<>,*& \t]*\b' + re.escape(name) + r'\([^;{}]*\)\s*(?:noexcept\s*)?(?:override\s*)?\{)'
            text, count = re.subn(pattern, lambda m: m[1] + '\n  HPP_SCOPE(' + label + ');', text, flags=re.M)
            if count != expected:
                raise ValueError(f'{path}: {name}: expected {expected} definitions, found {count}')
        changed[path] = '#include <hikoboshi/profile_diagnostic.hpp>\n' + text

    edit('cpp/cli/main.cpp', [('main', 'cli_total', 1)])
    changed['cpp/cli/main.cpp'] = changed['cpp/cli/main.cpp'].replace('  HPP_SCOPE(cli_total);', '  hikoboshi_phase_profile::Report hpp_report;\n  HPP_SCOPE(cli_total);')
    edit('cpp/io/structure_batch_loader.cpp', [('load_structures_from_files', 'load_phase', 1)])
    edit('cpp/io/structure_loader.cpp', [('load_structure_from_file', 'load_file', 1)])
    edit('cpp/api/engine.cpp', [('resolve_pair_list', 'resolve_pairs', 1)])
    edit('cpp/algorithms/all_vs_all/all_vs_all.cpp', [
        ('init', 'cache_init', 2), ('encode_structures_serial', 'encode_phase', 1),
        ('encode_structures_parallel', 'encode_phase', 1),
        ('encode_sequences_serial', 'encode_phase', 1), ('encode_sequences_parallel', 'encode_phase', 1),
        ('dispatch_pairs', 'pair_phase', 1)])
    edit('cpp/cli/commands/pair_list.cpp', [('receive', 'summary_stage', 1), ('finish', 'summary_publish', 1)])
    edit('cpp/modules/mpnn/mpnn_scalar.cpp', [
        ('mpnn64_forward_scalar_unchecked', 'encoder_protein', 1), ('build_knn', 'encoder_knn', 1),
        ('build_edge_rbf_features', 'encoder_rbf', 1), ('apply_edge_embedding', 'encoder_edge_embedding', 1),
        ('apply_mpnn_layers', 'encoder_layers', 1)])
    edit('cpp/include/hikoboshi/modules/mpnn/detail/mpnn_inner_inline.hpp', [
        ('linear_nt_inline', 'encoder_linear_bulk', 1), ('linear_row_nt_inline', 'encoder_linear_row', 1),
        ('apply_edge_update_inline', 'encoder_edge_update', 1)])
    edit('cpp/include/hikoboshi/modules/mpnn/detail/message_layer_inline.hpp', [('mpnn_message_layer_scalar_inline', 'encoder_message', 1)])
    edit('cpp/include/hikoboshi/modules/mpnn/detail/ffn_layer_inline.hpp', [('mpnn_ffn_layer_scalar_inline', 'encoder_ffn', 1)])
    edit('cpp/algorithms/pairwise/pairwise.cpp', [
        ('run_pairwise_embeddings', 'pair_compute', 1), ('build_similarity_score_matrix', 'similarity', 1),
        ('run_hard_sw', 'hard_sw', 1), ('build_traceback_path', 'traceback', 1)])
    edit('cpp/algorithms/metrics/tm_score.cpp', [('compute_superposition_metrics', 'geometry_superposition', 1)])
    edit('cpp/algorithms/metrics/lddt.cpp', [('compute_lddt', 'geometry_lddt', 1)])
    header = HEADER.replace('@IDS@', ', '.join(n for n, _ in LABELS)).replace('@NAMES@', ', '.join(json.dumps(n) for n, _ in LABELS)).replace('@LEVELS@', ', '.join(str(level) for _, level in LABELS))
    changed['cpp/include/hikoboshi/profile_diagnostic.hpp'] = header
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(output, filter='data')
    for path, text in changed.items():
        (output / path).write_text(text)
    manifest = dict(commit=commit, source=str(repo), archive_sha256=hashlib.sha256(archive).hexdigest(),
                    source_sha256=before, instrumented_sha256={p: hashlib.sha256(s.encode()).hexdigest() for p, s in changed.items()},
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    semantics='Inclusive elapsed scopes; detail sums overlap and span threads. Pair-list summary_stage is nested in pair_phase. cli_total excludes process startup and report emission.')
    (output / 'PHASE_PROFILE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.revision, args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Real-encoder cache reuse retains TSV, geometry, and structure artifacts."""
from __future__ import annotations
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def counts(result):
    match = re.search(r"embedding-cache: hits=(\d+) misses=(\d+) writes=(\d+) rejected=(\d+)", result.stderr)
    assert match, result.stderr
    return tuple(map(int, match.groups()))


def main():
    binary = Path(os.environ['HIKOBOSHI_BUILD_ROOT']) / 'hikoboshi'
    fixtures = Path(__file__).resolve().parents[1] / 'parity/proteinmpnn_v48_020/fixtures'
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        inputs = root / 'pdb'; inputs.mkdir()
        for name, source in zip(['a.pdb', 'b.pdb', 'c.pdb'], ['3htn_a_016.pdb', '5l33_a_018.pdb', '4gyt_a_018.pdb']):
            shutil.copyfile(fixtures / source, inputs / name)
        pairs = root / 'pairs.tsv'
        pairs.write_text('a.pdb\tb.pdb\nb.pdb\ta.pdb\na.pdb\ta.pdb\nb.pdb\tc.pdb\nc.pdb\ta.pdb\n')
        self_pairs = root / 'self.tsv'
        self_pairs.write_text('a.pdb\ta.pdb\nb.pdb\tb.pdb\nc.pdb\tc.pdb\n')
        cache = root / 'cache'
        env = dict(os.environ)
        env.pop('HIKOBOSHI_SKIP_GEOMETRY', None)
        def run(command, executable=binary):
            result = subprocess.run([str(executable), *map(str, command)], env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            return result
        for parity in ['fast', 'strict']:
            env['HIKOBOSHI_GEMM_PARITY_MODE'] = parity
            for mode in ['hard', 'soft', 'both']:
                summary = root / 'summary.tsv'; artifacts = root / 'artifacts'
                command = ['pair-list', '--pairs', pairs, inputs, '--threads', '4', '--mode', mode,
                           '--summary', summary]
                baseline = run(command)
                cold = run([*command, '--embedding-cache', cache])
                assert counts(cold) == ((0, 3, 3, 0) if mode == 'hard' else (3, 0, 0, 0))
                warm = run([*command, '--embedding-cache', cache])
                assert counts(warm) == (3, 0, 0, 0)
                assert baseline.stdout == cold.stdout == warm.stdout == summary.read_text()
                if mode != 'soft':
                    artifact_command = ['pair-list', '--pairs', self_pairs, inputs, '--threads', '4',
                                        '--mode', mode, '--output-dir', artifacts]
                    expected = run(artifact_command)
                    expected_artifacts = {str(p.relative_to(artifacts)): p.read_bytes() for p in artifacts.rglob('*') if p.is_file()}
                    assert len(expected_artifacts) >= 6
                    actual = run([*artifact_command, '--embedding-cache', cache])
                    assert actual.stdout == expected.stdout and counts(actual) == (3, 0, 0, 0)
                    assert expected_artifacts == {str(p.relative_to(artifacts)): p.read_bytes() for p in artifacts.rglob('*') if p.is_file()}
                for route in ['structure', 'coords']:
                    all_command = ['all-vs-all', route, inputs, '--threads', '4', '--mode', mode]
                    all_baseline = run(all_command)
                    all_warm = run([*all_command, '--embedding-cache', cache])
                    assert all_baseline.stdout == all_warm.stdout
                    assert counts(all_warm) == (3, 0, 0, 0)
        assert len(list(cache.glob('*'))) == 2, 'GEMM modes must have separate identities'
        # Separate processes race on the same cold cache; readers only see complete entries.
        concurrent_cache = root / 'concurrent'
        args = [str(binary), 'pair-list', '--pairs', str(pairs), str(inputs), '--threads', '4', '--embedding-cache', str(concurrent_cache)]
        first = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        second = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out_a, err_a = first.communicate(); out_b, err_b = second.communicate()
        assert first.returncode == second.returncode == 0, err_a + err_b
        assert out_a == out_b
        warm = run(args[1:]); assert counts(warm) == (3, 0, 0, 0) and warm.stdout == out_a
        entry = next(concurrent_cache.rglob('*.hke'))
        payload = bytearray(entry.read_bytes()); payload[-1] ^= 1; entry.write_bytes(payload)
        recovered = run(args[1:])
        assert counts(recovered) == (2, 1, 1, 1) and recovered.stdout == out_a
        # Byte-identical executable copies reuse. Changed executable bytes invalidate
        # even if ELF execution is otherwise unchanged. Linux-only trailer probe.
        if os.uname().sysname == 'Linux':
            copied = root / 'binary-copy'; shutil.copy2(binary, copied)
            assert counts(run(args[1:], copied)) == (3, 0, 0, 0)
            with copied.open('ab') as output: output.write(b'cache-test-identity-change')
            modified = run(args[1:], copied)
            assert counts(modified) == (0, 3, 3, 0) and modified.stdout == out_a
        for route in ['sequence', 'embeddings']:
            invalid_route = subprocess.run([str(binary), 'all-vs-all', route, str(inputs), '--include-self', '--embedding-cache', str(cache)], env=env, capture_output=True, text=True)
            assert invalid_route.returncode != 0 and 'requires structure or coords' in invalid_route.stderr
        invalid = root / 'not-a-directory'; invalid.write_text('ordinary file')
        rejected = subprocess.run([str(binary), 'pair-list', '--pairs', str(pairs), str(inputs), '--embedding-cache', str(invalid)], env=env, capture_output=True, text=True)
        assert rejected.returncode != 0 and not rejected.stdout
    print('Real cache cold/warm, geometry, artifacts, modes, concurrency, corruption and executable identity passed')


if __name__ == '__main__':
    main()

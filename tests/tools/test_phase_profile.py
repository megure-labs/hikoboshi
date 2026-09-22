#!/usr/bin/env python3
"""Standalone checks for isolated profile preparation and timer accounting."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('phase_profile', ROOT / 'tools/prepare_phase_profile.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProfileTests(unittest.TestCase):
    def test_output_guards_do_not_write(self):
        with tempfile.TemporaryDirectory() as name:
            target = Path(name)
            marker = target / 'marker'
            marker.write_text('preserve')
            with self.assertRaises(ValueError):
                module.prepare(ROOT, 'HEAD', target)
            self.assertEqual(marker.read_text(), 'preserve')
            self.assertEqual(list(target.iterdir()), [marker])
        with self.assertRaises(ValueError):
            module.prepare(ROOT, 'HEAD', ROOT / 'uncreated-profile-output')
        self.assertFalse((ROOT / 'uncreated-profile-output').exists())

    def test_nested_parallel_scopes_and_opt_in(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            header = module.HEADER.replace('@IDS@', ', '.join(n for n, _ in module.LABELS)).replace('@NAMES@', ', '.join(json.dumps(n) for n, _ in module.LABELS)).replace('@LEVELS@', ', '.join(str(level) for _, level in module.LABELS))
            (root / 'profile.hpp').write_text(header)
            (root / 'probe.cpp').write_text(r'''
#include "profile.hpp"
#include <thread>
#include <barrier>
#include <iostream>
int main() {
  hikoboshi_phase_profile::Report report;
  HPP_SCOPE(cli_total);
  {
    HPP_SCOPE(encode_phase);
    std::barrier ready(2);
    auto work = [&] {
      HPP_SCOPE(encoder_protein);
      ready.arrive_and_wait();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    };
    std::thread first(work), second(work);
    first.join(); second.join();
  }
  std::cout << "unchanged\n";
}
''')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++20', '-O2', '-pthread', str(root / 'probe.cpp'), '-o', str(root / 'probe')], check=True, capture_output=True)
            for value in [None, 'disabled', 'phase', 'detail']:
                env = dict(os.environ)
                env.pop('HIKOBOSHI_PROFILE', None)
                if value is not None:
                    env['HIKOBOSHI_PROFILE'] = value
                result = subprocess.run([str(root / 'probe')], env=env, check=True, capture_output=True, text=True)
                self.assertEqual(result.stdout, 'unchanged\n')
                if value not in ['phase', 'detail']:
                    self.assertEqual(result.stderr, '')
                    continue
                self.assertEqual(len(result.stderr.splitlines()), 1)
                prefix, payload = result.stderr.strip().split(' ', 1)
                self.assertEqual(prefix, 'HIKO_PROFILE_JSON')
                data = json.loads(payload)
                self.assertEqual(data['level'], value)
                timers = data['timers']
                self.assertEqual(timers['cli_total']['calls'], 1)
                self.assertEqual(timers['encode_phase']['calls'], 1)
                self.assertGreaterEqual(timers['cli_total']['summed_wall_seconds'], timers['encode_phase']['summed_wall_seconds'])
                if value == 'phase':
                    self.assertNotIn('encoder_protein', timers)
                else:
                    self.assertEqual(timers['encoder_protein']['calls'], 2)
                    self.assertGreaterEqual(timers['encoder_protein']['summed_wall_seconds'], .04)


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ResolverTests(unittest.TestCase):
    def run_case(self, association='CONTRIBUTOR', permission='admin', *, api_fail=False, mismatch=False, enforcement=False, malformed=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            target = root/'.github/scripts/resolve-pr-author-association.sh'
            target.parent.mkdir(parents=True)
            shutil.copy2(ROOT/'.github/scripts/resolve-pr-author-association.sh', target)
            if enforcement:
                (root/'.provenance').mkdir()
                (root/'.provenance/KANAME_ENFORCEMENT_BASE').touch()
            mock = root/'gh'
            mock.write_text('#!/bin/sh\ncase "$*" in\n */collaborators/*) printf "%s\\n" "$ACCESS"; exit "$API_EXIT";;\n *) printf "%s\\n" "$IDENTITY";;\nesac\n')
            mock.chmod(0o755)
            identity = association + '\tcaseysm\t141592509'
            access = permission + '\tcaseysm\t' + ('123' if mismatch else '141592509')
            if malformed:
                access += '\nadmin'
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'], IDENTITY=identity, ACCESS=access, API_EXIT='1' if api_fail else '0')
            return subprocess.run([str(target),'megure-labs/hikoboshi','1'], env=env, capture_output=True, text=True)
    def test_existing_membership(self):
        for association in ['OWNER','MEMBER']:
            result=self.run_case(association, api_fail=True)
            self.assertEqual(result.returncode,0)
            self.assertEqual(result.stdout.strip(),association)
    def test_verified_admin(self):
        result=self.run_case()
        self.assertEqual(result.returncode,0)
        self.assertEqual(result.stdout.strip(),'BOOTSTRAP_REPOSITORY_ADMIN')
    def test_lower_permissions_not_admitted(self):
        for permission in ['write','maintain','read','triage','none']:
            result=self.run_case(permission=permission)
            self.assertEqual(result.returncode,0)
            self.assertEqual(result.stdout.strip(),'CONTRIBUTOR')
    def test_api_failure_closed(self):
        self.assertNotEqual(self.run_case(api_fail=True).returncode,0)
    def test_identity_mismatch_closed(self):
        self.assertNotEqual(self.run_case(mismatch=True).returncode,0)
    def test_malformed_response_closed(self):
        self.assertNotEqual(self.run_case(malformed=True).returncode,0)
    def test_enforcement_disables_fallback(self):
        result=self.run_case(enforcement=True,api_fail=True)
        self.assertEqual(result.returncode,0)
        self.assertEqual(result.stdout.strip(),'CONTRIBUTOR')

if __name__ == '__main__':
    unittest.main()

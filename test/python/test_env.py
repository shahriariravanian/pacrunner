"""pacrunner._env: discovery and configuration, without MadSpace.

    python -m unittest discover -s test/python      (or pytest test/python)

The MadSpace tests need a MadGraph7 checkout (PACRUNNER_MADGRAPH7); they are skipped without one."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pacrunner
from pacrunner import _env


class Repository(unittest.TestCase):
    def test_the_package_has_the_c_sources(self):
        c = _env.c_sources()
        self.assertEqual(c, Path(pacrunner.__file__).resolve().parent / "c")
        self.assertTrue((c / "umami_pac.c").is_file())
        self.assertTrue((c / "pacops.c").is_file())

    def test_repository_is_the_clone(self):
        # these tests run from the clone (installed editable)
        root = pacrunner.repository()
        self.assertEqual(root, Path(pacrunner.__file__).resolve().parent.parent)
        self.assertTrue((root / "pyproject.toml").is_file())

    def test_an_installed_copy_has_no_repository(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(_env, "REPOSITORY", Path(tmp)):
            self.assertIsNone(pacrunner.repository())

    def test_source_revision(self):
        from pacrunner import _artifacts
        digest = _artifacts._sources_digest()
        self.assertTrue(digest.startswith(pacrunner.__version__ + ":"), digest)
        self.assertEqual(digest, _artifacts._sources_digest())
        with mock.patch.object(_env, "repository", lambda: None):
            self.assertEqual(_artifacts.source_revision(), digest)


class NoJulia(unittest.TestCase):
    def test_configure_julia_is_ignored_with_a_warning(self):
        with self.assertWarns(DeprecationWarning):
            pacrunner.configure(julia="julia")
        self.assertFalse(hasattr(_env.CONFIG, "julia"))


class MadGraph7(unittest.TestCase):
    def test_not_a_checkout(self):
        with mock.patch.object(_env.CONFIG, "madgraph7", tempfile.mkdtemp()):
            with self.assertRaises(pacrunner.PacRunnerError):
                _env.madgraph7()

    def test_none_configured(self):
        env = {k: v for k, v in os.environ.items() if k != "PACRUNNER_MADGRAPH7"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(_env.CONFIG, "madgraph7", None):
            self.assertIsNone(_env.madgraph7(required=False))
            with self.assertRaises(pacrunner.PacRunnerError):
                _env.madgraph7()

    def test_installed_madspace_without_madgraph7(self):
        # no MadGraph7 configured: the madspace installed in this python, or an error naming both ways
        env = {k: v for k, v in os.environ.items() if k != "PACRUNNER_MADGRAPH7"}
        installed = "from pacrunner import _env; print(_env.import_madspace().__file__)"
        r = subprocess.run([sys.executable, "-c", installed], capture_output=True, text=True, env=env)
        if "no MadSpace" not in r.stderr:   # this python has a madspace of its own
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("site-packages", r.stdout)
        hidden = ("import sys; sys.path = [p for p in sys.path if 'site-packages' not in p]; "
                  f"sys.path.insert(0, {os.fspath(_env.PACKAGE.parent)!r}); from pacrunner import _env\n"
                  "try:\n    _env.import_madspace()\nexcept _env.PacRunnerError as e:\n    print('error:', e)")
        r = subprocess.run([sys.executable, "-c", hidden], capture_output=True, text=True, env=env)
        self.assertIn("error: no MadSpace", r.stdout, r.stderr)
        self.assertIn("PACRUNNER_MADGRAPH7", r.stdout)

    def test_madspace_build_choice(self):
        mg7 = Path(tempfile.mkdtemp())
        (mg7 / "bin").mkdir()
        (mg7 / "bin" / "madgraph").write_text("")
        (mg7 / "madspace" / "install" / "madspace").mkdir(parents=True)
        with mock.patch.object(_env.CONFIG, "cuda", None):
            self.assertEqual(_env.madspace_dir(mg7), mg7 / "madspace" / "install")   # no CUDA build: the CPU one
            (mg7 / "madspace" / "install-cuda" / "madspace").mkdir(parents=True)
            self.assertEqual(_env.madspace_dir(mg7), mg7 / "madspace" / "install-cuda")   # the CUDA build by default
        with mock.patch.object(_env.CONFIG, "cuda", False):
            self.assertEqual(_env.madspace_dir(mg7), mg7 / "madspace" / "install")


class Imports(unittest.TestCase):
    def test_madgraph_run_needs_no_numpy(self):
        # `python -m pacrunner._madgraph` runs in MadGraph7's python, which may have no numpy
        code = "import sys; sys.modules['numpy'] = None\nimport pacrunner, pacrunner._madgraph\nprint('ok')"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "ok", r.stderr)


MG7 = os.environ.get("PACRUNNER_MADGRAPH7")


@unittest.skipUnless(MG7, "PACRUNNER_MADGRAPH7 not set")
class MadSpace(unittest.TestCase):
    def run_python(self, code):
        # a fresh process: MadSpace is imported once per process
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    def test_import_and_late_configure(self):
        r = self.run_python(
            "import pacrunner; from pacrunner import _env\n"
            "ms = _env.import_madspace()\n"
            "print(ms.cpu_device() is not None)\n"
            "try:\n    pacrunner.configure(cuda=False)\nexcept pacrunner.PacRunnerError as e:\n    print('refused')\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["True", "refused"])


if __name__ == "__main__":
    unittest.main()

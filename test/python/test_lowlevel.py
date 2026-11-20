"""pacrunner.lowlevel: the low-level API's names, in Python (PacRunner.jl's exports until 0.3).

    python -m unittest discover -s test/python      (or pytest test/python)

The compile test needs symjit >= 2.29 (the kernels' MIR)."""

import os
import tempfile
import unittest

import pacrunner
from pacrunner import lowlevel as pr

ROOT = pacrunner.repository()
EXPORT = ROOT / "test" / "data" / "export" / "zggg_union.json"

try:
    import symjit
    HAVE_MIR_JSON = hasattr(symjit, "mir_json")
except ImportError:
    HAVE_MIR_JSON = False


class LowLevel(unittest.TestCase):
    def test_every_name_is_there(self):
        for name in pr.EXPORTS:
            self.assertTrue(hasattr(pr, name), name)
        self.assertEqual(set(pr.__all__) - {"EXPORTS"}, set(pr.EXPORTS))

    def test_julia_only_names_are_a_clear_error(self):
        for name in ("evaluate", "LArena", "read_sjb", "complexify", "jl"):
            with self.assertRaises(AttributeError) as e:
                getattr(pr, name)
            self.assertIn("not ported", str(e.exception))

    def test_programs_and_references(self):
        programs = pr.load_recurrence_export(EXPORT)
        refs = pr.recurrence_references(EXPORT)
        self.assertEqual(len(programs), len(refs))
        self.assertIsInstance(programs[0], pr.Lowered)
        e = pr.encode(pr.optimize(programs[0]))
        self.assertEqual(pr.decode(e, pr.optimize(programs[0])).ops, pr.optimize(programs[0]).ops)

    @unittest.skipUnless(HAVE_MIR_JSON, "no symjit >= 2.29")
    def test_compile_writes_the_folder(self):
        program = pr.load_recurrence_export(EXPORT)[0]
        with tempfile.TemporaryDirectory() as out:
            pr.compile(out, pr.optimize(program), interpreter=True)
            for f in ("runner.c", "runner.h", "uirvm.c", "Makefile"):
                self.assertTrue(os.path.isfile(os.path.join(out, f)), f)


if __name__ == "__main__":
    unittest.main()

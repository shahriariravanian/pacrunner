"""pacrunner.codegen against the goldens: every compiled folder of the golden sets (goldens.py; the
fixture pp_3j's are committed) written again, runner.c, runner.h, the Makefile and testing.c byte for
byte pac-julia's, in its four variants (imported or optimized, straight-line or interpreter); and the
fixture's folders built and run (`make test`: the outputs against RustiCol's reference values).

Skipped without symjit >= 2.29 (the kernels' MIR); the builds also without cc and make.

    python -m unittest discover -s test/python -p test_codegen.py
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from goldens import FIXTURE_EXPORTS, golden_sets

from pacrunner import codegen, uir

try:
    import symjit
    HAVE_MIR_JSON = hasattr(symjit, "mir_json")
except ImportError:
    HAVE_MIR_JSON = False
HAVE_MAKE = bool(shutil.which("cc") and shutil.which("make"))

VARIANTS = ("imported-straight", "imported-interpreter", "optimized-straight", "optimized-interpreter")
GENERATED = ("runner.c", "runner.h", "Makefile", "testing.c")


def variant_program(l, variant):
    return (uir.optimize(l) if variant.startswith("optimized") else l), variant.endswith("interpreter")


@unittest.skipUnless(HAVE_MIR_JSON, "no symjit >= 2.29 (symjit.mir_json)")
class Goldens(unittest.TestCase):
    def test_folders(self):
        compared = 0
        for (name, folder, export, _) in golden_sets():
            if not (folder / "compile").is_dir():
                continue
            ls, refs = uir.load_recurrence_export(export), uir.recurrence_references(export)
            for entry in sorted((folder / "compile").iterdir()):
                k = int(entry.name)
                examples = codegen.Examples.of(ls[k], refs[k])
                for variant in VARIANTS:
                    golden = entry / variant
                    (prog, interpreter) = variant_program(ls[k], variant)
                    with self.subTest(export=name, program=k, variant=variant), tempfile.TemporaryDirectory() as out:
                        if (golden / "ERROR").exists():
                            with self.assertRaises(Exception):
                                codegen.compile(out, prog, examples=examples, interpreter=interpreter)
                            continue
                        codegen.compile(out, prog, examples=examples, interpreter=interpreter)
                        for f in GENERATED:
                            self.assertEqual((golden / f).exists(), (Path(out) / f).exists(), f)
                            if (golden / f).exists():
                                self.assertEqual((Path(out) / f).read_bytes(), (golden / f).read_bytes(), f"{f} differs")
                        compared += 1
        self.assertGreater(compared, 0)


@unittest.skipUnless(HAVE_MIR_JSON and HAVE_MAKE, "no symjit >= 2.29, or no cc and make")
class MakeTest(unittest.TestCase):
    def test_make_test(self):
        export = FIXTURE_EXPORTS / "pp_3j.json"
        ls, refs = uir.load_recurrence_export(export), uir.recurrence_references(export)
        picked = [k for k, r in enumerate(refs) if r is not None]
        picked = sorted({picked[0], picked[-1]} | {k for k, l in enumerate(ls) if l.stubs and refs[k] is not None})
        with tempfile.TemporaryDirectory() as work:
            for k in picked:
                for variant in VARIANTS:
                    (prog, interpreter) = variant_program(ls[k], variant)
                    out = Path(work) / f"{k}-{variant}"
                    codegen.compile(out, prog, examples=codegen.Examples.of(ls[k], refs[k]), interpreter=interpreter)
                    r = subprocess.run(["make", "-s", "test"], cwd=out, capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, f"program {k} {variant}:\n{r.stdout}{r.stderr}")
                    self.assertIn("worst 0)", r.stdout, f"program {k} {variant}: not bit for bit RustiCol's")


class Units(unittest.TestCase):
    def test_describe(self):
        l = uir.load_recurrence_export(FIXTURE_EXPORTS / "pp_3j.json")[0]
        texts = [uir.describe(l, op) for op in l.ops]
        self.assertTrue(any(t.startswith("SOURCE c") for t in texts))
        self.assertTrue(any(t.startswith("REDUCE out[0] += w1 × |a") for t in texts))

    def test_lit(self):
        self.assertEqual(codegen.lit(1e-5), "1.0e-5")
        with self.assertRaises(codegen.CompileError):
            codegen.lit(float("inf"))

    def test_refuses_examples_of_another_program(self):
        ls = uir.load_recurrence_export(FIXTURE_EXPORTS / "pp_3j.json")
        refs = uir.recurrence_references(FIXTURE_EXPORTS / "pp_3j.json")
        k = next(k for k, r in enumerate(refs) if r is not None)
        # (the examples are checked before the kernels are read: no symjit needed)
        other = next(j for j, l in enumerate(ls) if (l.process, l.colour) != (ls[k].process, ls[k].colour))
        with tempfile.TemporaryDirectory() as out, self.assertRaises(codegen.CompileError):
            codegen.compile(out, ls[other], examples=codegen.Examples.of(ls[k], refs[k]))


if __name__ == "__main__":
    unittest.main()

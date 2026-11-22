"""The Metal runner (c/metal/; pac-julia's compile_metal.jl, kernel_msl.jl), on any system:

- compile_metal's files (runner.metal, program.bin, runner_metal.c, runner_metal.h, the Makefile) of
  every compile_metal golden (goldens.py: the fixtures' committed, without runner_metal.c; the corpus)
  byte for byte pac-julia's, imported and optimized;
- the Metal sources on the CPU, through c/metal/msl_shim.h (C++, f32 denormals flushed as Apple GPUs
  do; pac-julia's test_metal_shim.jl): the df64 operations against f64, and whole programs
  (runner.metal, test/metal_shim_run.cpp) at their export's reference points against RustiCol's
  values: the matrix element (the sum of the helicity outputs) within mixed precision's targets
  (c/metal/PLAN.md: median about 2e-7, worst 1e-4), each output relative to it. The programs: the
  fixtures', and of the corpus the compile_metal sample (all of them with PACRUNNER_TEST_METAL_ALL=1).

Skipped without symjit >= 2.29 (the kernels' MIR); the shim also without a C++ compiler (CXX, default
c++).

    python -m unittest discover -s test/python -p test_metal.py
"""

import ctypes
import importlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from goldens import ROOT, golden_sets

from pacrunner import codegen, kernels, uir
cm = importlib.import_module("pacrunner.codegen.compile_metal")   # the module (codegen.compile_metal is the function)

HAVE_MIR_JSON = kernels.symjit_problem() is None   # symjit >= 2.29 (its native library's version)
CXX = os.environ.get("CXX", "c++")
HAVE_CXX = shutil.which(CXX.split()[0]) is not None
CXXFLAGS = ["-std=c++17", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-Wno-attributes", "-Wno-unused-function",
            "-Wno-unused-variable"]


def build_shim(source: Path, out: Path, includes) -> ctypes.CDLL:
    subprocess.run([*CXX.split(), *CXXFLAGS, *[f"-I{i}" for i in includes], "-o", str(out), str(source)], check=True)
    return ctypes.CDLL(str(out))


@unittest.skipUnless(HAVE_MIR_JSON, "no symjit >= 2.29 (symjit.mir_json)")
class Goldens(unittest.TestCase):
    def test_folders(self):
        compared = 0
        for (name, folder, export, _) in golden_sets():
            if not (folder / "compile_metal").is_dir():
                continue
            ls = uir.load_recurrence_export(export)
            for entry in sorted((folder / "compile_metal").iterdir()):
                k = int(entry.name)
                for variant in ("imported", "optimized"):
                    golden = entry / variant
                    prog = uir.optimize(ls[k]) if variant == "optimized" else ls[k]
                    with self.subTest(export=name, program=k, variant=variant), tempfile.TemporaryDirectory() as out:
                        if (golden / "ERROR").exists():
                            with self.assertRaises(Exception):
                                codegen.compile_metal(out, prog)
                            continue
                        codegen.compile_metal(out, prog)
                        for f in cm.GENERATED:
                            if (golden / f).exists():
                                self.assertEqual((Path(out) / f).read_bytes(), (golden / f).read_bytes(), f"{f} differs")
                                compared += 1
                        for f in cm.METAL_HOST:
                            self.assertEqual((Path(out) / f).read_bytes(), (cm.METAL_DIR / f).read_bytes(), f)
        self.assertGreater(compared, 0)


@unittest.skipUnless(HAVE_MIR_JSON and HAVE_CXX, "no symjit >= 2.29 or no C++ compiler")
class Shim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.ops = build_shim(ROOT / "test" / "metal_shim_ops.cpp", Path(cls.work.name) / "libshim_ops.so", [cm.METAL_DIR])
        cls.ops.shim_dd.restype = ctypes.c_double
        cls.ops.shim_dd.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]
        cls.ops.shim_flush_denormals()

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_df64(self):
        """+, −, ×, ÷, sqrt within 2^-44 of f64 (the inputs' own df64 rounding is 2^-48); <, max, floor,
        rint exact"""
        dd = self.ops.shim_dd
        rng = np.random.default_rng(20261115)
        worst = {}
        def note(k, e):
            worst[k] = max(worst.get(k, 0.0), e)
        for _ in range(20000):
            x = float(np.exp(rng.standard_normal() * 10) * rng.choice((-1, 1)))
            y = float(np.exp(rng.standard_normal() * 10) * rng.choice((-1, 1)))
            note("+", abs(dd(0, x, y) - (x + y)) / max(abs(x), abs(y)))
            note("-", abs(dd(1, x, y) - (x - y)) / max(abs(x), abs(y)))
            note("*", abs(dd(2, x, y) - x * y) / abs(x * y))
            note("/", abs(dd(3, x, y) - x / y) / abs(x / y))
            note("sqrt", abs(dd(4, abs(x), 0.0) - abs(x) ** 0.5) / abs(x) ** 0.5)
            note("<", 0.0 if dd(5, x, y) == (1.0 if x < y else 0.0) else 1.0)
            rx, ry = dd(0, x, 0.0), dd(0, y, 0.0)          # the inputs as df64 holds them
            note("max", 0.0 if dd(8, x, y) == max(rx, ry) else 1.0)
        for x in (0.0, -0.0, 1.0, 2.5, -2.5, 3.5, 1e7 + 0.5, -7.25):
            note("floor", 0.0 if dd(6, x, 0.0) == np.floor(x) else 1.0)
            note("rint", 0.0 if dd(7, x, 0.0) == np.rint(x) else 1.0)
        for k in ("+", "-", "*", "/", "sqrt"):
            self.assertLess(worst[k], 2.0 ** -44, k)
        for k in ("<", "max", "floor", "rint"):
            self.assertEqual(worst[k], 0.0, k)

    def run_program(self, l, refs, where: Path):
        """the program on the shim at its reference points: (ME errors, worst output error relative to the ME)"""
        codegen.compile_metal(where, l)
        lib = build_shim(ROOT / "test" / "metal_shim_run.cpp", where / "libshim_run.so", [cm.METAL_DIR, where])
        program = (where / "program.bin").read_bytes()
        points, values = refs
        n = l.output_count()
        momenta = np.ascontiguousarray(np.array(points, dtype=np.float64).reshape(-1))
        out = np.zeros(n * len(points))
        status = lib.shim_run(ctypes.c_char_p(program), momenta.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(len(points)),
                              out.ctypes.data_as(ctypes.c_void_p))
        self.assertEqual(status, 0, f"{l.process} {l.colour}: status {status}")
        me_errors, out_worst = [], 0.0
        for p, ref in enumerate(values):
            ref = np.asarray(ref, dtype=np.float64)
            got = out[p * n:(p + 1) * n]
            me = ref.sum()
            me_errors.append(abs(got.sum() - me) / abs(me))
            out_worst = max(out_worst, float(np.max(np.abs(got - ref))) / abs(me))
        return me_errors, out_worst

    def test_programs(self):
        every = bool(os.environ.get("PACRUNNER_TEST_METAL_ALL"))
        checked = 0
        for (name, folder, export, _) in golden_sets():
            ls, refs = uir.load_recurrence_export(export), uir.recurrence_references(export)
            fixture = folder.parent.name == "goldens"
            picked = (range(len(ls)) if every or fixture or not (folder / "compile_metal").is_dir()
                      else [int(e.name) for e in sorted((folder / "compile_metal").iterdir())])
            me, outs = [], 0.0
            for k in picked:
                if refs[k] is None:
                    continue
                with tempfile.TemporaryDirectory() as where:
                    e, o = self.run_program(ls[k], refs[k], Path(where))
                me += e
                outs = max(outs, o)
                checked += 1
            if me:
                with self.subTest(export=name):
                    print(f"  {name}: {len(me)} points: ME error median {np.median(me):.1e}, max {max(me):.1e}; "
                          f"outputs (relative to the ME) max {outs:.1e}")
                    self.assertLess(max(me), 1e-4)
                    self.assertLess(float(np.median(me)), 1e-6)
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

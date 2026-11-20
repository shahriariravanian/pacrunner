"""pacrunner.kernels: the kernels' MIR through symjit's MIR JSON (symjit >= 2.29) and their C.

- the C of every kernel of the golden sets (goldens.py) equals pac-julia's, byte for byte, in full
  and in mixed precision;
- the C, built with cc, gives the bits symjit's own native kernel gives (special inputs and random
  ones; any two NaNs count as the same: the sign of a NaN from 0/0 follows the operand order);
- julia_repr against Julia's repr (test/data/julia_repr.txt), and the C of hand-made MIR.

Skipped without symjit >= 2.29 (symjit.mir_json); the native comparison also without cc.

    python -m unittest discover -s test/python -p test_kernels.py
"""

import ctypes
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from goldens import FIXTURE_EXPORTS, ROOT, golden_sets

from pacrunner import kernels, uir
from pacrunner._literal import julia_repr

HAVE_MIR_JSON = kernels.symjit_problem() is None   # symjit >= 2.29 (its native library's version)
if HAVE_MIR_JSON:
    import symjit
CC = shutil.which("cc")


def _double(h: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


@unittest.skipUnless(HAVE_MIR_JSON, "no symjit >= 2.29 (symjit.mir_json)")
class Goldens(unittest.TestCase):
    def test_kernel_c(self):
        compared = 0
        for (name, folder, export, _) in golden_sets():
            for sha, k in uir.export_kernels(uir.read_export(export)).items():
                mir = kernels.read_mir(k.application)
                for suffix, f in ((".c", kernels.kernel_c), (".mixed.c", kernels.kernel_c_mixed)):
                    golden = (folder / "kernels" / f"{sha}{suffix}").read_text()
                    self.assertEqual(f(k.name, mir), golden, f"{name}: {sha}{suffix}")
                    compared += 1
        self.assertGreater(compared, 0)

    def test_header(self):
        k = next(iter(uir.export_kernels(uir.read_export(FIXTURE_EXPORTS / "pp_3j.json")).values()))
        mir = kernels.read_mir(k.application)
        self.assertTrue(mir.complexified and mir.is_complex() and mir.option("direct_arena"))
        self.assertEqual(mir.count_params, 0)
        self.assertGreater(mir.count_states, 0)
        self.assertEqual(kernels.read_mir(k.application), kernels.parse_mir(symjit.mir_json(k.application)))


@unittest.skipUnless(HAVE_MIR_JSON and CC, "no symjit >= 2.29 or no cc")
class Native(unittest.TestCase):
    """the C against symjit's native kernel (the reference sjb2obj's libraries were in pac-julia)"""

    POINTS = 300

    @staticmethod
    def special(n):
        return [[0.0] * n, [-0.0] * n, [-1.5 if j % 2 == 0 else 0.0 for j in range(n)],
                [0.0 if j % 2 == 0 else -2.0 for j in range(n)], [1e-300 if j % 2 == 0 else -1e300 for j in range(n)],
                [1.0] * n]

    def test_bit_for_bit(self):
        import numpy as np
        rng = np.random.default_rng(11)
        ks = uir.export_kernels(uir.read_export(FIXTURE_EXPORTS / "pp_3j.json"))
        ks.update(uir.export_kernels(uir.read_export(FIXTURE_EXPORTS / "zggg_union.json")))
        with tempfile.TemporaryDirectory() as work:
            for sha, k in ks.items():
                mir = kernels.read_mir(k.application)
                src, so = Path(work) / f"{k.name}.c", Path(work) / f"{k.name}.so"
                src.write_text(kernels.KERNEL_PRELUDE + kernels.kernel_c(k.name, mir) +
                               f"\nint c_{k.name}(double *mem) {{ return {k.name}(mem); }}\n")
                subprocess.run([CC, "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(so),
                                str(src), "-lm"], check=True)
                fn = getattr(ctypes.CDLL(str(so)), f"c_{k.name}")
                fn.restype = ctypes.c_int
                sjb = Path(work) / f"{sha}.sjb"
                sjb.write_bytes(k.application)
                f = symjit.load_func(str(sjb))
                c = f.complex_compiler or f.compiler
                (n, no) = (mir.count_states, mir.count_obs)
                self.assertEqual((c.states.size, c.obs.size), (n, no))
                points = [np.array(x) for x in self.special(n)]
                points += [rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, n) for _ in range(self.POINTS)]
                for x in points:
                    c.states[:] = x
                    c.execute()
                    mem = np.zeros(n + no + 1)
                    mem[:n] = x
                    self.assertEqual(fn(mem.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), 0)
                    got, want = mem[n:n + no], np.asarray(c.obs)
                    same = (got.view(np.uint64) == want.view(np.uint64)) | (np.isnan(got) & np.isnan(want))
                    self.assertTrue(same.all(), f"{k.name}: {x} gives {got}, symjit {want}")


class Literals(unittest.TestCase):
    def test_julia_repr(self):
        lines = (ROOT / "test" / "data" / "julia_repr.txt").read_text().splitlines()
        self.assertGreater(len(lines), 2000)
        for line in lines:
            (h, want) = line.split()
            self.assertEqual(julia_repr(_double(h)), want, h)

    def test_const_c(self):
        self.assertEqual(kernels.const_c(float("nan")), "sj_double(0x7ff8000000000000ull)")
        self.assertEqual(kernels.const_c(float("inf")), "HUGE_VAL")
        self.assertEqual(kernels.const_c(float("-inf")), "-HUGE_VAL")
        self.assertEqual(kernels.const_c(-0.0), "-0.0")
        self.assertEqual(kernels.const_c(1e-5), "1.0e-5")
        self.assertEqual(kernels.label_c("@complex_root"), "L__40_complex_root")


# a real kernel, by hand: mem[2] = mem[0] / consts[0] (through Ret), mem[3] = |mem[1]| via a funclet
def _mir_json(code, consts=("0x4000000000000000",), labels=None, **options):
    return json.dumps({
        "format": "symjit-application", "format_version": 1, "symjit_version": "2.29.0",
        "program": {"count_states": 2, "count_params": 0, "count_obs": 2, "count_diffs": 0, "count_loops": 0,
                    "config": {"ty": "native", "options": {"complex": False, "direct_arena": True, **options}},
                    "builder": {"count_loops": 0, "count_stack": 0, "consts": list(consts), "ft": []}},
        "prepared": {"compiled": True, "fast": False, "simd": False},
        "mir": {"complexified": False, "code": code, "consts": list(consts), "labels": labels or {}},
        "reals": [],
    })


REAL = [
    {"Load": {"dst": {"Gen": 0}, "loc": {"Mem": 0}}},
    {"LoadConstMath": {"op": "Divide", "dst": {"Gen": 1}, "s1": {"Gen": 0}, "idx": 0}},
    {"Save": {"src": {"Gen": 1}, "loc": {"Mem": 2}}},
    {"Call": {"label": "@abs", "num_args": 0}},
    {"Branch": {"label": "@over"}},
    {"Label": {"label": "@abs"}},
    {"Load": {"dst": {"Gen": 0}, "loc": {"Stack": 4}}},
    {"Load": {"dst": "Ret", "loc": {"Mem": 1}}},
    {"Uni": {"op": "Abs", "dst": {"Gen": 0}, "s1": "Ret"}},
    {"Save": {"src": {"Gen": 0}, "loc": {"Mem": 3}}},
    {"Branch": {"label": ".ret"}},
    {"Label": {"label": "@over"}},
    "End",
]


class HandMade(unittest.TestCase):
    def test_real_kernel(self):
        mir = kernels.parse_mir(_mir_json(REAL))
        c = kernels.kernel_c("k", mir)
        self.assertIn("r0 = 2.0; g1 = g0 / r0;", c)          # a real model: LoadConstMath through Ret
        self.assertIn("ret = 0; goto L__40_abs; R_0:;", c)
        self.assertIn("case 0: goto R_0;", c)
        self.assertIn("double stack4 = 0.0;", c)
        m = kernels.kernel_c_mixed("k", mir)
        self.assertIn("g1 = (double)g0_f / r0;", m)          # the division and its divisor f64, the numerator f32
        self.assertIn("r0 = 2.0;", m)
        self.assertIn("g0_f = (float)mem[0];", m)
        if CC:
            with tempfile.TemporaryDirectory() as work:
                for name, text in (("full", c), ("mixed", m)):
                    src = Path(work) / f"{name}.c"
                    src.write_text("#define PAC_FN\n" + kernels.KERNEL_PRELUDE + text)
                    # it compiles (not -Werror: this made-up kernel sets g0 and never reads it)
                    subprocess.run([CC, "-std=c99", "-O2", "-c", "-o",
                                    str(Path(work) / f"{name}.o"), str(src)], check=True)

    def test_refusals(self):
        with self.assertRaises(kernels.KernelCError):
            kernels.kernel_c("k", kernels.parse_mir(_mir_json(REAL, direct_arena=False)))
        param = [{"Load": {"dst": {"Gen": 0}, "loc": {"Param": 0}}}]
        with self.assertRaises(kernels.KernelCError):
            kernels.kernel_c("k", kernels.parse_mir(_mir_json(param)))
        call = [{"Call": {"label": "sin", "num_args": 1}}]
        with self.assertRaises(kernels.KernelCError):
            kernels.kernel_c("k", kernels.parse_mir(_mir_json(call)))
        fused = [{"Fused": {"op": "Frobnicate", "dst": {"Gen": 0}, "a": {"Gen": 0}, "b": {"Gen": 0}, "c": {"Gen": 0}}}]
        for f in (kernels.kernel_c, kernels.kernel_c_mixed):   # an unknown operation: an error naming it, not a KeyError
            with self.assertRaises((kernels.KernelCError, kernels.MirError)) as e:
                f("k", kernels.parse_mir(_mir_json(fused)))
            self.assertIn("Frobnicate", str(e.exception))
        with self.assertRaises(kernels.MirError):
            kernels.parse_mir(_mir_json([{"Frobnicate": {}}]))
        with self.assertRaises(kernels.MirError):
            kernels.parse_mir(_mir_json([{"Mov": {"dst": {"Gen": 0}}}]))



class SymjitVersion(unittest.TestCase):
    """require_symjit: symjit >= 2.29, by its native library's version"""

    def problem(self, version):
        with mock.patch("pacrunner.kernels.mir.symjit_version", return_value=version):
            return kernels.symjit_problem()

    def test_versions(self):
        self.assertIn("not installed", self.problem(None))
        for old in ("2.28.3", "2.9.0", "1.3.3", "symjit 1.3.3", "nonsense"):
            self.assertIn("too old", self.problem(old), old)
        if HAVE_MIR_JSON:   # the installed symjit has mir_json
            for ok in ("2.29.0", "2.29.1", "2.30", "3.0.0"):
                self.assertIsNone(self.problem(ok), ok)

    def test_require_raises(self):
        with mock.patch("pacrunner.kernels.mir.symjit_version", return_value="2.28.3"):
            with self.assertRaises(kernels.MirError) as e:
                kernels.require_symjit()
            self.assertIn("symjit>=2.29", str(e.exception))

    @unittest.skipUnless(HAVE_MIR_JSON, "no symjit >= 2.29")
    def test_the_installed_symjit(self):
        self.assertEqual(kernels.require_symjit(), kernels.symjit_version())

    def test_the_mir_json_before_the_application_format(self):
        with self.assertRaises(kernels.MirError) as e:
            kernels.parse_mir(json.dumps({"format": "symjit-mir", "format_version": 1, "symjit_version": "2.29.0"}))
        self.assertIn("update symjit", str(e.exception))


if __name__ == "__main__":
    unittest.main()

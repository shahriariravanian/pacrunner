"""The mixed fragments (simd/mixed.py, written as MIR by simd/mir.py) against the Metal runner on the CPU.

Each fragment is compiled by symjit's JIT as an f64 + double-double SIMD kernel (AVX2 and, where the
CPU has it, AVX-512), downgraded to f32 + df64 (simd/downgrade.py) and run on random inputs; its
outputs must be the Metal sources' (c/metal/pacops.metal through msl_shim.h, test/metal_shim_ops.cpp,
without flushing denormals: the CPU keeps them) bit for bit.

  Propagators  propagator:<weyl_a|weyl_b|vector>:mixed against shim_propagator

Skipped without symjit >= 2.29, a C++ compiler (CXX, default c++) or an x86-64 CPU with AVX2.

    python -m unittest discover -s test/python -p test_mixed.py
"""

import ctypes
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from goldens import ROOT

from pacrunner import kernels
from pacrunner.codegen.compile_simd import host_lanes
from pacrunner.simd import downgrade as D
from pacrunner.simd import library, mixed
from test_downgrade import Machine

HAVE_SYMJIT = kernels.symjit_problem() is None
CXX = os.environ.get("CXX", "c++")
HAVE_CXX = shutil.which(CXX.split()[0]) is not None
C = ROOT / "pacrunner" / "c"
PROPAGATOR = {"weyl_a": 0, "weyl_b": 1, "vector": 2}     # pacops.h: PROPAGATOR_WEYL_A, _WEYL_B, _VECTOR
f32 = np.float32


def application(frag: dict, lanes: int) -> dict:
    """a fragment as a symjit application of its own: inputs the states, outputs the observables"""
    cfg = library.config()
    cfg["options"].update({"use_threads": False, "fastmath": False, "enable_simd512": lanes == 8})
    return {"format": "symjit-application", "format_version": 1, "symjit_version": "2.29.1",
            "program": {"config": cfg, "count_states": len(frag["inputs"]), "count_params": 0,
                        "count_obs": frag["outputs"], "count_diffs": 0, "count_loops": 0,
                        "builder": {"consts": frag["consts"], "count_loops": 0, "count_stack": frag["count_stack"],
                                    "ft": []}},
            "prepared": {"compiled": True, "fast": False, "simd": False},
            "mir": {"complexified": False, "code": frag["code"] + ["End"], "consts": frag["consts"], "labels": {}},
            "reals": []}


def simd_code(frag: dict, lanes: int, folder) -> bytes:
    import symjit
    f = symjit.load_json(json.dumps(application(frag, lanes)))
    path = os.path.join(folder, "k.bin")
    f.dump(path, "simd")
    with open(path, "rb") as fb:
        return fb.read()


def todd(x: float) -> tuple[float, float]:
    """a double as a df64 (hi, lo), as the Metal host stages it"""
    h = f32(x)
    return float(h), float(f32(x - float(h)))


@unittest.skipUnless(HAVE_SYMJIT and HAVE_CXX and 4 in host_lanes(), "no symjit >= 2.29, C++ compiler or AVX2")
class Propagators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        so = os.path.join(cls.tmp.name, "shim.so")
        subprocess.run([*CXX.split(), "-std=c++17", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-Wno-attributes",
                        "-Wno-unused-function", f"-I{C / 'metal'}", f"-I{C}", "-o", so,
                        str(ROOT / "test" / "metal_shim_ops.cpp")], check=True)
        cls.shim = ctypes.CDLL(so)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_propagators_are_the_metal_runners_bits(self):
        rng = np.random.default_rng(3)
        for kind in PROPAGATOR:
            frag = mixed.propagator(kind)
            n = 4 if kind == "vector" else 2
            for lanes in host_lanes():
                with self.subTest(kind=kind, lanes=lanes):
                    code32 = D.downgrade(simd_code(frag, lanes, self.tmp.name))
                    rows, want = [], []
                    for _ in range(16 * lanes):
                        c = [float(f32(x)) for x in rng.standard_normal(2 * n)]
                        k = rng.standard_normal(4) * 300
                        k[0] = abs(k[0]) * 1.2
                        fac = [float(f32(x)) for x in rng.standard_normal(2)]
                        F = fac
                        if kind == "vector":     # rmul(factor, (0, -1)) in f32, as the Metal runner
                            a, b = f32(fac[0]), f32(fac[1])
                            F = [float(a * f32(0.0) - b * f32(-1.0)), float(a * f32(-1.0) + b * f32(0.0))]
                        rows.append(c + [y for x in k for y in todd(x)] + F)
                        out = (ctypes.c_double * 8)()
                        self.shim.shim_propagator(out, PROPAGATOR[kind], (ctypes.c_double * 8)(*c, *[0.0] * (8 - 2 * n)),
                                                  (ctypes.c_double * 4)(*k), (ctypes.c_double * 2)(*fac))
                        want.append(list(out)[:2 * n])
                    want = np.array(want, dtype=np.float32)
                    got = Machine(code32).run(np.array(rows), 2 * n, 2 * lanes, np.float32).astype(np.float32)
                    differ = int((got.view(np.uint32) != want.view(np.uint32)).sum())
                    self.assertEqual(differ, 0, f"{differ} of {want.size} values differ")


if __name__ == "__main__":
    unittest.main()

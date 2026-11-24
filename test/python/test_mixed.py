"""The mixed fragments (simd/mixed.py, written as MIR by simd/mir.py) against the Metal runner on the CPU.

Each fragment is compiled by symjit's JIT as an f64 + double-double SIMD kernel (AVX2 and, where the
CPU has it, AVX-512), downgraded to f32 + df64 (simd/downgrade.py) and run on random inputs; its
outputs must be the Metal sources' (c/metal/pacops.metal through msl_shim.h, test/metal_shim_ops.cpp,
without flushing denormals: the CPU keeps them) bit for bit.

  Vertices       the shipped f64 vertex fragments (vertex:<kind>:<set|add>), downgraded, against
                 shim_vertex's o × s (+ the old current): Metal's vertex row, s = rmul(coupling, factor)
                 folded in f32 as the Metal runner computes it (the f64 fragments need no mixed twin)
  Propagators    propagator:<weyl_a|weyl_b|vector>:mixed against shim_propagator
  Wavefunctions  wave:<scalar|weyl|dirac|gluon|massive_vector>:...:mixed against shim_wavefunction, on
                 random momenta and the branches' edges (E < 0, on the z axis, pz = 0, at rest)
  Programs       whole programs (compile_simd.MixedAssembler, downgraded with its roles) against the
                 program's runner.metal on the CPU (compile_metal, test/metal_shim_run.cpp with
                 -DSHIM_KEEP_DENORMALS), bit for bit: per golden export without model kernels, sampled
                 programs as exported and with a leg permutation (a crossing), at the export's reference
                 points; and the matrix element against RustiCol's within mixed precision's target

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

from goldens import ROOT, golden_sets

from pacrunner import codegen, kernels, uir, umami
from pacrunner.codegen.compile_simd import MixedAssembler, host_lanes
from pacrunner.uir.types import is_kernel
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


def rmul(a, b):
    """pacops.metal's rmul in f32: (a.re b.re - a.im b.im, a.re b.im + a.im b.re), no contraction"""
    return (f32(a[0]) * f32(b[0]) - f32(a[1]) * f32(b[1]), f32(a[0]) * f32(b[1]) + f32(a[1]) * f32(b[0]))


def floats(rng, n, scale=1.0):
    return [float(f32(x)) for x in rng.standard_normal(n) * scale]


def momenta(rng, n, mass=0.0):
    """n momenta (E, px, py, pz): random, and the wavefunctions' edges"""
    out = []
    for i in range(n):
        p = rng.standard_normal(3) * 200
        e = float(np.sqrt(p @ p + mass * mass))
        if i % 7 == 1:
            p[:2] = 0.0                              # on the z axis
        if i % 7 == 2:
            p[:2] = 0.0
            p[2] = -abs(p[2])
        if i % 7 == 3:
            p[:] = 0.0                               # at rest (pt = pp = 0)
            e = mass or 1.0
        if i % 7 == 4:
            p[2] = 0.0
        if i % 2 == 1:
            e, p = -e, -p                            # E < 0 (crossed legs)
        out.append([e, *p])
    return out


class _Shim(unittest.TestCase):
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

    def run_f32(self, frag, rows, lanes):
        """the fragment downgraded, on rows of inputs (a multiple of 2 lanes): its outputs as floats"""
        code32 = D.downgrade(simd_code(frag, lanes, self.tmp.name), consts=frag["consts"])
        return Machine(code32).run(np.array(rows), frag["outputs"], 2 * lanes, np.float32).astype(np.float32)


@unittest.skipUnless(HAVE_SYMJIT and HAVE_CXX and 4 in host_lanes(), "no symjit >= 2.29, C++ compiler or AVX2")
class Vertices(_Shim):
    def test_f64_vertices_downgraded_are_the_metal_runners_bits(self):
        from pacrunner.simd.library import VERTICES, fragment
        rng = np.random.default_rng(5)
        types = {"weyl_vector_a": 0, "weyl_vector_b": 1, "tensor_vector": 2, "vector_wedge": 3, "three_vector": 4}
        for kind, (nl, nr, no) in VERTICES.items():
            for mode in ("set", "add"):
                frag = fragment(f"vertex:{kind}:{mode}")
                for lanes in host_lanes():
                    with self.subTest(kind=kind, mode=mode, lanes=lanes):
                        rows, want = [], []
                        for _ in range(16 * lanes):
                            l, r = floats(rng, 2 * nl), floats(rng, 2 * nr)
                            u, v = floats(rng, 4, 100), floats(rng, 4, 100)
                            scale, factor = floats(rng, 2), floats(rng, 2)
                            s = [float(x) for x in rmul(scale, factor)]
                            old = floats(rng, 2 * no)
                            o = (ctypes.c_double * 12)()
                            self.shim.shim_vertex(o, types[kind], (ctypes.c_double * 12)(*l, *[0.0] * (12 - 2 * nl)),
                                                  (ctypes.c_double * 12)(*r, *[0.0] * (12 - 2 * nr)),
                                                  (ctypes.c_double * 4)(*u), (ctypes.c_double * 4)(*v))
                            w = []
                            for j in range(no):
                                c = rmul((o[2 * j], o[2 * j + 1]), s)
                                if mode == "add":
                                    c = (f32(old[2 * j]) + c[0], f32(old[2 * j + 1]) + c[1])
                                w += [c[0], c[1]]
                            values = {"two": 2.0, "s.re": s[0], "s.im": s[1]}
                            for j in range(nl):
                                values[f"l{j}.re"], values[f"l{j}.im"] = l[2 * j], l[2 * j + 1]
                            for j in range(nr):
                                values[f"r{j}.re"], values[f"r{j}.im"] = r[2 * j], r[2 * j + 1]
                            for j in range(no):
                                values[f"old{j}.re"], values[f"old{j}.im"] = old[2 * j], old[2 * j + 1]
                            for j in range(4):
                                values[f"u{j}"], values[f"v{j}"] = u[j], v[j]
                            rows.append([values[n] for n in frag["inputs"]])
                            want.append(w)
                        want = np.array(want, dtype=np.float32)
                        got = self.run_f32(frag, rows, lanes)
                        differ = int((got.view(np.uint32) != want.view(np.uint32)).sum())
                        self.assertEqual(differ, 0, f"{differ} of {want.size} values differ")


@unittest.skipUnless(HAVE_SYMJIT and HAVE_CXX and 4 in host_lanes(), "no symjit >= 2.29, C++ compiler or AVX2")
class Wavefunctions(_Shim):
    CASES = ([("scalar", 0, 0, 0, 0, False)]
             + [("weyl", 1, a, h, ch, False) for a in (0, 1) for h in (1, -1) for ch in (1, -1)]
             + [("dirac", 2, a, h, 0, massive) for a in (0, 1) for h in (1, -1) for massive in (False, True)]
             + [("gluon", 3, 0, h, 0, False) for h in (1, -1)]
             + [("massive_vector", 3, 0, h, 0, True) for h in (1, 0, -1)])

    @staticmethod
    def frag(kind, anti, h, ch, massive):
        side = "anti" if anti else "part"
        if kind == "scalar":
            return mixed.wave_scalar()
        if kind == "weyl":
            return mixed.wave_weyl(bool(anti), h, ch)
        if kind == "dirac":
            return mixed.wave_dirac_massive(bool(anti), h) if massive else mixed.wave_dirac_massless(bool(anti), h)
        if kind == "gluon":
            return mixed.wave_gluon(h)
        return mixed.wave_massive_vector(h)

    def test_wavefunctions_are_the_metal_runners_bits(self):
        rng = np.random.default_rng(11)
        for kind, family, anti, h, ch, massive in self.CASES:
            frag = self.frag(kind, anti, h, ch, massive)
            n = frag["outputs"] // 4
            for lanes in host_lanes():
                with self.subTest(fragment=frag["name"], lanes=lanes):
                    mass = 91.188 if kind == "massive_vector" else 4.7 if massive else 0.0
                    rows, want = [], []
                    for k in momenta(rng, 16 * lanes, mass):
                        out = (ctypes.c_double * 8)()
                        got_n = self.shim.shim_wavefunction(out, family, anti, (ctypes.c_double * 4)(*k), h, ch,
                                                            ctypes.c_double(mass))
                        self.assertEqual(got_n, n)
                        rows.append([y for x in k for y in todd(x)] + (list(todd(mass)) if "mass.hi" in frag["inputs"] else []))
                        want.append(list(out)[:2 * n])
                    got = self.run_f32(frag, rows, lanes).astype(np.float64)
                    value = got[:, 0::2] + got[:, 1::2]          # hi + lo per (re, im)
                    want = np.array(want)
                    same = (value == want) | (np.isnan(value) & np.isnan(want))
                    self.assertTrue(same.all(), f"{int((~same).sum())} of {same.size} values differ: "
                                                f"{value[~same.all(axis=1)][:2]} vs {want[~same.all(axis=1)][:2]}")


@unittest.skipUnless(HAVE_SYMJIT and HAVE_CXX and 4 in host_lanes(), "no symjit >= 2.29, C++ compiler or AVX2")
class Propagators(_Shim):
    def test_propagators_are_the_metal_runners_bits(self):
        rng = np.random.default_rng(3)
        for kind in PROPAGATOR:
            frag = mixed.propagator(kind)
            n = 4 if kind == "vector" else 2
            for lanes in host_lanes():
                with self.subTest(kind=kind, lanes=lanes):
                    code32 = D.downgrade(simd_code(frag, lanes, self.tmp.name), consts=frag["consts"])
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


SETS = [(n.split("/")[-1], e) for (n, d, e, m) in golden_sets()]


def split_columns(X):
    """momenta X[points, 4 legs] as the mixed kernel's states: the hi parts, then the lo parts"""
    hi = X.astype(np.float32).astype(np.float64)
    lo = (X - hi).astype(np.float32).astype(np.float64)
    return np.hstack([hi, lo])


@unittest.skipUnless(HAVE_SYMJIT and HAVE_CXX and 4 in host_lanes(), "no symjit >= 2.29, C++ compiler or AVX2")
class Programs(unittest.TestCase):
    PER = 2           # programs sampled per export

    def shim(self, l, X, where):
        import importlib
        cm = importlib.import_module("pacrunner.codegen.compile_metal")
        codegen.compile_metal(where, l)
        so = os.path.join(where, "shim.so")
        subprocess.run([*CXX.split(), "-std=c++17", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-Wno-attributes",
                        "-Wno-unused-function", "-DSHIM_KEEP_DENORMALS", f"-I{cm.METAL_DIR}", f"-I{where}", "-o", so,
                        str(ROOT / "test" / "metal_shim_run.cpp")], check=True)
        lib = ctypes.CDLL(so)
        n = l.output_count()
        m = np.ascontiguousarray(X.reshape(-1))
        out = np.zeros(n * len(X))
        with open(os.path.join(where, "program.bin"), "rb") as f:
            program = f.read()
        status = lib.shim_run(ctypes.c_char_p(program), m.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(len(X)),
                              out.ctypes.data_as(ctypes.c_void_p))
        self.assertEqual(status, 0)
        return out.reshape(len(X), n)

    def test_programs_are_the_metal_runners_bits(self):
        import symjit
        checked = 0
        for name, export in SETS:
            ls, refs = uir.load_recurrence_export(export), uir.recurrence_references(export)
            cand = [k for k, l in enumerate(ls) if refs[k] is not None and not any(is_kernel(o.op) for o in l.ops)]
            if not cand:
                continue
            for k in sorted({cand[int(i)] for i in np.linspace(0, len(cand) - 1, self.PER)}):
                pts = np.array([np.array(p, dtype=float).ravel() for p in refs[k][0]])
                for variant in ("exported", "permuted"):
                    l = ls[k]
                    if variant == "permuted":
                        perm = list(range(l.legs))
                        perm[0], perm[-1] = perm[-1], perm[0]
                        l = umami.permuted(l, perm)
                    with self.subTest(export=name, program=k, variant=variant), tempfile.TemporaryDirectory() as where:
                        a = MixedAssembler(l)
                        app = a.assemble()
                        n = l.output_count()
                        for lanes in host_lanes():
                            X = np.resize(pts, (4 * lanes, pts.shape[1]))
                            want = self.shim(l, X, where) if lanes == host_lanes()[0] else want
                            app["program"]["config"]["options"]["enable_simd512"] = lanes == 8
                            f = symjit.load_json(json.dumps(app))
                            path = os.path.join(where, "k.bin")
                            f.dump(path, "simd")
                            with open(path, "rb") as fb:
                                code = fb.read()
                            code32 = D.downgrade(code, consts=app["mir"]["consts"], roles=a.roles)
                            got = Machine(code32).run(split_columns(X), 2 * n, 2 * lanes, np.float32)
                            value = got[:, :n] + got[:, n:]
                            w = np.resize(want, value.shape)
                            same = (value == w) | (np.isnan(value) & np.isnan(w))
                            self.assertTrue(same.all(), f"lanes {lanes}: {int((~same).sum())} of {same.size} differ")
                            if variant == "exported":
                                me, ref = value.sum(axis=1), np.resize(np.array(refs[k][1]).sum(axis=1), len(value))
                                self.assertLess(float(np.max(np.abs(me / ref - 1))), 1e-5)
                        checked += 1
        if not checked:
            self.skipTest("no golden program without model kernels")


if __name__ == "__main__":
    unittest.main()

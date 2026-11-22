"""device "simd": the library of pacops operations (pacrunner/simd/), compile_simd's kernels and the UMAMI
libraries built from them.

  Library   every fragment of the catalogue shipped and readable; fragments compiled again by this symjit
            equal the shipped ones (when it is the symjit that wrote them)
  Kernels   per golden export, sampled programs as exported and with a leg permutation (the libraries'
            crossing): the kernel (symjit's JIT, scalar and SIMD) against the straight-line C runner
            (codegen.compile, cc), bit for bit, on the export's reference points and random momenta
            (both signs of E, beams on the z axis)
  Umami     the simd libraries of small providers against the cpu ones through the UMAMI API (ctypes),
            bit for bit: drawn and fixed helicities, 4 and 8 lanes (as the CPU has them), several chunks,
            an offset and a stride; needs symjit-utils' sjb2obj with --simd (PACRUNNER_SJB2OBJ or the PATH)
  Units     the device's checks

Skipped without the corpus (goldens), symjit >= 2.29, cc, or (Umami) sjb2obj and an x86-64 CPU with AVX2.

    python -m unittest discover -s test/python -p test_simd.py
"""

import ctypes
import json
import os
import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from goldens import golden_sets

from pacrunner import _env, codegen, kernels, uir, umami
from pacrunner.codegen.common import ArenaIndex
from pacrunner.codegen.compile_simd import Assembler, SimdError, find_sjb2obj, simd_application
from pacrunner.simd import library

HAVE_SYMJIT = kernels.symjit_problem() is None
HAVE_CC = shutil.which("cc") is not None
SETS = {n.split("/")[-1]: (e, m) for (n, d, e, m) in golden_sets() if n.startswith("pacrunner-goldens/")}
PROGRAMS = {"zggg_union": 1}   # programs sampled per export (default 2): zggg_union's are large


def c_runner(prog, folder):
    """the straight-line C runner of `prog` as run(momenta) -> outputs"""
    codegen.compile(folder, prog)
    so = Path(folder) / "librunner.so"
    subprocess.run(["cc", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-I", str(folder), "-o", str(so),
                    str(Path(folder) / "runner.c"), str(Path(folder) / "pacops.c"), "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    P = ctypes.POINTER(ctypes.c_double)
    arena = np.zeros(ArenaIndex.of(prog).size)

    def run(m):
        out = np.zeros(prog.output_count())
        lib.run(out.ctypes.data_as(P), arena.ctypes.data_as(P), np.ascontiguousarray(m).ctypes.data_as(P))
        return out
    return run


def points(legs, refs, n=48, seed=7):
    rng = np.random.default_rng(seed)
    pts = [np.array(m, dtype=float).ravel() for m in refs[0]] if refs is not None else []
    for i in range(n):
        p3 = rng.standard_normal((legs, 3)) * 100
        if i % 5 == 0:
            p3[:2, :2] = 0.0                     # beams on the z axis
        e = np.linalg.norm(p3, axis=1) * rng.choice([1.0, 1.3], legs) * rng.choice([-1.0, 1.0], legs)
        pts.append(np.column_stack([e, p3]).ravel())
    return np.array(pts)


def differing(got, want):
    """the points whose outputs differ in some bit (NaN = NaN)"""
    same = (got.view(np.uint64) == want.view(np.uint64)) | (np.isnan(got) & np.isnan(want))
    return int((~same).any(axis=1).sum())


class Library(unittest.TestCase):
    def test_the_catalogue_is_shipped(self):
        for name in library.catalogue():
            with self.subTest(name=name):
                d = library.fragment(name)
                self.assertEqual(d["name"], name)
                self.assertGreater(len(d["code"]), 0)
                self.assertGreater(d["outputs"], 0)
        self.assertIn("options", library.config())

    def test_an_unknown_fragment(self):
        with self.assertRaisesRegex(library.LibraryError, "no fragment"):
            library.fragment("vertex:five_vector:set")

    @unittest.skipUnless(HAVE_SYMJIT, "no symjit >= 2.29")
    def test_compiled_again(self):
        from importlib.metadata import version
        names = ["vertex:three_vector:add", "propagator:weyl_a", "closure:4", "wave:massive_vector:0",
                 "wave:dirac:anti:1:massive"]
        if library.fragment(names[0])["symjit_version"] != version("symjit"):
            self.skipTest("the library was written by another symjit")
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(library.compile_fragment(name)[0], library.fragment(name))


@unittest.skipUnless(SETS and HAVE_SYMJIT and HAVE_CC, "no corpus, symjit >= 2.29 or cc")
class Kernels(unittest.TestCase):
    def test_kernels_are_the_c_runners_bits(self):
        import symjit
        for name, (export, m) in sorted(SETS.items()):
            progs = uir.load_recurrence_export(export)
            refs = uir.recurrence_references(export)
            picks = sorted(set(np.linspace(0, len(progs) - 1, min(PROGRAMS.get(name, 2), len(progs))).astype(int)))
            for k in picks:
                prog = uir.optimize(progs[k])
                perm = list(range(prog.legs))
                perm[0], perm[-1] = perm[-1], perm[0]
                for variant, l in (("exported", prog), ("permuted", umami.permuted(prog, perm))):
                    with self.subTest(export=name, program=k, variant=variant), tempfile.TemporaryDirectory() as d:
                        app, stats = simd_application(l)
                        g = symjit.load_json(json.dumps(app))
                        run = c_runner(l, d)
                        X = points(l.legs, refs[k] if variant == "exported" else None)
                        want = np.array([run(x) for x in X])
                        scalar = np.array([g(*x) for x in X])
                        simd = np.asarray(g(*X.T)).reshape(l.output_count(), len(X)).T
                        self.assertEqual(differing(scalar, want), 0, "scalar")
                        self.assertEqual(differing(simd, want), 0, "SIMD")
                        self.assertGreater(stats["calls"], 0)


# ---- the UMAMI libraries through ctypes ----------------------------------------------------------------------

IN_MOMENTA, IN_RANDOM_HELICITY, IN_HELICITY_INDEX = 0, 4, 6
OUT_MATRIX_ELEMENT, OUT_COLOR_INDEX, OUT_HELICITY_INDEX = 0, 2, 3


class Library_(object):
    """one UMAMI library: matrix_element(momenta [events, particles, 4], random or fixed helicities)"""

    def __init__(self, path):
        self.so = ctypes.CDLL(str(path))
        self.handle = ctypes.c_void_p()
        if self.so.umami_initialize(ctypes.byref(self.handle), None) != 0:
            raise RuntimeError(f"{path}: umami_initialize failed")

    def call(self, p, random=None, fixed=None, offset=2, pad=3):
        events, n = p.shape[0], p.shape[1]
        stride = events + offset + pad
        mom = np.zeros(stride * n * 4)
        for j in range(n):
            for k in range(4):
                mom[offset + stride * (n * k + j): offset + stride * (n * k + j) + events] = p[:, j, k]
        keys, ins = [IN_MOMENTA], [mom.ctypes.data]
        if random is not None:
            r = np.zeros(stride)
            r[offset:offset + events] = random
            keys.append(IN_RANDOM_HELICITY)
            ins.append(r.ctypes.data)
        if fixed is not None:
            h = np.zeros(stride, dtype=np.int32)
            h[offset:offset + events] = fixed
            keys.append(IN_HELICITY_INDEX)
            ins.append(h.ctypes.data)
        me = np.full(stride, np.nan)
        colour = np.zeros(stride, dtype=np.int32)
        hel = np.full(stride, -7, dtype=np.int32)
        okeys, outs = [OUT_MATRIX_ELEMENT, OUT_COLOR_INDEX], [me.ctypes.data, colour.ctypes.data]
        if fixed is None:
            okeys.append(OUT_HELICITY_INDEX)
            outs.append(hel.ctypes.data)
        K = ctypes.c_int * len(keys)
        V = ctypes.c_void_p * len(ins)
        OK = ctypes.c_int * len(okeys)
        OV = ctypes.c_void_p * len(outs)
        status = self.so.umami_matrix_element(self.handle, ctypes.c_size_t(events), ctypes.c_size_t(stride),
                                              ctypes.c_size_t(offset), ctypes.c_size_t(len(keys)), K(*keys), V(*ins),
                                              ctypes.c_size_t(len(okeys)), OK(*okeys), OV(*outs))
        if status != 0:
            raise RuntimeError(f"umami_matrix_element: {status}")
        return me[offset:offset + events], colour[offset:offset + events], hel[offset:offset + events]

    def close(self):
        self.so.umami_free(self.handle)


def simd_cpu() -> list:
    """the lane counts this CPU runs (4: AVX2 and FMA, 8: AVX-512 F and DQ)"""
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return []
    out = []
    if " avx2" in flags and " fma" in flags:
        out.append(4)
    if " avx512f" in flags and " avx512dq" in flags:
        out.append(8)
    return out


@unittest.skipUnless(SETS and HAVE_SYMJIT and HAVE_CC and _env.simd_machine() and simd_cpu() and find_sjb2obj(),
                     "no corpus, symjit >= 2.29, cc, x86-64 with AVX2, or sjb2obj with --simd")
class Umami(unittest.TestCase):
    def test_simd_libraries_are_the_cpu_ones(self):
        rng = np.random.default_rng(11)
        for name in ("zgg", "gg_3g", "uu_ttgg"):
            if name not in SETS:
                continue
            export, m = SETS[name]
            with tempfile.TemporaryDirectory() as out:
                tables = {}
                for device in ("cpu", "simd"):
                    path = umami.umami_libraries(export, m["metadata"], Path(out) / device, device=device, quiet=True)
                    tables[device] = json.loads(Path(path).read_text())
                self.assertEqual(tables["simd"]["device"], "simd")
                meta = json.loads(Path(m["metadata"]).read_text())
                helicities = len(meta["helicities"])
                for k, entry in enumerate(tables["cpu"]["libraries"]):
                    n = int(tables["cpu"]["particle_count"])
                    events = 150
                    p3 = rng.uniform(-300, 300, (events, n, 3))
                    p = np.concatenate([np.linalg.norm(p3, axis=2)[:, :, None], p3], axis=2)
                    random = rng.uniform(0, 1, events)
                    fixed = rng.integers(0, helicities, events).astype(np.int32)
                    cpu = Library_(Path(out) / "cpu" / entry["path"])
                    want = [cpu.call(p, random=random), cpu.call(p, fixed=fixed)]
                    cpu.close()
                    for lanes in simd_cpu():
                        with self.subTest(export=name, library=k, lanes=lanes), \
                                mock.patch.dict(os.environ, {"PAC_SIMD_LANES": str(lanes)}):
                            simd = Library_(Path(out) / "simd" / tables["simd"]["libraries"][k]["path"])
                            got = [simd.call(p, random=random), simd.call(p, fixed=fixed)]
                            simd.close()
                            for (g, w) in zip(got, want):
                                self.assertTrue(np.array_equal(g[0].view(np.uint64), w[0].view(np.uint64)))
                                self.assertTrue(np.array_equal(g[1], w[1]))
                                self.assertTrue(np.array_equal(g[2], w[2]))


class Units(unittest.TestCase):
    def test_devices(self):
        self.assertEqual(_env.device_name("simd"), "simd")
        with mock.patch.object(_env.platform, "machine", lambda: "x86_64"):
            self.assertEqual(_env.compile_target("simd"), ("simd", "f64"))
            with self.assertRaisesRegex(_env.PacRunnerError, "needs a GPU device"):
                _env.compile_target("simd", "mixed")
        with mock.patch.object(_env.platform, "machine", lambda: "arm64"):
            with self.assertRaisesRegex(_env.PacRunnerError, "x86-64 only"):
                _env.compile_target("simd")

    def test_simd_needs_sjb2obj(self):
        if not (SETS and HAVE_SYMJIT):
            self.skipTest("no corpus or symjit >= 2.29")
        export, m = SETS["zgg"]
        with mock.patch.object(umami, "find_sjb2obj", lambda: None), \
                mock.patch.object(umami._env, "simd_machine", lambda: True), tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(umami.UmamiError, "sjb2obj"):
                umami.umami_libraries(export, m["metadata"], out, device="simd", quiet=True)

    def test_folder_names(self):
        from pacrunner._mad import library_folder_name
        with mock.patch.object(_env.platform, "machine", lambda: "x86_64"):
            self.assertEqual(library_folder_name("simd"), "pacrunner-simd")


if __name__ == "__main__":
    unittest.main()

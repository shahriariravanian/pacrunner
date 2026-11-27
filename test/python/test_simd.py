"""The SIMD devices (simd-avx2, simd-avx512, simd-neon; "simd" this machine's): the library of pacops operations (pacrunner/simd/), compile_simd's kernels and the UMAMI
libraries built from them.

  Library   every fragment of the catalogue shipped and readable; fragments compiled again by this symjit
            equal the shipped ones (when it is the symjit that wrote them)
  Kernels   per golden export, sampled programs as exported and with a leg permutation (the libraries'
            crossing): the kernel (symjit's JIT, scalar and SIMD) against the straight-line C runner
            (codegen.compile, cc), bit for bit, on the export's reference points and random momenta
            (both signs of E, beams on the z axis)
  Umami     the libraries of small providers on every SIMD device of this machine against the cpu ones
            through the UMAMI API (ctypes), bit for bit: drawn and fixed helicities, several chunks, an
            offset and a stride
  Objects   the object files (simd/elf.py): their code is symjit's JIT code (x86-64: an ELF object with no
            relocations and the symbol; aarch64: the bytes an assembler file includes); a kernel calling a
            function refused
  Units     the device's checks

Skipped without the corpus (goldens), symjit >= 2.29, cc, or (Umami, Objects) an x86-64 CPU with AVX2.

    python -m unittest discover -s test/python -p test_simd.py
"""

import ctypes
import json
import os
import re
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
from pacrunner.codegen.compile_simd import ISAS, LANES, Assembler, SimdError, simd_application
from pacrunner.simd import elf, library

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
        names = ["vertex:three_vector:add", "propagator:weyl_a", "closure:4", "wave:massive_vector:0",
                 "wave:dirac:anti:1:massive"]
        # the symjit that compiles now (its own version, not the installed distribution's: a symjit on
        # PYTHONPATH may be another)
        if library.compile_fragment(names[0])[0]["symjit_version"] != library.fragment(names[0])["symjit_version"]:
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


DEVICES = _env.simd_devices()   # this machine's SIMD devices


@unittest.skipUnless(SETS and HAVE_SYMJIT and HAVE_CC and DEVICES, "no corpus, symjit >= 2.29, cc, or a SIMD device")
class Umami(unittest.TestCase):
    def test_simd_libraries_are_the_cpu_ones(self):
        rng = np.random.default_rng(11)
        for name in ("zgg", "gg_3g", "uu_ttgg"):
            if name not in SETS:
                continue
            export, m = SETS[name]
            with tempfile.TemporaryDirectory() as out:
                tables = {}
                for device in ("cpu", *DEVICES, "cpu-jit"):
                    path = umami.umami_libraries(export, m["metadata"], Path(out) / device, device=device, quiet=True)
                    tables[device] = json.loads(Path(path).read_text())
                    self.assertEqual(tables[device]["device"], device)
                    self.assertIsNone(tables[device]["gpu_memory"])   # no GPU (the table's GPU fields)
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
                    for device in (*DEVICES, "cpu-jit"):
                        with self.subTest(export=name, library=k, device=device):
                            simd = Library_(Path(out) / device / tables[device]["libraries"][k]["path"])
                            got = [simd.call(p, random=random), simd.call(p, fixed=fixed)]
                            simd.close()
                            for (g, w) in zip(got, want):
                                self.assertTrue(np.array_equal(g[0].view(np.uint64), w[0].view(np.uint64)))
                                self.assertTrue(np.array_equal(g[1], w[1]))
                                self.assertTrue(np.array_equal(g[2], w[2]))


    def test_a_thread_with_too_small_a_stack_is_refused(self):
        """the kernel's frame lives on the calling thread's stack: too little left is UMAMI_ERROR, not a crash"""
        import threading
        export, m = SETS["zgg"]
        device = DEVICES[-1]                        # the widest lanes: the largest frame
        with tempfile.TemporaryDirectory() as out:
            table = json.loads(Path(umami.umami_libraries(export, m["metadata"], out, device=device, quiet=True)).read_text())
            path = Path(out) / table["libraries"][0]["path"]
            h = (path.parent / "kernel.h").read_text()
            need = int(re.search(r"PAC_KERNEL_STACK_BYTES (\d+)", h).group(1)) + 16384   # PAC_SIMD_STACK_MARGIN
            small = 32768                               # threading's least stack
            if need <= small:
                self.skipTest(f"zgg's {device} kernel needs only {need} bytes")
            n = int(table["particle_count"])
            p3 = np.random.default_rng(5).uniform(-300, 300, (8, n, 3))
            p = np.concatenate([np.linalg.norm(p3, axis=2)[:, :, None], p3], axis=2)
            lib = Library_(path)
            want = lib.call(p, random=np.full(8, 0.5))[0]

            def in_thread(stack):
                result = {}

                def body():
                    try:
                        result["me"] = lib.call(p, random=np.full(8, 0.5))[0]
                    except RuntimeError as e:
                        result["error"] = str(e)
                old = threading.stack_size(stack)
                try:
                    t = threading.Thread(target=body)
                    t.start()
                    t.join()
                finally:
                    threading.stack_size(old)
                return result

            refused = in_thread(small)
            self.assertIn("umami_matrix_element: 1", refused.get("error", ""), refused)   # UMAMI_ERROR
            ok = in_thread(max(16 << 20, need * 4))
            self.assertNotIn("error", ok)
            self.assertTrue(np.array_equal(ok["me"].view(np.uint64), want.view(np.uint64)))
            lib.close()


@unittest.skipUnless(SETS and HAVE_SYMJIT and DEVICES, "no corpus, symjit >= 2.29, or a SIMD device")
class Objects(unittest.TestCase):
    def test_objects_hold_the_jit_code(self):
        import symjit
        export, m = SETS["zgg"]
        prog = uir.optimize(uir.load_recurrence_export(export)[0])
        for device in DEVICES:
            isa, n = ISAS[device], LANES[ISAS[device]]
            with self.subTest(device=device), tempfile.TemporaryDirectory() as d:
                codegen.compile_simd(d, prog, isa=isa)
                app = json.loads((Path(d) / "kernel.json").read_text())
                a = json.loads(json.dumps(app))
                a["program"]["config"]["options"]["enable_simd512"] = isa == "avx512"
                symjit.load_json(json.dumps(a)).dump(str(Path(d) / "jit.bin"), "simd")
                jit = (Path(d) / "jit.bin").read_bytes()
                if isa == "neon":
                    self.assertEqual((Path(d) / "kernel.bin").read_bytes(), jit)
                    src = (Path(d) / "kernel.S").read_text()
                    self.assertIn(f'.incbin "{Path(d).resolve() / "kernel.bin"}"', src)
                    self.assertIn(".p2align 12", src)
                else:
                    obj = (Path(d) / "kernel.o").read_bytes()
                    self.assertEqual(obj[4096:4096 + len(jit)], jit)   # .text at 4096: the ELF writer's layout
                    if shutil.which("readelf"):
                        r = subprocess.run(["readelf", "-rsW", str(Path(d) / "kernel.o")], capture_output=True, text=True)
                        self.assertIn("There are no relocations", r.stdout)
                        self.assertRegex(r.stdout, r"FUNC +GLOBAL +DEFAULT +1 pac_kernel")
                h = (Path(d) / "kernel.h").read_text()
                self.assertIn(f"#define PAC_KERNEL_LANES {n}", h)
                self.assertGreater(int(re.search(r"PAC_KERNEL_STACK_BYTES (\d+)", h).group(1)),
                                   app["program"]["builder"]["count_stack"] * 8 * n)
                self.assertIn(f"#define PAC_SIMD_ISA_{isa.upper()} 1", (Path(d) / "runner.h").read_text())

    def test_cpu_jit_object_is_the_scalar_code(self):
        import symjit
        export, m = SETS["zgg"]
        prog = uir.optimize(uir.load_recurrence_export(export)[0])
        isa = _env.cpu_jit_isa()
        with tempfile.TemporaryDirectory() as d:
            codegen.compile_simd(d, prog, scalar=True)
            app = json.loads((Path(d) / "kernel.json").read_text())
            simd_app, _ = simd_application(prog, isa=isa)     # the SIMD device's application
            self.assertEqual(len(app["mir"]["code"]), len(simd_app["mir"]["code"]))
            symjit.load_json(json.dumps(app)).dump(str(Path(d) / "jit.bin"), "scalar")
            jit = (Path(d) / "jit.bin").read_bytes()
            if isa == "neon":
                self.assertEqual((Path(d) / "kernel.bin").read_bytes(), jit)
            else:
                self.assertEqual((Path(d) / "kernel.o").read_bytes()[4096:4096 + len(jit)], jit)
            h = (Path(d) / "kernel.h").read_text()
            self.assertIn("#define PAC_KERNEL_LANES 1", h)
            self.assertIn("A scalar kernel (1 lane)", h)
            r = (Path(d) / "runner.h").read_text()
            self.assertIn('#define PAC_SIMD_DEVICE "cpu-jit"', r)
            self.assertIn(f"#define PAC_SIMD_ISA_{isa.upper()} 1", r)
            with self.assertRaisesRegex(SimdError, "f64 only"):
                codegen.compile_simd(d, prog, precision="split", scalar=True)

    def test_another_machine_s_device_is_refused(self):
        export, m = SETS["zgg"]
        other = [d for d in _env.SIMD_DEVICES if d not in DEVICES]
        prog = uir.optimize(uir.load_recurrence_export(export)[0])
        for device in other:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(SimdError, "to build on"):
                    codegen.compile_simd(d, prog, isa=ISAS[device])
                with self.assertRaisesRegex(umami.UmamiError, "does not build on this machine"):
                    umami.umami_libraries(export, m["metadata"], d, device=device, quiet=True)


class Units(unittest.TestCase):
    def machine(self, machine, flags=""):
        """this machine as another: platform.machine() and /proc/cpuinfo's flags"""
        return mock.patch.multiple(_env, platform=mock.Mock(machine=lambda: machine),
                                   _cpu_flags=lambda: set(flags.split()))

    def test_devices(self):
        with self.machine("x86_64", "sse avx2 fma avx512f avx512dq"):
            self.assertEqual(_env.simd_devices(), ("simd-avx2", "simd-avx512"))
            self.assertEqual(_env.device_name("simd"), "simd-avx512")
            self.assertEqual(_env.compile_target("simd"), ("simd-avx512", "f64"))
            self.assertEqual(_env.compile_target("simd-split"), ("simd-avx512", "split"))
            self.assertEqual(_env.compile_target("simd", "split"), ("simd-avx512", "split"))
            self.assertEqual(_env.compile_target("simd-avx2-split"), ("simd-avx2", "split"))
            self.assertEqual(_env.compile_target("simd-avx2", "mixed"), ("simd-avx2", "split"))   # the former name
            self.assertEqual(_env.library_name("simd-split"), "simd-avx512-split")
            self.assertEqual(_env.library_name("simd-avx2"), "simd-avx2")
            with self.assertRaisesRegex(_env.PacRunnerError, "does not build on this machine"):
                _env.compile_target("simd-neon")
            with self.assertRaisesRegex(_env.PacRunnerError, "needs a SIMD device"):
                _env.compile_target("cpu", "split")
            with self.assertRaisesRegex(_env.PacRunnerError, "needs a SIMD device"):
                _env.compile_target("cuda", "split")
            with self.assertRaisesRegex(_env.PacRunnerError, "needs a GPU device"):
                _env.compile_target("cpu", "mixed")
            self.assertEqual(_env.compile_target("cpu-jit"), ("cpu-jit", "f64"))
            self.assertEqual(_env.library_name("cpu-jit"), "cpu-jit")
            self.assertEqual(_env.cpu_jit_isa(), "avx2")      # avx2's fragments also where AVX-512 is
            with self.assertRaisesRegex(_env.PacRunnerError, "f64 only"):
                _env.compile_target("cpu-jit", "split")
            with self.assertRaisesRegex(_env.PacRunnerError, "f64 only"):
                _env.compile_target("cpu-jit", "mixed")
        with self.machine("x86_64", "sse avx2 fma"):
            self.assertEqual(_env.device_name("simd"), "simd-avx2")
            self.assertEqual(_env.cpu_jit_isa(), "avx2")
            with self.assertRaisesRegex(_env.PacRunnerError, "does not build on this machine"):
                _env.compile_target("simd-avx512")
        for arm in ("aarch64", "arm64"):
            with self.machine(arm):
                self.assertEqual(_env.compile_target("simd-split"), ("simd-neon", "split"))
                self.assertEqual(_env.library_name("simd-neon-split"), "simd-neon-split")
                self.assertEqual(_env.compile_target("cpu-jit"), ("cpu-jit", "f64"))
                self.assertEqual(_env.cpu_jit_isa(), "neon")
                with self.assertRaisesRegex(_env.PacRunnerError, "does not build on this machine"):
                    _env.compile_target("simd-avx2")
        with self.machine("riscv64"):
            self.assertEqual(_env.simd_devices(), ())
            with self.assertRaisesRegex(_env.PacRunnerError, "device 'simd'"):
                _env.compile_target("simd")
            with self.assertRaisesRegex(_env.PacRunnerError, "does not build on this machine"):
                _env.compile_target("cpu-jit")
            self.assertIsNone(_env.cpu_jit_isa())
        with self.machine("x86_64", "sse avx"):     # no AVX2: no SIMD device, no cpu-jit
            with self.assertRaisesRegex(_env.PacRunnerError, "does not build on this machine"):
                _env.compile_target("cpu-jit")

    def test_simd_needs_a_simd_device(self):
        if not (SETS and HAVE_SYMJIT):
            self.skipTest("no corpus or symjit >= 2.29")
        export, m = SETS["zgg"]
        with self.machine("riscv64"), tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(umami.UmamiError, "device 'simd'"):
                umami.umami_libraries(export, m["metadata"], out, device="simd", quiet=True)

    def test_calls_are_refused(self):
        with self.assertRaisesRegex(elf.ObjectError, "relocation"):
            elf.no_relocations([{"Call": {"label": "sin", "num_args": 1}}])
        elf.no_relocations([{"Call": {"label": "@f_x_ultra", "num_args": 0}}, "End"])
        with self.assertRaisesRegex(elf.ObjectError, "identifier"):
            elf.elf_object("not a name", b"\xc3")

    def test_folder_names(self):
        from pacrunner._mad import library_folder_name
        with self.machine("x86_64", "avx2 fma avx512f avx512dq"):
            self.assertEqual(library_folder_name("simd"), "pacrunner-simd-avx512")
            self.assertEqual(library_folder_name("simd", "split"), "pacrunner-simd-avx512-split")
            self.assertEqual(library_folder_name("simd-avx2-split"), "pacrunner-simd-avx2-split")
        with self.machine("aarch64"):
            self.assertEqual(library_folder_name("simd-split"), "pacrunner-simd-neon-split")


if __name__ == "__main__":
    unittest.main()

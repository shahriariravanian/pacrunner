"""device "simd", precision "mixed": the split kernels (codegen/compile_split.py, simd/split_kernel.py,
plans/F32-KERNEL.md "Split kernels"): kernel A in f64 (momenta to the transfers), kernel B in f32 (the
transfers to the outputs: an f64 kernel downgraded by simd/downgrade.py).

  Splits    every model kernel of the golden exports: its Writer steps are kernel_c's _MixedWriter's; its A
            side then its B side, both in f64, compute the kernel's f64 fragment bit for bit (random inputs)
  Programs  per golden export, sampled programs as exported and with a leg permutation: A (f64), the
            transfers rounded to f32, B (downgraded: every constant already an f32 value) at twice the
            lanes; the matrix element within 2e-6 of the export's references (RustiCol, f64)
  Umami     the simd mixed library of a small provider against the cpu one through the UMAMI API: the
            matrix element within 2e-6, 8 and 16 lanes (as the CPU has them), several chunks

Skipped without the corpus (goldens), symjit >= 2.29, cc, or an x86-64 CPU with AVX2.

    python -m unittest discover -s test/python -p test_split.py
"""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from goldens import golden_sets

from pacrunner import _env, kernels, uir, umami
from pacrunner.codegen.common import used_kernels
from pacrunner.codegen.compile_simd import host_lanes, kernel_fragment, simd_application
from pacrunner.simd import downgrade as D
from pacrunner.simd import library, split_kernel
from pacrunner.uir.types import InputKind
from test_downgrade import Machine
from test_simd import HAVE_CC, Library_

CS = importlib.import_module("pacrunner.codegen.compile_split")
HAVE_SYMJIT = kernels.symjit_problem() is None
SETS = [(n.split("/")[-1], e, m) for (n, d, e, m) in golden_sets() if n.startswith("pacrunner-goldens/")]
CAN_RUN = bool(SETS) and HAVE_SYMJIT and _env.simd_machine() and 4 in host_lanes()
TOLERANCE = 2e-6     # f32 currents and amplitudes: worst seen 8.1e-7 (gg_4g)
PER = {"zggg_union": 1}


def jit(app: dict, lanes: int, folder) -> bytes:
    """the SIMD kernel of an application (enable_simd512 for 8 lanes)"""
    import symjit
    a = json.loads(json.dumps(app))
    a["program"]["config"]["options"]["enable_simd512"] = lanes == 8
    path = os.path.join(folder, "k.bin")
    symjit.load_json(json.dumps(a)).dump(path, "simd")
    with open(path, "rb") as f:
        return f.read()


def fragment_app(frag: dict) -> dict:
    """a fragment as a symjit application of its own: inputs the states, outputs the observables"""
    cfg = library.config()
    cfg["options"].update({"use_threads": False, "fastmath": False})
    return {"format": "symjit-application", "format_version": 1, "symjit_version": "2.29.1",
            "program": {"config": cfg, "count_states": len(frag["inputs"]), "count_params": 0,
                        "count_obs": frag["outputs"], "count_diffs": 0, "count_loops": 0,
                        "builder": {"consts": frag["consts"], "count_loops": 0, "count_stack": frag["count_stack"],
                                    "ft": []}},
            "prepared": {"compiled": True, "fast": False, "simd": False},
            "mir": {"complexified": False, "code": frag["code"] + ["End"], "consts": frag["consts"], "labels": {}},
            "reals": []}


@unittest.skipUnless(CAN_RUN, "no corpus, symjit >= 2.29, or x86-64 with AVX2")
class Splits(unittest.TestCase):
    def test_a_then_b_is_the_kernel(self):
        seen = set()
        for name, export, _ in SETS:
            for l in uir.load_recurrence_export(export):
                for st in l.stubs:
                    info = next(i for i in used_kernels(l) if i.kernel.sha256 == st.kernel.sha256)
                    kinds = ["current" if i.kind in (InputKind.IN_L, InputKind.IN_R, InputKind.OUT) else "static"
                             for i in st.inputs]
                    key = (info.kernel.sha256, tuple(kinds))
                    if key in seen:
                        continue
                    seen.add(key)
                    with self.subTest(export=name, kernel=info.kernel.name), tempfile.TemporaryDirectory() as d:
                        mir = split_kernel.prepared(info.kernel.name, info.mir)
                        split_kernel.check_steps(info.kernel.name, mir)
                        sp = split_kernel.split(info.kernel.name, info.mir, kinds, info.inputs, info.outputs)
                        self.assertTrue(sp.a, "no A side: the kernel has no wide step")
                        X = np.random.default_rng(len(seen)).standard_normal((8, info.inputs)) * 50
                        static = [k for k in range(info.inputs) if kinds[k] == "static"]
                        want = Machine(jit(fragment_app(kernel_fragment(info.kernel)), 4, d)).run(
                            X, info.outputs, 4, np.float64)
                        cross = (Machine(jit(fragment_app(sp.a_fragment()), 4, d)).run(
                            X[:, static], len(sp.crossing), 4, np.float64) if sp.crossing else np.zeros((8, 0)))
                        got = Machine(jit(fragment_app(sp.b_fragment(f32=False)), 4, d)).run(
                            np.hstack([X, cross]), info.outputs, 4, np.float64)
                        self.assertTrue(np.array_equal(got.view(np.uint64), want.view(np.uint64)))
        if not seen:
            self.skipTest("no model kernel in the golden exports")


@unittest.skipUnless(CAN_RUN, "no corpus, symjit >= 2.29, or x86-64 with AVX2")
class Programs(unittest.TestCase):
    def test_programs_within_f32_accuracy(self):
        checked = 0
        for name, export, _ in SETS:
            ls, refs = uir.load_recurrence_export(export), uir.recurrence_references(export)
            cand = [k for k in range(len(ls)) if refs[k] is not None]
            if not cand:
                continue
            for k in sorted({cand[int(i)] for i in np.linspace(0, len(cand) - 1, PER.get(name, 2))}):
                pts = np.array([np.array(p, dtype=float).ravel() for p in refs[k][0]])
                ref = np.array(refs[k][1]).sum(axis=1)
                for variant in ("exported", "permuted"):
                    l = ls[k]
                    if variant == "permuted":
                        perm = list(range(l.legs))
                        perm[0], perm[-1] = perm[-1], perm[0]
                        l = umami.permuted(l, perm)
                    with self.subTest(export=name, program=k, variant=variant), tempfile.TemporaryDirectory() as d:
                        app_a, app_b, roles, items = CS.split_applications(l)
                        n = l.output_count()
                        f64, _ = simd_application(l)
                        for lanes in host_lanes():
                            X = np.resize(pts, (4 * lanes, pts.shape[1]))
                            T = Machine(jit(app_a, lanes, d)).run(X, len(items), lanes, np.float64)
                            T32 = T.astype(np.float32).astype(np.float64)
                            code = D.downgrade(jit(app_b, lanes, d), consts=app_b["mir"]["consts"], roles=roles)
                            got = Machine(code).run(T32, n, 2 * lanes, np.float32)
                            me = got.sum(axis=1)
                            if variant == "exported":
                                err = np.abs(me / np.resize(ref, len(me)) - 1)
                            else:          # no references for the permutation: the f64 kernel's
                                w = Machine(jit(f64, lanes, d)).run(X, n, lanes, np.float64).sum(axis=1)
                                err = np.abs(me / w - 1)
                            self.assertLess(float(np.max(err)), TOLERANCE, f"lanes {lanes}")
                        checked += 1
        if not checked:
            self.skipTest("no golden program with references")


@unittest.skipUnless(CAN_RUN and HAVE_CC, "no corpus, symjit >= 2.29, cc, or x86-64 with AVX2")
class Umami(unittest.TestCase):
    def test_library_against_cpu(self):
        export, m = next((e, m) for (n, e, m) in SETS if n == "zgg")
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as out:
            cpu = json.loads(Path(umami.umami_libraries(export, m["metadata"], Path(out) / "cpu", quiet=True)).read_text())
            mixed = json.loads(Path(umami.umami_libraries(export, m["metadata"], Path(out) / "mixed", device="simd",
                                                          precision="mixed", quiet=True)).read_text())
            folder = Path(out) / "mixed" / Path(mixed["libraries"][0]["path"]).parent
            self.assertIn("PAC_SIMD_SPLIT", (folder / "runner.h").read_text())
            n = int(cpu["particle_count"])
            p3 = rng.uniform(-300, 300, (150, n, 3))     # three chunks of 64, the last one short
            p = np.concatenate([np.linalg.norm(p3, axis=2)[:, :, None], p3], axis=2)
            random = rng.uniform(0, 1, len(p))     # drawn helicities: the matrix element is the sum
            for lib_c, lib_m in zip(cpu["libraries"], mixed["libraries"]):
                a = Library_(Path(out) / "cpu" / lib_c["path"])
                want = a.call(p, random=random)
                a.close()
                for lanes in [str(2 * x) for x in host_lanes()]:
                    with self.subTest(library=lib_m["path"], lanes=lanes), \
                            mock.patch.dict(os.environ, {"PAC_SIMD_LANES": lanes}):
                        b = Library_(Path(out) / "mixed" / lib_m["path"])
                        got = b.call(p, random=random)
                        b.close()
                        ok = want[0] != 0
                        self.assertTrue(ok.any())
                        self.assertLess(float(np.max(np.abs(got[0][ok] / want[0][ok] - 1))), TOLERANCE)


if __name__ == "__main__":
    unittest.main()

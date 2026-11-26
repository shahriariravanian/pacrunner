"""The split kernels (precision "split" of a SIMD device: codegen/compile_split.py, simd/split_kernel.py,
plans/F32-KERNEL.md "Split kernels"): kernel A in f64 (momenta to the transfers), kernel B in f32 (the
transfers to the outputs: symjit's f32 twin of its f64 kernel, dump(.., "simd-f32")).

  Splits    every model kernel of the golden exports: its Writer steps are kernel_c's _MixedWriter's; its A
            side then its B side, both in f64, compute the kernel's f64 fragment bit for bit (random inputs)
  Programs  per golden export, sampled programs as exported and with a leg permutation, for every SIMD device
            of this machine: A (f64), the transfers rounded to f32, B (f32) at twice the lanes; the matrix
            element within 2e-6 of the export's references (RustiCol, f64)
  Umami     the split libraries of a small provider (every SIMD device of this machine, as
            "<device>-split") against the cpu one through the UMAMI API: the matrix element within 2e-6,
            several chunks

Skipped without the corpus (goldens), symjit >= 2.29.2 (simd-f32), cc, or a SIMD device (simd-avx2,
simd-avx512, simd-neon).

    python -m unittest discover -s test/python -p test_split.py
"""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from goldens import golden_sets

from pacrunner import _env, kernels, uir, umami
from pacrunner.codegen.common import used_kernels
from pacrunner.codegen.compile_simd import ISAS, LANES, kernel_fragment, simd_application
from pacrunner.simd import library, split_kernel
from pacrunner.uir.types import InputKind
from simd_machine import HAVE_CC, machine
from test_simd import Library_

CS = importlib.import_module("pacrunner.codegen.compile_split")
HAVE_SYMJIT = kernels.symjit_problem() is None
SETS = [(n.split("/")[-1], e, m) for (n, d, e, m) in golden_sets() if n.startswith("pacrunner-goldens/")]
DEVICES = _env.simd_devices()        # this machine's SIMD devices: their kernels
CAN_RUN = bool(SETS) and HAVE_SYMJIT and HAVE_CC and bool(DEVICES)
ISA = ISAS[DEVICES[0]] if DEVICES else None
TOLERANCE = 2e-6     # f32 currents and amplitudes: worst seen 8.1e-7 (gg_4g)
PER = {"zggg_union": 1}


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


@unittest.skipUnless(CAN_RUN, "no corpus, symjit >= 2.29.2, cc, or a SIMD device")
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
                        n = LANES[ISA]
                        want = machine(fragment_app(kernel_fragment(info.kernel)), ISA).run(
                            X, info.outputs, n, np.float64)
                        cross = (machine(fragment_app(sp.a_fragment()), ISA).run(
                            X[:, static], len(sp.crossing), n, np.float64) if sp.crossing else np.zeros((8, 0)))
                        got = machine(fragment_app(sp.b_fragment(f32=False)), ISA).run(
                            np.hstack([X, cross]), info.outputs, n, np.float64)
                        self.assertTrue(np.array_equal(got.view(np.uint64), want.view(np.uint64)))
        if not seen:
            self.skipTest("no model kernel in the golden exports")


@unittest.skipUnless(CAN_RUN, "no corpus, symjit >= 2.29.2, cc, or a SIMD device")
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
                        app_a, app_b, _, items = CS.split_applications(l)
                        n = l.output_count()
                        f64, _ = simd_application(l)
                        for device in DEVICES:
                            isa = ISAS[device]
                            lanes = LANES[isa]
                            X = np.resize(pts, (8 * lanes, pts.shape[1]))
                            T = machine(app_a, isa).run(X, len(items), lanes, np.float64)
                            T32 = T.astype(np.float32).astype(np.float64)
                            got = machine(app_b, isa, f32=True).run(T32, n, 2 * lanes, np.float32)
                            me = got.sum(axis=1)
                            if variant == "exported":
                                err = np.abs(me / np.resize(ref, len(me)) - 1)
                            else:          # no references for the permutation: the f64 kernel's
                                w = machine(f64, isa).run(X, n, lanes, np.float64).sum(axis=1)
                                err = np.abs(me / w - 1)
                            self.assertLess(float(np.max(err)), TOLERANCE, device)
                        checked += 1
        if not checked:
            self.skipTest("no golden program with references")


@unittest.skipUnless(CAN_RUN and HAVE_CC, "no corpus, symjit >= 2.29.2, cc, or a SIMD device")
class Umami(unittest.TestCase):
    def test_library_against_cpu(self):
        export, m = next((e, m) for (n, e, m) in SETS if n == "zgg")
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as out:
            cpu = json.loads(Path(umami.umami_libraries(export, m["metadata"], Path(out) / "cpu", quiet=True)).read_text())
            split = {}
            for device in DEVICES:
                split[device] = json.loads(Path(umami.umami_libraries(export, m["metadata"], Path(out) / device,
                                                                      device=f"{device}-split", quiet=True)).read_text())
                self.assertEqual((split[device]["device"], split[device]["precision"]), (device, "split"))
                folder = Path(out) / device / Path(split[device]["libraries"][0]["path"]).parent
                self.assertIn("PAC_SIMD_SPLIT", (folder / "runner.h").read_text())
            n = int(cpu["particle_count"])
            p3 = rng.uniform(-300, 300, (150, n, 3))     # three chunks of 64, the last one short
            p = np.concatenate([np.linalg.norm(p3, axis=2)[:, :, None], p3], axis=2)
            random = rng.uniform(0, 1, len(p))     # drawn helicities: the matrix element is the sum
            for k, lib_c in enumerate(cpu["libraries"]):
                a = Library_(Path(out) / "cpu" / lib_c["path"])
                want = a.call(p, random=random)
                a.close()
                for device in DEVICES:
                    with self.subTest(library=lib_c["path"], device=device):
                        b = Library_(Path(out) / device / split[device]["libraries"][k]["path"])
                        got = b.call(p, random=random)
                        b.close()
                        ok = want[0] != 0
                        self.assertTrue(ok.any())
                        self.assertLess(float(np.max(np.abs(got[0][ok] / want[0][ok] - 1))), TOLERANCE)


if __name__ == "__main__":
    unittest.main()

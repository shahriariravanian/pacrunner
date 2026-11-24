"""simd/downgrade.py: an f64 SIMD kernel of symjit's JIT turned into its f32 twin.

  Encodings  every rewrite rule against the assembler: each packed or scalar double instruction the
             kernels use, assembled by GNU as, downgraded, equals as's encoding of its single twin
  Refusals   an instruction with no rule, a kernel already in f32, an embedded broadcast
  Kernels    per golden export, sampled programs (compile_simd, symjit's JIT, AVX2 and, where the CPU
             has it, AVX-512): the code decodes up to the data area, the downgraded kernel runs (twice
             the lanes per call) and its outputs are the f64 kernel's to f32 accuracy on the export's
             reference points

Skipped without the corpus, symjit >= 2.29, GNU as (Encodings) or an x86-64 CPU with AVX2 (Kernels).

    python -m unittest discover -s test/python -p test_downgrade.py
"""

import ctypes
import json
import mmap
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from goldens import golden_sets

from pacrunner import kernels, uir
from pacrunner.codegen.compile_simd import host_lanes, simd_application
from pacrunner.simd import downgrade as D

HAVE_SYMJIT = kernels.symjit_problem() is None
HAVE_AS = shutil.which("as") is not None and os.uname().machine == "x86_64"
SETS = {n.split("/")[-1]: (e, m) for (n, d, e, m) in golden_sets() if n.startswith("pacrunner-goldens/")}

# (double, single): each pair assembled on its own, so a RIP-relative operand points past the code
PAIRS = [
    ("vmovupd ymm0, ymmword ptr [rax+r12*8]", "vmovups ymm0, ymmword ptr [rax+r12*8]"),
    ("vmovupd ymmword ptr [r13+0x20], ymm0", "vmovups ymmword ptr [r13+0x20], ymm0"),
    ("vmovapd ymm3, ymm9", "vmovaps ymm3, ymm9"),
    ("vaddpd ymm1, ymm2, ymm3", "vaddps ymm1, ymm2, ymm3"),
    ("vsubpd ymm9, ymm10, ymm11", "vsubps ymm9, ymm10, ymm11"),
    ("vmulpd ymm1, ymm2, ymmword ptr [rbx+0x3800]", "vmulps ymm1, ymm2, ymmword ptr [rbx+0x3800]"),
    ("vdivpd ymm1, ymm2, ymm3", "vdivps ymm1, ymm2, ymm3"),
    ("vsqrtpd ymm1, ymm2", "vsqrtps ymm1, ymm2"),
    ("vandpd ymm1, ymm2, ymm3", "vandps ymm1, ymm2, ymm3"),
    ("vandnpd ymm1, ymm2, ymm3", "vandnps ymm1, ymm2, ymm3"),
    ("vorpd ymm1, ymm2, ymm3", "vorps ymm1, ymm2, ymm3"),
    ("vxorpd ymm5, ymm9, ymmword ptr [rip+0]", "vxorps ymm5, ymm9, ymmword ptr [rip+0]"),
    ("vcmppd ymm1, ymm2, ymm3, 2", "vcmpps ymm1, ymm2, ymm3, 2"),
    ("vminpd ymm1, ymm2, ymm3", "vminps ymm1, ymm2, ymm3"),
    ("vroundpd ymm1, ymm2, 9", "vroundps ymm1, ymm2, 9"),
    ("vfmadd231pd ymm1, ymm2, ymm3", "vfmadd231ps ymm1, ymm2, ymm3"),
    ("vfmadd132pd ymm11, ymm12, ymm13", "vfmadd132ps ymm11, ymm12, ymm13"),
    ("vfmsub231pd ymm1, ymm2, ymm3", "vfmsub231ps ymm1, ymm2, ymm3"),
    ("vfnmadd213pd ymm1, ymm2, ymm3", "vfnmadd213ps ymm1, ymm2, ymm3"),
    ("vbroadcastsd ymm2, qword ptr [rip+0]", "vbroadcastss ymm2, dword ptr [rip+0]"),
    ("vaddsd xmm1, xmm2, xmm3", "vaddss xmm1, xmm2, xmm3"),
    ("vmovupd zmm0, zmmword ptr [rax+r12*8]", "vmovups zmm0, zmmword ptr [rax+r12*8]"),
    ("vaddpd zmm1, zmm2, zmm3", "vaddps zmm1, zmm2, zmm3"),
    ("vmulpd zmm17, zmm18, zmmword ptr [rbx+0x40]", "vmulps zmm17, zmm18, zmmword ptr [rbx+0x40]"),
    ("vxorpd zmm5, zmm9, zmmword ptr [rip+0]", "vxorps zmm5, zmm9, zmmword ptr [rip+0]"),
    ("vcmppd k1, zmm2, zmm3, 1", "vcmpps k1, zmm2, zmm3, 1"),
    ("vfmadd231pd zmm1, zmm2, zmm3", "vfmadd231ps zmm1, zmm2, zmm3"),
    ("vbroadcastsd zmm2, qword ptr [rip+0]", "vbroadcastss zmm2, dword ptr [rip+0]"),
    ("vbroadcastsd zmm2, qword ptr [rax+8]", "vbroadcastss zmm2, dword ptr [rax+8]"),
    ("vpmovm2q zmm3, k1", "vpmovm2d zmm3, k1"),
    ("vpmovq2m k1, zmm1", "vpmovd2m k1, zmm1"),
    ("vsqrtpd zmm1, zmm2", "vsqrtps zmm1, zmm2"),
]
KEPT = ["knotw k1, k2", "movabs rcx, 0x14a014901480147", "shr rax, 0x10", "movzx rax, ax",
        "call .+0x100", "vzeroupper", "push rbp", "ret"]


def assemble(lines):
    """the machine code of `lines` (Intel syntax), one instruction per element, in order"""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "a.s")
        with open(src, "w") as f:
            f.write("\t.intel_syntax noprefix\n" + "".join(f"\t{x}\n" for x in lines))
        subprocess.run(["as", "-o", os.path.join(d, "a.o"), src], check=True)
        subprocess.run(["objcopy", "-O", "binary", "-j", ".text", os.path.join(d, "a.o"), os.path.join(d, "a.bin")],
                       check=True)
        with open(os.path.join(d, "a.bin"), "rb") as f:
            return f.read()


@unittest.skipUnless(HAVE_AS, "no GNU as on x86-64")
class Encodings(unittest.TestCase):
    def test_each_double_becomes_its_single_twin(self):
        for double, single in PAIRS:
            with self.subTest(double):
                code = assemble([double])
                self.assertEqual(bytes(D.rewrite_code(code)).hex(), assemble([single]).hex())

    def test_integer_and_mask_instructions_are_kept(self):
        for line in KEPT:
            with self.subTest(line):
                code = assemble([line])
                stats = {}
                self.assertEqual(bytes(D.rewrite_code(code, stats=stats)), code)
                self.assertEqual(stats, {"kept": 1})


@unittest.skipUnless(HAVE_AS, "no GNU as on x86-64")
class Refusals(unittest.TestCase):
    def test_no_rule(self):
        with self.assertRaisesRegex(D.DowngradeError, "no f32 rewrite"):
            D.rewrite_code(assemble(["vpermilpd ymm1, ymm2, 5"]))

    def test_twice(self):
        with self.assertRaisesRegex(D.DowngradeError, "single-precision FMA"):
            D.rewrite_code(D.rewrite_code(assemble(["vfmadd231pd ymm1, ymm2, ymm3"])))

    def test_embedded_broadcast(self):
        with self.assertRaisesRegex(D.DowngradeError, "embedded broadcast"):
            D.rewrite_code(assemble(["vaddpd zmm1, zmm2, qword ptr [rax]{1to8}"]))

    def test_constants(self):
        bits = lambda x: int.from_bytes(np.float64(x).tobytes(), "little")
        self.assertEqual(D._float_bits(bits(1.5)), 0x3FC000003FC00000)
        self.assertEqual(D._float_bits(bits(-0.0)), 0x8000000080000000)
        self.assertEqual(D._float_bits(0x7FFFFFFFFFFFFFFF), 0x7FFFFFFF7FFFFFFF)
        self.assertEqual(D._float_bits(0xFFFFFFFFFFFFFFFF), 0xFFFFFFFFFFFFFFFF)
        with self.assertRaisesRegex(D.DowngradeError, "has no float"):
            D._float_bits(bits(1e300))


class _Slice(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("len", ctypes.c_size_t)]


class Machine:
    """a kernel's code in executable memory, called as pac_simd<L>(NULL, states, idx, NULL)"""

    def __init__(self, code):
        self.m = mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        self.m.write(code)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self.m))
        self.f = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_void_p)(addr)

    def run(self, X, outputs, lanes, dtype):
        """outputs[points, outputs] of the momenta X[points, 4 legs] (the points a multiple of lanes)"""
        cols = [np.ascontiguousarray(X[:, k], dtype=dtype) for k in range(X.shape[1])]
        cols += [np.zeros(len(X), dtype=dtype) for _ in range(outputs)]
        sl = (_Slice * len(cols))(*[_Slice(c.ctypes.data, len(X)) for c in cols])
        for i in range(0, len(X), lanes):
            # the kernel reads element idx at byte 8 x idx: floats at idx / 2
            if self.f(None, ctypes.addressof(sl), i if dtype == np.float64 else i // 2, None) != 0:
                raise RuntimeError("the kernel failed")
        return np.column_stack(cols[X.shape[1]:]).astype(np.float64)


@unittest.skipUnless(SETS and HAVE_SYMJIT and 4 in host_lanes(), "no corpus, symjit >= 2.29 or AVX2")
class Kernels(unittest.TestCase):
    def test_f32_kernels_run_and_agree(self):
        import symjit
        for name, (export, m) in sorted(SETS.items()):
            progs = uir.load_recurrence_export(export)
            refs = uir.recurrence_references(export)
            for k in sorted({0, len(progs) - 1}):
                prog = uir.optimize(progs[k])
                if refs[k] is None:
                    continue
                pts = np.array([np.array(p, dtype=float).ravel() for p in refs[k][0]])
                for lanes in host_lanes():
                    with self.subTest(export=name, program=k, lanes=lanes), tempfile.TemporaryDirectory() as d:
                        app, _ = simd_application(prog)
                        app["program"]["config"]["options"]["enable_simd512"] = lanes == 8
                        f = symjit.load_json(json.dumps(app))
                        f.dump(os.path.join(d, "k.bin"), "simd")
                        with open(os.path.join(d, "k.bin"), "rb") as fb:
                            code = fb.read()
                        start = D.data_start(code, app["mir"]["consts"])
                        self.assertEqual(D.data_start(code), start)
                        self.assertEqual(sum(i.length for i in D.decode(code, start)), start)
                        stats = {}
                        code32 = D.downgrade(code, stats, consts=app["mir"]["consts"])
                        self.assertGreater(stats["ps"], 0)
                        # whole calls: the reference points repeated to 2 x 2 lanes
                        X = np.resize(pts, (4 * lanes, pts.shape[1]))
                        no = prog.output_count()
                        v64 = Machine(code).run(X, no, lanes, np.float64)
                        v32 = Machine(code32).run(X, no, 2 * lanes, np.float32)
                        self.assertTrue(np.isfinite(v32).all())
                        me64, me32 = v64.sum(axis=1), v32.sum(axis=1)
                        rel = np.abs(me32 / me64 - 1)
                        self.assertLess(rel.max(), 1e-3, f"matrix element: {rel.max():.2e}")


if __name__ == "__main__":
    unittest.main()

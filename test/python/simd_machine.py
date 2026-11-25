"""A SIMD kernel's machine code as a callable, for the tests: the code as compile_simd writes it (an x86-64
ELF object or an aarch64 assembler file, simd/elf.py) linked by cc into a shared library of its own and
loaded with ctypes. No executable mmap of our own: macOS needs MAP_JIT for that, and aarch64 an
instruction-cache flush; the dynamic linker does both.

    m = Machine(code, "avx2")          # or "avx512", "neon"
    out = m.run(X, outputs, lanes, np.float64)

`run` calls the kernel the way umami_pac.c does: one column per state (the columns of X) and per output,
kernel(NULL, columns, i, NULL) for every `lanes` points; float columns (an f32 kernel) are called with
i / 2 (element i is at byte 4 i = 8 (i / 2)).
"""

import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np

from pacrunner.codegen.compile_simd import ISAS, LANES, jit_kernel
from pacrunner.simd import elf

CC = os.environ.get("CC", "cc")
HAVE_CC = shutil.which(CC.split()[0]) is not None


class _Slice(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("len", ctypes.c_size_t)]


class Machine:
    """a kernel's code (bytes) for the instruction set `isa`, callable"""

    def __init__(self, code: bytes, isa: str):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        obj = elf.write_object(d, "kernel", "pac_kernel_test", code, isa)
        so = os.path.join(d, "kernel.so")
        # the symbol is hidden in the object: a C file exports a caller of it
        with open(os.path.join(d, "export.c"), "w") as f:
            f.write("#include <stddef.h>\nint pac_kernel_test(double *, const void *, size_t, const double *);\n"
                    "__attribute__((visibility(\"default\"))) int kernel_entry(double *m, const void *s, size_t i, "
                    "const double *p) { return pac_kernel_test(m, s, i, p); }\n")
        subprocess.run([*CC.split(), "-shared", "-fPIC", "-o", so, os.path.join(d, "export.c"), os.path.join(d, obj)],
                       check=True)
        self.lib = ctypes.CDLL(so)
        self.f = self.lib.kernel_entry
        self.f.restype = ctypes.c_int
        self.f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    def run(self, X, outputs, lanes, dtype):
        """outputs[points, outputs] of the states X[points, states] (the points a multiple of lanes)"""
        cols = [np.ascontiguousarray(X[:, k], dtype=dtype) for k in range(X.shape[1])]
        cols += [np.zeros(len(X), dtype=dtype) for _ in range(outputs)]
        sl = (_Slice * len(cols))(*[_Slice(c.ctypes.data, len(X)) for c in cols])
        for i in range(0, len(X), lanes):
            if self.f(None, ctypes.addressof(sl), i if dtype == np.float64 else i // 2, None) != 0:
                raise RuntimeError("the kernel failed")
        return np.column_stack(cols[X.shape[1]:]).astype(np.float64)


def machine(app: dict, isa: str, f32: bool = False) -> Machine:
    """the SIMD kernel of a symjit application for `isa` (symjit's JIT; its f32 twin with f32)"""
    with tempfile.TemporaryDirectory() as d:
        return Machine(jit_kernel(app, isa, "simd-f32" if f32 else "simd", d), isa)


__all__ = ["Machine", "machine", "HAVE_CC", "ISAS", "LANES"]

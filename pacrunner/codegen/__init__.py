"""pacrunner.codegen: a program as C (pac-julia's compile.jl and compile_vm.jl): straight-line code, or
the UIR interpreter's tables (interpreter=True); compile_metal (compile_metal.jl): the program for the
Metal runner (Apple GPUs, c/metal/); compile_simd: the program as one symjit kernel for AVX2, AVX-512
or NEON, as an object (the SIMD devices simd-avx2, simd-avx512, simd-neon; precision split: two kernels,
f64 and f32).

    from pacrunner import uir, codegen
    ls = uir.load_recurrence_export("export.json")
    refs = uir.recurrence_references("export.json")
    codegen.compile("out", uir.optimize(ls[0]), examples=codegen.Examples.of(ls[0], refs[0]))
    # out/: runner.c, runner.h, pacops.c/h, Makefile, testing.c (`make test`)
"""

from .common import ArenaIndex, CompileError, Examples, KernelInfo, kernel_info, lit, used_kernels
from .compile import compile
from .compile_metal import compile_metal
from .compile_simd import compile_simd
from .compile_vm import check_interpretable, compile_interpreter, gpu_bytes_per_point, vm_unsupported

__all__ = ["ArenaIndex", "CompileError", "Examples", "KernelInfo", "kernel_info", "lit", "used_kernels", "compile",
           "compile_metal", "compile_simd", "check_interpretable", "compile_interpreter", "gpu_bytes_per_point", "vm_unsupported"]

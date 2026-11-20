"""pacrunner.codegen: a program as C (pac-julia's compile.jl and compile_vm.jl): straight-line code, or
the UIR interpreter's tables (interpreter=True).

    from pacrunner import uir, codegen
    ls = uir.load_recurrence_export("export.json")
    refs = uir.recurrence_references("export.json")
    codegen.compile("out", uir.optimize(ls[0]), examples=codegen.Examples.of(ls[0], refs[0]))
    # out/: runner.c, runner.h, pacops.c/h, Makefile, testing.c (`make test`)
"""

from .common import ArenaIndex, CompileError, Examples, KernelInfo, kernel_info, lit, used_kernels
from .compile import compile
from .compile_vm import check_interpretable, compile_interpreter, gpu_bytes_per_point, vm_unsupported

__all__ = ["ArenaIndex", "CompileError", "Examples", "KernelInfo", "kernel_info", "lit", "used_kernels", "compile",
           "check_interpretable", "compile_interpreter", "gpu_bytes_per_point", "vm_unsupported"]

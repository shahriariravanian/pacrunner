"""pacrunner.kernels: the symjit kernels of a recurrence export as C. Their MIR comes from symjit's MIR
JSON (mir.py, symjit >= 2.29), the C from kernel_c.py (pac-julia's kernel_c.jl), the MSL of the Metal
runner from kernel_msl.py (kernel_msl.jl).

    from pacrunner import kernels, uir
    k = uir.export_kernels(uir.read_export("export.json"))[sha]
    c = kernels.kernel_c(k.name, k.application)          # or kernels.kernel_c_mixed
"""

from .kernel_c import KERNEL_PRELUDE, KernelCError, check_kernel, const_c, kernel_c, kernel_c_mixed, label_c
from .kernel_msl import inline_funclets, kernel_msl
from .mir import (Mir, MirError, listing, parse_mir, read_mir, require_symjit, symjit_fingerprint, symjit_problem,
                  symjit_version)

__all__ = ["KERNEL_PRELUDE", "KernelCError", "check_kernel", "const_c", "kernel_c", "kernel_c_mixed", "label_c",
           "inline_funclets", "kernel_msl",
           "Mir", "MirError", "listing", "parse_mir", "read_mir", "require_symjit", "symjit_fingerprint", "symjit_problem",
           "symjit_version"]

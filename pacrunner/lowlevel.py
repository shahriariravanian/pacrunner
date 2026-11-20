"""pacrunner's low-level API: the recurrence export, UIR, the kernels, the C and the UMAMI libraries, in
Python (until 0.3 PacRunner.jl's exports through juliacall; the names are kept where the function is).

    from pacrunner import lowlevel as pr
    programs = pr.load_recurrence_export("export.json")        # one Lowered per program entry
    pr.umami_libraries("export.json", "metadata.json", "out", device="cuda", precision="mixed")
    pr.compile("folder", pr.optimize(programs[0]), interpreter=True)

The modules are pacrunner.uir (UIR v0.8: the export reader, validate, the passes, the encoding),
pacrunner.kernels (the symjit kernels' MIR and C), pacrunner.codegen (compile) and pacrunner.umami.
Indices are 0-based (None: absent). Gone with Julia: LArena and evaluate (the Julia runner: the C is
the runner), reuse_planes (a no-op for every imported program), read_sjb, SavedApplication, MIR and
complexify (symjit's MIR JSON: kernels.read_mir).
"""

from .codegen import ArenaIndex, Examples, compile
from .kernels import kernel_c, kernel_c_mixed, listing, read_mir
from .umami import umami_libraries
from .uir import (Encoded, Lowered, decode, encode, fuse, load_recurrence_export, optimize, recurrence_references, sort_runs,
                  validate)

# the names of the low-level API (test_lowlevel checks them)
EXPORTS = (
    # RustiCol's recurrence export
    "load_recurrence_export", "recurrence_references", "umami_libraries",
    # UIR
    "Lowered", "validate", "sort_runs", "fuse", "optimize", "Encoded", "encode", "decode",
    # the C
    "compile", "ArenaIndex", "Examples",
    # the kernels
    "read_mir", "listing", "kernel_c", "kernel_c_mixed",
)
# PacRunner.jl's exports that have no Python counterpart
GONE = ("LArena", "evaluate", "reuse_planes", "read_sjb", "SavedApplication", "MIR", "complexify", "jl", "doc")


def __getattr__(name):
    if name in GONE:
        raise AttributeError(f"pacrunner.lowlevel.{name} was PacRunner.jl's (Julia), not ported: see the module's docstring")
    raise AttributeError(f"module 'pacrunner.lowlevel' has no attribute {name!r}")


__all__ = list(EXPORTS) + ["EXPORTS"]

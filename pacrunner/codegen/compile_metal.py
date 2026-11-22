"""compile_metal(path, l): one program for Apple GPUs (c/metal/), pac-julia's src/compile_metal.jl: the
UIR interpreter in the Metal Shading Language, mixed f32/df64 (c/metal/PLAN.md). The folder holds

  runner.metal     the program's whole MSL source: c/metal/df64.h, pacops.metal, uirvm.metal (their
                   text: runtime compilation resolves no #include), then the kernels (kernel_msl.py)
                   and uir_stub (a switch over the stubs, as compile_vm.py's)
  program.bin      the program as data: a uir_header (uirvm.metal) of counts and byte offsets, then
                   encode(l)'s rows and edges and the tables they index (df64 values as (hi, lo)
                   float pairs, rscales/rfactors f32)
  runner_metal.h   the program's sizes (RUNNER_*) and the entries of c/metal/uirmetal.h
  runner_metal.c   runner.metal and program.bin embedded as C arrays, for uirmetal.m
  uirmetal.h, uirmetal.m (the host: compile, launch), metalpool.h, metalpool.m (the process-wide slot
                   pool), Makefile (macOS: cc -fobjc-arc, the Metal and Foundation frameworks)

runner.metal is also C++: with c/metal/msl_shim.h before it, it compiles on any CPU, which is how
test_metal_shim.py checks it on Linux. The files are pac-julia's byte for byte.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np

from .. import _env
from ..kernels import kernel_msl
from ..kernels.kernel_c import df64_pair
from ..uir import Encoded, Lowered, encode, validate
from .common import CompileError, cstring, used_kernels
from .compile_vm import check_interpretable, firsts, halves, input_code

METAL_DIR = _env.C_DIR / "metal"
METAL_SOURCES = ("df64.h", "pacops.metal", "uirvm.metal")
METAL_HOST = ("uirmetal.h", "uirmetal.m", "metalpool.h", "metalpool.m")
GENERATED = ("runner.metal", "program.bin", "runner_metal.c", "runner_metal.h", "Makefile")

# the header's fields, in uirvm.metal's (and uirmetal.h's) order
METAL_TABLES = ("rows", "edges", "scales", "factors", "rscales", "rfactors", "form_first", "term_leg", "term_coefficient",
                "crossing_leg", "crossing_sign", "list_first", "list_factor", "embedding_first", "embedding_component",
                "embedding_wave", "embedding_factor", "scaleset_first", "scaleset_scale", "instance", "weights")
METAL_HEADER_BYTES = 128   # 10 counts + 21 offsets, 4 bytes each (124), padded to 8


def df64_pairs(xs) -> np.ndarray:
    return np.array([v for x in xs for v in df64_pair(float(x))], dtype="<f4")


def cdf64_pairs(zs) -> np.ndarray:
    return np.array([v for z in zs for part in (z.real, z.imag) for v in df64_pair(float(part))], dtype="<f4")


def _u32(xs) -> np.ndarray:
    return np.array(list(xs), dtype="<u4")


def _i32(xs) -> np.ndarray:
    return np.array(list(xs), dtype="<i4")


def metal_tables(prog: Lowered, e: Encoded) -> dict:
    """the tables of `prog` (0-based indices; an absent wavefunction component -1), in METAL_TABLES order"""
    crossed = prog.crossing is not None
    with np.errstate(over="ignore"):
        rscales = np.array([part for z in prog.scales for part in (z.real, z.imag)], dtype="<f4")
        rfactors = np.array([part for z in prog.factors for part in (z.real, z.imag)], dtype="<f4")
    return {
        "rows": _u32(halves(e.rows)),
        "edges": _u32(e.edges),
        "scales": cdf64_pairs(prog.scales),
        "factors": cdf64_pairs(prog.factors),
        "rscales": rscales,
        "rfactors": rfactors,
        "form_first": _u32(firsts(len(f.terms) for f in prog.forms)),
        "term_leg": _u32(leg for f in prog.forms for (leg, _) in f.terms),
        "term_coefficient": df64_pairs(c for f in prog.forms for (_, c) in f.terms),
        "crossing_leg": _u32(leg for (leg, _) in prog.crossing) if crossed else _u32(()),
        "crossing_sign": df64_pairs(s for (_, s) in prog.crossing) if crossed else df64_pairs(()),
        "list_first": _u32(firsts(len(x) for x in prog.lists)),
        "list_factor": _u32(x for lst in prog.lists for x in lst),
        "embedding_first": _u32(firsts(len(x) for x in prog.embeddings)),
        "embedding_component": _i32(j for lst in prog.embeddings for (j, _, _) in lst),
        "embedding_wave": _i32(-1 if w is None else w for lst in prog.embeddings for (_, w, _) in lst),
        "embedding_factor": _u32(x for lst in prog.embeddings for (_, _, x) in lst),
        "scaleset_first": _u32(firsts(len(s) for s in prog.scalesets)),
        "scaleset_scale": _u32(x for s in prog.scalesets for x in s),
        "instance": _u32(x for (s, ss) in prog.instances for x in (s, ss)),
        "weights": df64_pairs(prog.weights),
    }


def metal_program(prog: Lowered) -> bytes:
    """program.bin: the header, then each table at a multiple of 8 bytes"""
    e = encode(prog)
    check_interpretable(prog, e)
    tables = metal_tables(prog, e)
    (num, den) = prog.group_factor
    if not (-2 ** 31 <= num < 2 ** 31 and -2 ** 31 <= den < 2 ** 31):
        raise CompileError(f"{prog.process} {prog.colour}: group factor {num}/{den} beyond 32 bits")
    blob = bytearray(METAL_HEADER_BYTES)
    offsets = []
    for name in METAL_TABLES:
        blob += bytes(-len(blob) % 8)
        offsets.append(len(blob))
        blob += tables[name].tobytes()
    counts = [(0 << 16) | 8, prog.legs, len(prog.forms), prog.planes, prog.amplitudes, prog.output_count(), len(e.rows),
              1 if prog.crossing is not None else 0]
    header = struct.pack(f"<{len(counts)}I2i{len(offsets)}I", *counts, num, den, *offsets)
    blob[:len(header)] = header
    return bytes(blob)


def metal_stubs(prog: Lowered) -> str:
    """uir_stub: one case per stub, its inputs gathered into the kernel's mem (df64), the kernel, its
    output pairs stored (as compile_vm_stubs)"""
    kernels = {k.name: k for k in used_kernels(prog)}
    out = ["static int uir_stub(uint stub, thread const uir_program &g, thread const uir_arena &a, device const uint *w,",
           "                    device const uint *scaleset) {",
           "switch (stub) {"]
    for s, stub in enumerate(prog.stubs):
        k = kernels[stub.kernel.name]
        out.append(f"case {s}: {{  /* {k.name} */")
        out.append(f"df64 mem[{k.inputs + k.outputs + 1}];")
        for j, i in enumerate(stub.inputs):
            out.append(f"mem[{j}] = uir_input(g, a, w, {input_code(i)}u, scaleset);")
        out.append(f"int status = {k.name}(mem);")
        out.append("if (status != 0) return status;")
        for n, c in enumerate(stub.outputs):
            out.append(f"uir_output(g, a, w, {c}u, cdd(mem[{k.inputs + 2 * n}], mem[{k.inputs + 2 * n + 1}]));")
        out.append("return 0;")
        out.append("}")
    out += ["default: return -1;", "}", "}"]
    return "".join(line + "\n" for line in out)


def _source_file(name: str) -> str:
    return (METAL_DIR / name).read_bytes().decode("utf-8")


def metal_source(prog: Lowered) -> str:
    """runner.metal: the whole source of the program"""
    out = [f"// {prog.process} {prog.colour}: the program's Metal source, by PacRunner's compile_metal\n",
           "#ifndef MSL_SHIM_H\n",
           "#include <metal_stdlib>\n",
           "using namespace metal;\n",
           # no fma contraction (MTLMathModeSafe alone still contracts a * b + c): the C sources and the
           # shim are compiled with -ffp-contract=off, and the GPU gives their bits only with this
           "#pragma METAL fp contract(off)\n",
           "#endif\n"]
    for f in METAL_SOURCES:
        out.append(f"\n// ---- c/metal/{f} ----\n")
        out.append(_source_file(f))
    out.append("\n// ---- the program's kernels (kernel_msl.jl) and stubs ----\n")
    for k in used_kernels(prog):
        out.append(kernel_msl(k.name, k.mir) + "\n")
    out.append(metal_stubs(prog))
    return "".join(out)


def c_bytes(data: bytes) -> str:
    """bytes as a C array body, 16 a line"""
    n = len(data)
    parts = [f"0x{b:02x}" + ("" if k == n else ",") + ("\n" if k % 16 == 0 else "") for k, b in enumerate(data, 1)]
    return "".join(parts) + ("0" if n == 0 else "") + "\n"


def metal_runner_c(prog: Lowered, source: str, program: bytes) -> str:
    """runner_metal.c: the source and the program for uirmetal.m"""
    src = source.encode("utf-8")
    return (f"/* {prog.process} {prog.colour}: its Metal source and program, by PacRunner's compile_metal */\n"
            "#include <stddef.h>\n"
            "#include \"runner_metal.h\"\n"
            "const char uir_metal_source[] = {\n"
            + c_bytes(src) +
            ", 0};\n"
            f"const size_t uir_metal_source_size = {len(src)};\n"
            "const unsigned char uir_metal_program[] __attribute__((aligned(16))) = {\n"
            + c_bytes(program) +
            "};\n"
            f"const size_t uir_metal_program_size = {len(program)};\n")


def metal_runner_h(prog: Lowered) -> str:
    return f"""\
/* {prog.process} {prog.colour}: the program for Metal, by PacRunner's compile_metal
 * (runner_metal.c: its source and program; uirmetal.h: the entries). */
#ifndef RUNNER_METAL_H
#define RUNNER_METAL_H

#define RUNNER_PROCESS {cstring(prog.process)}
#define RUNNER_COLOUR {cstring(prog.colour)}
#define RUNNER_LEGS {prog.legs}
#define RUNNER_OUTPUTS {prog.output_count()}
#define RUNNER_FORMS {len(prog.forms)}
#define RUNNER_AMPLITUDES {prog.amplitudes}
#define RUNNER_PLANES {prog.planes}

#include <stddef.h>
extern const char uir_metal_source[];
extern const size_t uir_metal_source_size;
extern const unsigned char uir_metal_program[];
extern const size_t uir_metal_program_size;

#include "uirmetal.h"

#endif
"""


def metal_makefile(prog: Lowered) -> str:
    return f"""\
# {prog.process} {prog.colour} for Metal, compiled by PacRunner's compile_metal (macOS).
#   make         librunner_metal.dylib: run_metal, pac_metal_run, pac_metal_values (uirmetal.h)
#   make clean
CC ?= cc
CFLAGS ?= -O2
PACFLAGS = -ffp-contract=off -fPIC
FRAMEWORKS = -framework Metal -framework Foundation
all: librunner_metal.dylib

librunner_metal.dylib: uirmetal.m uirmetal.h metalpool.m metalpool.h runner_metal.c runner_metal.h
\t$(CC) $(CFLAGS) $(PACFLAGS) -fobjc-arc -dynamiclib -o $@ uirmetal.m metalpool.m runner_metal.c $(FRAMEWORKS)

clean:
\trm -f librunner_metal.dylib

.PHONY: all clean
"""


def compile_metal(path, prog: Lowered):
    """the folder of `prog` for Metal (see the module); returns `path`"""
    validate(prog)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    source = metal_source(prog)
    program = metal_program(prog)
    (path / "runner.metal").write_bytes(source.encode("utf-8"))
    (path / "program.bin").write_bytes(program)
    (path / "runner_metal.h").write_bytes(metal_runner_h(prog).encode("utf-8"))
    (path / "runner_metal.c").write_bytes(metal_runner_c(prog, source, program).encode("utf-8"))
    for f in METAL_HOST:
        shutil.copyfile(METAL_DIR / f, path / f)
    (path / "Makefile").write_bytes(metal_makefile(prog).encode("utf-8"))
    return path

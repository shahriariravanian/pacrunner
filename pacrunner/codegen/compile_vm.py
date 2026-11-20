"""compile(path, l, interpreter=True) (pac-julia's src/compile_vm.jl): the folder of compile.py, but
runner.c runs the program on the UIR interpreter (c/uirvm.c, which runner.c includes) instead of
straight-line code. The program is data there: encode(l)'s words (the rows, SOURCE rows among them)
and edge table, and the tables they index, as C arrays of one uir_program (c/uirvm.h); the kernels
are reached through uir_stub, a switch over the stubs (each a kernel and its input layout, the
kernels' C as compile.py's). runner.h, testing.c and the Makefile are compile.py's, so run() has the
same prototype and the values must be the straight-line C's bit for bit.

The CPU build is the interpreter's test; the same uirvm.c is the GPU's interpreter.
"""

from __future__ import annotations

from itertools import accumulate
from typing import Optional

from ..uir import Encoded, InputKind, KernelInput, Lowered, Opcode, encode, validate
from .common import (VM_SOURCES, ArenaIndex, CodeWriter, CompileError, Examples, check_examples, emit_kernels, lit,
                     used_kernels, write_beside)

GPU_MEMORY_DEFAULT = 256 * 2 ** 20   # device bytes per slot of the GPU pool (pool.h)

# the most components uirvm's closure and scale rows take (UIR_MAX_COMPONENTS: each count 0-6 has its
# constant-count code); a kernel-only (UNTYPED) current may have more
VM_MAX_COMPONENTS = 6


def carray(w: CodeWriter, ty: str, name: str, values, per_line: int = 8) -> None:
    """a C array: `UIR_DATA <ty> <name>[] = {...};` (static const, or __device__ const under nvcc; an
    empty one holds a single unused 0: C has no empty arrays)"""
    v = [str(x) for x in values]
    if not v:
        w.emit(f"UIR_DATA {ty} {name}[1] = {{0}};")
        return
    w.emit(f"UIR_DATA {ty} {name}[] = {{")
    for k in range(0, len(v), per_line):
        w.emit(", ".join(v[k:k + per_line]) + ",")
    w.emit("};")


def hex32(x: int) -> str:
    return f"0x{x:08x}u"


def halves(words) -> list[int]:
    """the words as uirvm reads them: each as two 32-bit halves, low first"""
    return [(x >> s) & 0xFFFFFFFF for x in words for s in (0, 32)]


def cdoubles(zs) -> list[str]:
    return [lit(f) for z in zs for f in (z.real, z.imag)]


def firsts(lengths) -> list[int]:
    """a "first" table: the offsets of each entry's items, then their total"""
    return list(accumulate(lengths, initial=0))


def input_code(i: KernelInput) -> int:
    """a kernel input as uirvm's code: kind, imaginary part, n (component or scale-set slot)"""
    return int(i.kind) | int(i.im) << 3 | (0 if i.kind == InputKind.FACTOR else i.n) << 4


def gpu_bytes_per_point(prog: Lowered, precision: str = "f64") -> int:
    """the device bytes per point of a program's arena on the GPU (uirgpu.c: momenta, outputs, forms,
    amplitudes, planes; precision "mixed": planes in f32)"""
    return 8 * sum(max(n, 1) for n in (4 * prog.legs, prog.output_count(), 4 * len(prog.forms), 2 * prog.amplitudes)) + \
        (4 if precision == "mixed" else 8) * max(2 * prog.planes, 1)


def run_gpu_c(prog: Lowered) -> str:
    """the GPU part of runner.c (under nvcc): the sizes uirgpu.c needs, then uirgpu.c (the program on a
    slot of the pool, pool.h: plane-major arena, ld = points). The pool's size (UIRGPU_SLOTS,
    UIRGPU_SLOT_BYTES) defaults to one slot per CPU of GPU_MEMORY_DEFAULT bytes; nvcc -D overrides."""
    return f"""\
#define UIRGPU_LEGS {prog.legs}
#define UIRGPU_OUTPUTS {prog.output_count()}
#define UIRGPU_FORMS {len(prog.forms)}
#define UIRGPU_AMPLITUDES {prog.amplitudes}
#define UIRGPU_PLANES {prog.planes}
#ifndef UIRGPU_SLOT_BYTES
#define UIRGPU_SLOT_BYTES ((size_t){GPU_MEMORY_DEFAULT})  /* {gpu_bytes_per_point(prog)} bytes per point ({gpu_bytes_per_point(prog, "mixed")} mixed) */
#endif
#include "uirgpu.c"
"""


def vm_unsupported(prog: Lowered) -> list[str]:
    """what uirvm cannot run in `prog`: closure, scale, copy and scatter rows beyond VM_MAX_COMPONENTS"""
    out = []
    for k, op in enumerate(prog.ops, 1):
        if op.op == Opcode.CLOSURE:
            n = len(prog.lists[op.t])
        elif op.op == Opcode.SCALE_ROW:
            n = op.n
        elif op.op in (Opcode.COPY, Opcode.SCATTER):
            n = prog.edge_layout(op.d).components
        else:
            n = 0
        if n > VM_MAX_COMPONENTS:
            out.append(f"row {k} ({op.op.name}): {n} components, beyond the interpreter's {VM_MAX_COMPONENTS}")
    return out


def check_interpretable(prog: Lowered, e: Encoded) -> None:
    """what the interpreter cannot run: a wide row (uirvm reads one-word rows only) or too many components"""
    if len(e.rows) != len(prog.ops) + 1:
        raise CompileError(f"{prog.process} {prog.colour}: {len(e.rows) - len(prog.ops) - 1} wide rows (an index beyond the "
                           "one-word fields: 13-bit edges, 10-bit T, 6-bit factors); the UIR interpreter reads one-word rows only")
    bad = vm_unsupported(prog)
    if bad:
        raise CompileError(f"{prog.process} {prog.colour}: " + "; ".join(bad))


def compile_vm_head(w: CodeWriter, prog: Lowered) -> None:
    """the interpreter's sources beside runner.c, and runner.c's head"""
    w.copy_home(*VM_SOURCES)
    w.emit(f"/* {prog.process} {prog.colour} on the UIR interpreter (uirvm.c), compiled by PacRunner's compile */")
    w.emit("#include <stdlib.h>")
    w.emit('#include "uirvm.c"')
    w.emit("#ifdef __CUDACC__")
    w.emit('#include "pacops.c"  /* one translation unit for the device */')
    w.emit("#endif")
    w.emit('#include "runner.h"')
    w.newline()


def compile_vm_tables(w: CodeWriter, prog: Lowered, e: Encoded) -> None:
    """the program's tables as C arrays: the encoded rows and edges, then the tables they index (each
    list table as its entries' firsts and its flattened entries)"""
    carray(w, "uint32_t", "rows", [hex32(x) for x in halves(e.rows)], per_line=8)
    carray(w, "uint32_t", "edges", e.edges)
    carray(w, "double", "scales", cdoubles(prog.scales), per_line=4)
    carray(w, "double", "factors", cdoubles(prog.factors), per_line=4)
    carray(w, "uint32_t", "form_first", firsts(len(f.terms) for f in prog.forms))
    carray(w, "uint32_t", "term_leg", [leg for f in prog.forms for (leg, _) in f.terms])
    carray(w, "double", "term_coefficient", [lit(c) for f in prog.forms for (_, c) in f.terms], per_line=4)
    if prog.crossing is not None:
        carray(w, "uint32_t", "crossing_leg", [leg for (leg, _) in prog.crossing])
        carray(w, "double", "crossing_sign", [lit(s) for (_, s) in prog.crossing], per_line=4)
    carray(w, "uint32_t", "list_first", firsts(len(x) for x in prog.lists))
    carray(w, "uint32_t", "list_factor", [x for entries in prog.lists for x in entries])
    carray(w, "uint32_t", "embedding_first", firsts(len(x) for x in prog.embeddings))
    carray(w, "int32_t", "embedding_component", [j for entries in prog.embeddings for (j, _, _) in entries])
    carray(w, "int32_t", "embedding_wave", [-1 if v is None else v for entries in prog.embeddings for (_, v, _) in entries])
    carray(w, "uint32_t", "embedding_factor", [x for entries in prog.embeddings for (_, _, x) in entries])
    carray(w, "uint32_t", "scaleset_first", firsts(len(s) for s in prog.scalesets))
    carray(w, "uint32_t", "scaleset_scale", [x for s in prog.scalesets for x in s])
    carray(w, "uint32_t", "instance", [x for pair in prog.instances for x in pair])
    carray(w, "double", "weights", [lit(x) for x in prog.weights], per_line=4)
    # the scales and factors as pac_real: rounded copies under PAC_MIXED (pacops.h), else the same tables
    w.emit("#ifdef PAC_MIXED")
    carray(w, "pac_real", "scales_r", cdoubles(prog.scales), per_line=4)
    carray(w, "pac_real", "factors_r", cdoubles(prog.factors), per_line=4)
    w.emit("#else")
    w.emit("#define scales_r scales")
    w.emit("#define factors_r factors")
    w.emit("#endif")
    w.newline()


def compile_vm_program(w: CodeWriter, prog: Lowered, e: Encoded) -> None:
    """the uir_program (uirvm.h) of the tables"""
    crossed = prog.crossing is not None
    (num, den) = prog.group_factor
    w.emit("UIR_DATA uir_program program = {")
    w.emit(f"UIR_VERSION, {prog.legs}, {len(prog.forms)}, {prog.planes}, {prog.amplitudes}, {prog.output_count()},")
    w.emit(f"rows, {len(e.rows)}, edges, scales, factors,")
    w.emit(f"form_first, term_leg, term_coefficient, {'crossing_leg' if crossed else 'NULL'}, "
           f"{'crossing_sign' if crossed else 'NULL'},")
    w.emit("list_first, list_factor,")
    w.emit("embedding_first, embedding_component, embedding_wave, embedding_factor,")
    w.emit("scaleset_first, scaleset_scale, instance,")
    w.emit(f"weights, {num}L, {den}L,")
    w.emit("scales_r, factors_r")
    w.emit("};")
    w.newline()


def compile_vm_stubs(w: CodeWriter, prog: Lowered) -> None:
    """the kernels and uir_stub, one case per stub: its inputs gathered into the kernel's mem (constant
    codes), the kernel, its output pairs stored (constant components)"""
    kernels = {k.name: k for k in used_kernels(prog)}
    emit_kernels(w, list(kernels.values()), mixed=True)
    w.emit("PAC_FN int uir_stub(uint32_t stub, const uir_program *g, const uir_arena *a, const uint32_t *w, "
           "const uint32_t *scaleset) {")
    w.emit("switch (stub) {")
    for s, stub in enumerate(prog.stubs):
        k = kernels[stub.kernel.name]
        w.emit(f"case {s}: {{  /* {k.name} */")
        w.emit(f"double mem[{k.inputs + k.outputs + 1}];")
        for j, i in enumerate(stub.inputs):
            w.emit(f"mem[{j}] = uir_input(g, a, w, {input_code(i)}u, scaleset);")
        w.emit(f"int status = {k.name}(mem);")
        w.emit("if (status != 0) return status;")
        for n, c in enumerate(stub.outputs):
            w.emit(f"uir_output(g, a, w, {c}, cx(mem[{k.inputs + 2 * n}], mem[{k.inputs + 2 * n + 1}]));")
        w.emit("return 0;")
        w.emit("}")
    w.emit("default: return -1;")
    w.emit("}")
    w.emit("}")
    w.newline()


def compile_vm_run(w: CodeWriter, prog: Lowered, b: ArenaIndex) -> None:
    """run(): one point on the CPU (the arena of compile.py's layout, ld = 1); under nvcc the GPU entries"""
    w.emit("#ifndef __CUDACC__")
    w.emit("int run(double *out, double *p, const double *momenta) {")
    w.emit(f"for (int i = 0; i < {b.size}; i++) p[i] = 0.0;")
    w.emit(f"uir_arena a = {{p + {b.forms}, p + {b.amplitudes}, p + {b.planes}, 1, 0}};")
    w.emit("return uir_run(&program, &a, momenta, out);  /* 0, or a failing kernel's status */")
    w.emit("}")
    w.emit("#else")
    w.emit_lines(run_gpu_c(prog))
    w.emit("#endif")


def compile_interpreter(path, prog: Lowered, examples: Optional[Examples] = None):
    validate(prog)
    b = ArenaIndex.of(prog)
    if examples is not None:
        check_examples(prog, examples)
    e = encode(prog)
    check_interpretable(prog, e)
    w = CodeWriter(path)
    try:
        compile_vm_head(w, prog)
        compile_vm_tables(w, prog, e)
        compile_vm_program(w, prog, e)
        compile_vm_stubs(w, prog)
        compile_vm_run(w, prog, b)
    finally:
        w.close()
    write_beside(path, prog, b, examples, interpreter=True)
    return path

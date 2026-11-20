"""compile(path, l): a program as C (pac-julia's src/compile.jl). The folder gets runner.c (`int
run(double *out, double *p, const double *momenta)`), runner.h (its sizes and prototype), pacops.c and
pacops.h, and a Makefile (`make`: runner.o and pacops.o). With `examples` (common.Examples: the
program's reference points), also testing.c, which runs those points and compares with RustiCol's
outputs (`make test`). A Lowered is compiled as it is (uir.optimize it first for fused runs). With
`interpreter`, runner.c runs the program on the UIR interpreter (compile_vm.py, c/uirvm.c) instead of
straight-line code, with the same runner.h and values. The kernels are C in runner.c
(kernels/kernel_c.py): the folder needs no kernel library.

A fused run (FIRST ... LAST) is written as its rows one by one, each storing with its own mode: by
the run invariants (validate) no row of a run reads its destination and the sums keep their order,
so the values are the run's, and the C compiler may keep the destination in registers.

Indices are 0-based in the Lowered as in the C: written as they are.
"""

from __future__ import annotations

from typing import Optional

from ..uir import (InputKind, KernelInput, LOp, Lowered, Opcode, PhysicalType, describe, is_kernel, is_propagator, is_vertex,
                   source_fields, validate)
from .common import (ArenaIndex, CodeWriter, CompileError, Examples, check_examples, emit_kernels, lit, used_kernels,
                     write_beside)

# pacops.h's names of the opcodes and source families
C_VERTEX = {Opcode.V_WEYL_VECTOR_A: "VERTEX_WEYL_VECTOR_A", Opcode.V_WEYL_VECTOR_B: "VERTEX_WEYL_VECTOR_B",
            Opcode.V_TENSOR_VECTOR: "VERTEX_TENSOR_VECTOR", Opcode.V_VECTOR_WEDGE: "VERTEX_VECTOR_WEDGE",
            Opcode.V_THREE_VECTOR: "VERTEX_THREE_VECTOR"}
C_PROPAGATOR = {Opcode.P_WEYL_A: "PROPAGATOR_WEYL_A", Opcode.P_WEYL_B: "PROPAGATOR_WEYL_B", Opcode.P_VECTOR: "PROPAGATOR_VECTOR"}
C_FAMILY = {PhysicalType.SCALAR: "FAMILY_SCALAR", PhysicalType.WEYL: "FAMILY_WEYL", PhysicalType.DIRAC: "FAMILY_DIRAC",
            PhysicalType.VECTOR: "FAMILY_VECTOR"}


def c_mode(op: LOp) -> str:
    return "MODE_SET" if op.set else "MODE_ADD"


def scale(i: int) -> str:
    return f"scales[{2 * i}]"


def factor(i: int) -> str:
    return f"factors[{2 * i}]"


def compile(path, prog: Lowered, examples: Optional[Examples] = None, interpreter: bool = False):
    """the folder `path` for program `prog` (see the module); returns `path`"""
    if interpreter:
        from .compile_vm import compile_interpreter
        return compile_interpreter(path, prog, examples=examples)
    validate(prog)
    b = ArenaIndex.of(prog)
    if examples is not None:
        check_examples(prog, examples)
    w = CodeWriter(path)
    try:
        w.emit("#include <stdlib.h>")
        w.emit('#include "pacops.h"')
        w.emit('#include "runner.h"')
        w.newline()

        compile_scales(w, prog)
        compile_factors(w, prog)
        compile_kernels(w, prog)
        compile_forms(w, prog, b)
        compile_tables(w, prog)
        compile_body(w, prog, b)

        # the arena and the outputs at zero, the forms, the rows (REDUCE rows add to out), the group factor
        (num, den) = prog.group_factor
        w.emit("int run(double *out, double *p, const double *momenta) {")
        w.emit(f"for(int i=0; i<{b.size}; i++) p[i] = 0.0;")
        w.emit(f"for(int h=0; h<{prog.output_count()}; h++) out[h] = 0.0;")
        w.emit("forms(p, momenta);")
        w.emit("body(out, p);")
        if num != den:
            w.emit(f"for(int h=0; h<{prog.output_count()}; h++) out[h] = group(out[h], {num}L, {den}L);")
        w.emit("return 0;")
        w.emit("}")
    finally:
        w.close()
    write_beside(path, prog, b, examples, interpreter=False)
    return path


def _complex_lines(w: CodeWriter, name: str, values) -> None:
    w.emit(f"const double {name}[] = {{")
    for c in values:
        w.emit(f"{lit(c.real)}, {lit(c.imag)},")
    w.emit("};")
    w.newline()


def compile_scales(w: CodeWriter, prog: Lowered) -> None:
    _complex_lines(w, "scales", prog.scales)


def compile_factors(w: CodeWriter, prog: Lowered) -> None:
    _complex_lines(w, "factors", prog.factors)


def _pair(prog: Lowered, x: int) -> str:
    z = prog.factors[x]
    return f"{lit(z.real)}, {lit(z.imag)}"


def compile_tables(w: CodeWriter, prog: Lowered) -> None:
    """the constant arrays of the rows, at file scope (C++ and CUDA have no compound literals): the
    component factors of closure list t, `cf<t>`; the output components of stub s, `outputs<s>`; the
    embeddings. Their names count from 1, as pac-julia's."""
    for t, entries in enumerate(prog.lists, 1):
        if not entries:
            continue
        w.emit(f"static const double cf{t}[] = {{{', '.join(_pair(prog, x) for x in entries)}}};")
    for s, stub in enumerate(prog.stubs, 1):
        w.emit(f"static const int outputs{s}[] = {{{', '.join(str(j) for j in stub.outputs)}}};")
    # embedding k: the current's components, the wavefunction's (-1: zero), the factors
    for k, entries in enumerate(prog.embeddings, 1):
        w.emit(f"static const int emb_components{k}[] = {{{', '.join(str(j) for (j, _, _) in entries)}}};")
        w.emit(f"static const int emb_waves{k}[] = {{{', '.join(str(-1 if v is None else v) for (_, v, _) in entries)}}};")
        w.emit(f"static const double emb_factors{k}[] = {{{', '.join(_pair(prog, x) for (_, _, x) in entries)}}};")
    w.newline()


def compile_kernels(w: CodeWriter, prog: Lowered) -> None:
    """one adapter per kernel: its inputs as arguments, then its output pairs into `out` (as they are
    for KERNEL_DIRECT, else × factor with the mode)"""
    kernels = used_kernels(prog)
    emit_kernels(w, kernels)
    for k in kernels:
        name, inputs, outputs = k.name, k.inputs, k.outputs
        s = f"void adapter_{name}(double *out, " + "".join(f"double x{i}, " for i in range(inputs))
        w.emit(f"{s} const int *outputs, int direct, int mode, const double *factor) {{")
        w.emit(f"double mem[{inputs + outputs + 1}];")
        for i in range(inputs):
            w.emit(f"mem[{i}] = x{i};")
        w.emit(f"if ({name}(mem) != 0) abort();  /* the kernel failed */")
        w.emit("cplx f = get(factor, 0);")
        w.emit(f"for (int n = 0; n < {outputs // 2}; n++) {{")
        w.emit(f"cplx pair = get(mem + {inputs}, n);")
        w.emit("put(out, outputs[n], combine(get(out, outputs[n]), direct ? pair : mul(pair, f), mode));")
        w.emit("}")
        w.emit("}")
        w.newline()


def compile_forms(w: CodeWriter, prog: Lowered, b: ArenaIndex) -> None:
    w.emit("void forms(double *p, const double *momenta) {")

    # the program's leg l: the caller's leg times its sign, or the caller's leg l (q = 0.0 + Σ coef ×
    # (sign × p[leg]), in term order)
    def external(l: int, j: int) -> str:
        if prog.crossing is None:
            return f"momenta[{4 * l + j}]"
        (leg, sign) = prog.crossing[l]
        return f"({lit(sign)} * momenta[{4 * leg + j}])"

    for i, f in enumerate(prog.forms):
        q = b.form_at(i)
        for j in range(4):
            s = f"p[{q + j}] = 0.0" + "".join(f" + {lit(coef)} * {external(leg, j)}" for (leg, coef) in f.terms) + ";"
            w.emit(s)
    w.emit("}")
    w.newline()


def compile_body(w: CodeWriter, prog: Lowered, b: ArenaIndex) -> None:
    w.emit("void body(double *out, double *p) {")
    for k, op in enumerate(prog.ops, 1):
        w.emit(f"// {k}: {describe(prog, op)}")
        compile_op(w, op, prog, b)
    w.emit("}")
    w.newline()


def compile_op(w: CodeWriter, op: LOp, prog: Lowered, b: ArenaIndex) -> None:
    def p(e):
        return f"&p[{b.edge_at(prog, e)}]"

    def momentum(e):
        return f"&p[{b.form_at(prog.edge_form(e))}]"

    o = op.op
    if is_vertex(o):
        (u, v) = (momentum(op.l), momentum(op.r)) if o == Opcode.V_THREE_VECTOR else ("NULL", "NULL")
        w.emit(f"vertex({p(op.d)}, {C_VERTEX[o]}, {c_mode(op)}, {p(op.l)}, {p(op.r)}, {u}, {v}, "
               f"&{scale(op.t)}, &{factor(op.x)});")
    elif is_propagator(o):
        w.emit(f"propagator({p(op.d)}, {C_PROPAGATOR[o]}, {p(op.d)}, {momentum(op.d)}, &{factor(op.x)});")
    elif o == Opcode.SCALE_ROW:
        w.emit(f"scale({p(op.d)}, {op.n}, &{factor(op.x)});")
    elif o == Opcode.CLEAR:
        w.emit(f"clear({p(op.d)}, {op.n});")
    elif o == Opcode.SOURCE:
        (family, anti, h, ch) = source_fields(op)
        ori = "ANTIPARTICLE" if anti else "PARTICLE"
        mass = lit(prog.scales[op.t].real)
        if op.l is None:
            w.emit(f"source({p(op.d)}, {C_FAMILY[family]}, {ori}, {momentum(op.d)}, {h}, {ch}, {mass}, &{factor(op.x)});")
        else:
            k = op.l + 1
            w.emit(f"source_embedded({p(op.d)}, {C_FAMILY[family]}, {ori}, {momentum(op.d)}, {h}, {ch}, {mass}, "
                   f"emb_components{k}, emb_waves{k}, emb_factors{k}, {len(prog.embeddings[op.l])});")
    elif o == Opcode.CLEAR_AMPLITUDE:
        a = b.amplitude_at(op.d)
        w.emit(f"p[{a}] = 0.0; p[{a + 1}] = 0.0;")
    elif o == Opcode.COPY:
        w.emit(f"copy_row({p(op.d)}, {p(op.l)}, {prog.edge_layout(op.d).components}, &{factor(op.x)});")
    elif o == Opcode.SCATTER:
        w.emit(f"scatter({p(op.d)}, {p(op.l)}, {prog.edge_layout(op.d).components}, {c_mode(op)}, &{factor(op.x)});")
    elif o == Opcode.REDUCE:
        sc = "NULL" if op.x is None else f"&{factor(op.x)}"
        w.emit(f"out[{op.d}] = reduce_term(out[{op.d}], &p[{b.amplitude_at(op.t)}], {sc}, {lit(prog.weights[op.n])});")
    elif o == Opcode.CLOSURE:
        entries = prog.lists[op.t]
        cf = "NULL" if not entries else f"cf{op.t + 1}"
        w.emit(f"closure(&p[{b.amplitude_at(op.d)}], {p(op.l)}, {p(op.r)}, {cf}, {len(entries)}, &{factor(op.x)});")
    elif is_kernel(o):
        (s, set_) = prog.instances[op.t]
        stub = prog.stubs[s]
        args = [p(op.d), *(kernel_arg(i, op, prog, b, prog.scalesets[set_]) for i in stub.inputs), f"outputs{s + 1}",
                "1" if o == Opcode.KERNEL_DIRECT else "0", c_mode(op), f"&{factor(op.x)}"]   # a stub may have no inputs
        w.emit(f"adapter_{stub.kernel.name}({', '.join(args)});")
    else:
        raise CompileError(f"unknown opcode {o.name}")


def kernel_arg(i: KernelInput, op: LOp, prog: Lowered, b: ArenaIndex, scaleset) -> str:
    """a kernel input as a C expression: a part of an operand's component, a momentum component, the
    real part of a scale (by its slot in the scale set), a part of the row's factor"""
    im = 1 if i.im else 0
    if i.kind == InputKind.IN_L:
        return f"p[{b.edge_at(prog, op.l) + 2 * i.n + im}]"
    if i.kind == InputKind.IN_R:
        return f"p[{b.edge_at(prog, op.r) + 2 * i.n + im}]"
    if i.kind == InputKind.OUT:
        return f"p[{b.edge_at(prog, op.d) + 2 * i.n + im}]"
    if i.kind == InputKind.MOMENTUM_U:
        return f"p[{b.form_at(prog.edge_form(op.l)) + i.n}]"
    if i.kind == InputKind.MOMENTUM_V:
        return f"p[{b.form_at(prog.edge_form(op.r)) + i.n}]"
    if i.kind == InputKind.SCALE:
        return scale(scaleset[i.n])
    return f"factors[{2 * op.x + im}]"   # FACTOR

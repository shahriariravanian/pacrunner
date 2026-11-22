"""A symjit kernel as C (pac-julia's src/kernel_c.jl): kernel_c(name, mir) writes the MIR of a saved
application, lowered as the native generators run it (mir.py: symjit's json-complex, ComplexBi
kept), as one C function, `static int name(double *mem)`, so the compiled runners need no library
built by symjit-utils' sjb2obj. The kernel is the direct-arena form RustiCol saves: its inputs at
mem[0 .. states), its outputs (re, im of each output pair) at mem[states .. states + observables), no
parameters.

The operations are those of the native code (sjb2obj's x86-64 generator), which is the reference
the C must match bit for bit: Fused is one fma; Sign is the sign bit (x & -0.0); Half is x × 0.5;
comparisons and IsZero give all ones or zero; And, Or, Xor, AndNot, Not act on the bits. A complex
× or ÷ (ComplexBi) is the generator's sequence: fused, through Gen(2), Gen(3) (and Temp for ÷); +
and − are two real operations. LoadMath and LoadConstMath of a complex model go through Temp.
Registers follow symjit's: Ret and Left are one register (r0), Temp and Right another (r1), Gen(k) is
gk, and stack slot k the local stackk. A funclet (a Call with no arguments, its label in the code,
".ret" to return) is a goto with the return point kept in `ret`, so the C has no calls and no
computed goto (nvcc). Every function is PAC_FN: under nvcc a device function too.

The C is pac-julia's byte for byte (its comments too, which name kernel_c.jl).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

from .._literal import julia_repr, julia_repr_f32
from .mir import (Bi, Branch, BranchIf, Call, ComplexBi, End, Fused, IfElse, Instruction, Label, Load, LoadComplex, LoadConst,
                  LoadConstMath, LoadMath, Loc, Mir, Mov, Nop, Reg, Save, SaveComplex, Uni, read_mir)


class KernelCError(ValueError):
    def __init__(self, msg: str):
        super().__init__(f"kernel_c: {msg}")


def refuse(msg: str):
    raise KernelCError(msg)


# the helpers the kernels use, once per C file before them
KERNEL_PRELUDE = """\
#include <math.h>
#include <stdint.h>
#include <string.h>

/* a function of the host and, under nvcc, of the device (as pacops.h's) */
#ifndef PAC_FN
#ifdef __CUDACC__
#define PAC_FN __host__ __device__
#else
#define PAC_FN
#endif
#endif

#ifdef __CUDACC__
#pragma nv_diag_suppress 177  /* a helper a program's kernels do not use */
#endif

/* the symjit kernels' bit operations on doubles (kernel_c.jl); a program's kernels use some of
   them, so each is marked unused (clang's -Wunused-function warns of an unused static inline) */
#define SJ_FN PAC_FN static inline __attribute__((unused))
SJ_FN uint64_t sj_bits(double x) { uint64_t u; memcpy(&u, &x, sizeof u); return u; }
SJ_FN double sj_double(uint64_t u) { double x; memcpy(&x, &u, sizeof x); return x; }
SJ_FN double sj_mask(int b) { return sj_double(b ? ~(uint64_t)0 : 0); }
SJ_FN double sj_and(double a, double b) { return sj_double(sj_bits(a) & sj_bits(b)); }
SJ_FN double sj_andnot(double a, double b) { return sj_double(~sj_bits(a) & sj_bits(b)); }
SJ_FN double sj_or(double a, double b) { return sj_double(sj_bits(a) | sj_bits(b)); }
SJ_FN double sj_xor(double a, double b) { return sj_double(sj_bits(a) ^ sj_bits(b)); }
SJ_FN double sj_not(double a) { return sj_double(~sj_bits(a)); }
SJ_FN double sj_sign(double a) { return sj_double(sj_bits(a) & ((uint64_t)1 << 63)); }
"""


def reg_c(r: Reg) -> str:
    if r.kind in ("Ret", "Left"):
        return "r0"
    if r.kind in ("Temp", "Right"):
        return "r1"
    if r.kind == "Gen":
        return f"g{r.index}"
    refuse(f"a static register {r!r}")


def loc_c(l: Loc, offset: int = 0) -> str:
    """a location as C: mem[k], or the local stack<k> (symjit's stack slots are fixed indices, the low
    ones mostly unused, so each slot a kernel names is a variable of its own)"""
    if l.kind == "Mem":
        return f"mem[{l.index + offset}]"
    if l.kind == "Stack":
        return f"stack{l.index + offset}"
    refuse(f"a parameter location {l!r} (the kernel must take no parameters)")


UNI_C: dict[str, Callable[[str], str]] = {
    "Neg": lambda x: f"-{x}", "Not": lambda x: f"sj_not({x})", "Abs": lambda x: f"fabs({x})", "Abs2": lambda x: f"{x} * {x}",
    "Root": lambda x: f"sqrt({x})", "RealRoot": lambda x: f"sqrt({x})", "Recip": lambda x: f"1.0 / {x}",
    "Round": lambda x: f"nearbyint({x})",  # vroundsd mode 0: to nearest, ties to even
    "Floor": lambda x: f"floor({x})", "Ceiling": lambda x: f"ceil({x})", "Trunc": lambda x: f"trunc({x})",
    "Real": lambda x: x, "Conjugate": lambda x: x, "Imaginary": lambda _: "0.0", "Half": lambda x: f"{x} * 0.5",
    "IsZero": lambda x: f"sj_mask({x} == 0.0)", "IsNotZero": lambda x: f"sj_mask({x} != 0.0)", "Sign": lambda x: f"sj_sign({x})",
}

BI_C: dict[str, Callable[[str, str], str]] = {
    "Plus": lambda a, b: f"{a} + {b}", "Minus": lambda a, b: f"{a} - {b}", "Times": lambda a, b: f"{a} * {b}",
    "Divide": lambda a, b: f"{a} / {b}",
    "GreaterThan": lambda a, b: f"sj_mask({a} > {b})", "GreaterThanEqual": lambda a, b: f"sj_mask({a} >= {b})",
    "LittleThan": lambda a, b: f"sj_mask({a} < {b})", "LittleThanEqual": lambda a, b: f"sj_mask({a} <= {b})",
    "Equal": lambda a, b: f"sj_mask({a} == {b})", "NotEqual": lambda a, b: f"sj_mask({a} != {b})",
    "And": lambda a, b: f"sj_and({a}, {b})", "AndNot": lambda a, b: f"sj_andnot({a}, {b})",
    "Or": lambda a, b: f"sj_or({a}, {b})", "Xor": lambda a, b: f"sj_xor({a}, {b})",
}

FUSED_C: dict[str, Callable[[str, str, str], str]] = {
    "MulAdd": lambda a, b, c: f"fma({a}, {b}, {c})", "MulSub": lambda a, b, c: f"fma({a}, {b}, -{c})",
    "NegMulAdd": lambda a, b, c: f"fma(-{a}, {b}, {c})", "NegMulSub": lambda a, b, c: f"fma(-{a}, {b}, -{c})",
}


def label_c(s: str) -> str:
    """a label as a C identifier (labels are symjit's strings: "@complex_root", ...)"""
    return "L_" + re.sub(r"[^A-Za-z0-9_]", lambda m: f"_{ord(m.group(0)):x}_", s)


def const_c(x: float) -> str:
    """a double as a C literal that reads back exactly"""
    if x != x:
        import struct
        return f"sj_double(0x{struct.unpack('<Q', struct.pack('<d', x))[0]:x}ull)"
    if x in (float("inf"), float("-inf")):
        return "HUGE_VAL" if x > 0 else "-HUGE_VAL"
    return julia_repr(x)


def _labels(mir: Mir) -> set[str]:
    return {i.label for i in mir.code if isinstance(i, Label)}


def check_kernel(name: str, mir: Mir) -> None:
    """a kernel compile can write: no parameters, saved for a direct arena"""
    if mir.count_params != 0:
        refuse(f"{name} takes {mir.count_params} parameters")
    if not mir.option("direct_arena"):
        refuse(f"{name} was not saved for a direct arena")


def _known_label(labels: set[str], name: str, label: str) -> None:
    if label not in labels:
        refuse(f"{name}: no label {label}")


def _funclet(labels: set[str], name: str, i: Call) -> None:
    if not (i.num_args == 0 and i.label in labels):
        refuse(f"{name}: a call of {i.label} (math functions are not supported)")


def declared(names) -> list[str]:
    """the names of a declaration, shortest first"""
    return sorted(names, key=lambda r: (len(r), r))


# ---- kernel_c --------------------------------------------------------------------------------------------

@dataclass
class _Writer:
    """What kernel_c collects while it writes a kernel: the C lines of the body, the registers and
    stack slots they use (declared at the top), and the funclet calls (each a return point)."""
    name: str
    mir: Mir
    complex: bool               # a complex model: the operand of a LoadMath goes to Temp, else to Ret
    labels: set
    regs: set = field(default_factory=set)
    slots: set = field(default_factory=set)
    body: list = field(default_factory=list)
    returns: int = 0            # funclet calls so far: the return points R_0, R_1, ...
    has_ret: bool = False       # a funclet returns (a branch to ".ret")

    def line(self, s: str) -> None:
        self.body.append(s)

    def used(self, r: Reg) -> str:
        s = reg_c(r)
        self.regs.add(s)
        return s

    def at(self, l: Loc, offset: int = 0) -> str:
        if l.kind == "Stack":
            self.slots.add(l.index + offset)
        return loc_c(l, offset)

    def assign(self, lhs: str, rhs: str) -> None:
        """`lhs = rhs;`, but none when they are the same name (a stack slot kept in its local:
        clang's -Wself-assign)"""
        if lhs != rhs:
            self.line(f"{lhs} = {rhs};")

    def emit(self, i: Instruction) -> None:
        if isinstance(i, (Nop, End)):
            return
        if isinstance(i, Uni):
            f = UNI_C.get(i.op) or refuse(f"{self.name}: the operation {i.op}")
            self.line(f"{self.used(i.dst)} = {f(self.used(i.s1))};")
        elif isinstance(i, Bi):
            f = BI_C.get(i.op) or refuse(f"{self.name}: the operation {i.op}")
            self.line(f"{self.used(i.dst)} = {f(self.used(i.s1), self.used(i.s2))};")
        elif isinstance(i, Fused):
            f = FUSED_C.get(i.op) or refuse(f"{self.name}: the operation {i.op}")
            self.line(f"{self.used(i.dst)} = {f(self.used(i.a), self.used(i.b), self.used(i.c))};")
        elif isinstance(i, Mov):
            lhs = self.used(i.dst)
            self.assign(lhs, self.used(i.s1))
        elif isinstance(i, Load):
            lhs = self.used(i.dst)
            self.assign(lhs, self.at(i.loc))
        elif isinstance(i, Save):
            lhs = self.at(i.loc)
            self.assign(lhs, self.used(i.src))
        elif isinstance(i, LoadConst):
            self.line(f"{self.used(i.dst)} = {const_c(self.mir.consts[i.idx])};")
        elif isinstance(i, LoadComplex):
            self.line(f"{self.used(i.xd)} = {self.at(i.loc)}; {self.used(i.yd)} = {self.at(i.loc, 1)};")
        elif isinstance(i, SaveComplex):
            self.line(f"{self.at(i.loc)} = {self.used(i.xs)}; {self.at(i.loc, 1)} = {self.used(i.ys)};")
        elif isinstance(i, (LoadMath, LoadConstMath)):
            # the operand goes to Temp (complex) or Ret (real) first, as symjit replays it
            t = self.used(Reg("Temp" if self.complex else "Ret"))
            operand = self.at(i.loc) if isinstance(i, LoadMath) else const_c(self.mir.consts[i.idx])
            f = BI_C.get(i.op) or refuse(f"{self.name}: the operation {i.op}")
            self.line(f"{t} = {operand}; {self.used(i.dst)} = {f(self.used(i.s1), t)};")
        elif isinstance(i, ComplexBi):
            self.complex_bi(i)
        elif isinstance(i, IfElse):
            self.line(f"{self.used(i.dst)} = {self.at(i.cond)} != 0.0 ? {self.used(i.true_val)} : {self.used(i.false_val)};")
        elif isinstance(i, Label):
            self.line(f"{label_c(i.label)}:;")
        elif isinstance(i, Branch):
            if i.label == ".ret":
                self.line("goto L_return;")
                self.has_ret = True
            else:
                _known_label(self.labels, self.name, i.label)
                self.line(f"goto {label_c(i.label)};")
        elif isinstance(i, BranchIf):
            _known_label(self.labels, self.name, i.label)
            self.line(f"if ({'' if i.is_else else '!'}({self.used(i.cond)} == 0.0)) goto {label_c(i.label)};")
        elif isinstance(i, Call):
            # a funclet call: its return point is the next R_k
            _funclet(self.labels, self.name, i)
            self.line(f"ret = {self.returns}; goto {label_c(i.label)}; R_{self.returns}:;")
            self.returns += 1
        else:
            refuse(f"{self.name}: the instruction {i}")

    def complex_bi(self, i: ComplexBi) -> None:
        """+, -, ×, ÷ of register pairs, as symjit's x86-64 generator (amd/scalar.rs times_complex,
        divide_complex)"""
        (xd, yd, x1, y1, x2, y2) = (self.used(r) for r in (i.xd, i.yd, i.x1, i.y1, i.x2, i.y2))
        (xt, yt, t) = (self.used(r) for r in (Reg("Gen", 2), Reg("Gen", 3), Reg("Temp")))
        if i.op in ("Plus", "Minus"):
            o = "+" if i.op == "Plus" else "-"
            self.line(f"{xd} = {x1} {o} {x2}; {yd} = {y1} {o} {y2};")
        elif i.op == "Times":
            self.line(f"{xt} = {y1} * {y2}; {xt} = fma({x1}, {x2}, -{xt}); {yt} = {x1} * {y2}; {yd} = fma({x2}, {y1}, {yt}); "
                      f"{xd} = {xt};")
        elif i.op == "Divide":
            self.line(f"{xt} = {y1} * {y2}; {xt} = fma({x1}, {x2}, {xt}); {yt} = {x1} * {y2}; {yt} = fma({x2}, {y1}, -{yt}); "
                      f"{t} = {x2} * {x2}; {t} = fma({y2}, {y2}, {t}); {xd} = {xt} / {t}; {yd} = {yt} / {t};")
        else:
            refuse(f"{self.name}: the complex operation {i.op}")


def kernel_function(name: str, comment: str, declarations: list[str], body: list[str], returns: int, has_ret: bool) -> str:
    """The C function of a kernel: `comment`, the declarations, the body, and the funclets' return
    switch (`returns` return points; `has_ret`: a funclet returns)."""
    out = [comment, f"PAC_FN static int {name}(double *mem) {{"]
    out += declarations
    if has_ret:
        out.append("int ret = -1;")
    out += body
    out.append("return 0;")
    if has_ret:
        out += ["L_return:", "switch (ret) {"]
        out += [f"case {k}: goto R_{k};" for k in range(returns)]
        out += ["}", "return -1;  /* a return outside a funclet */"]
    out.append("}")
    return "".join(s + "\n" for s in out)


Source = Union[Mir, bytes, str]


def _mir(source: Source) -> Mir:
    return source if isinstance(source, Mir) else read_mir(source)


def kernel_c(name: str, source: Source) -> str:
    """the kernel `name` as C (KERNEL_PRELUDE before it); `source`: its Mir, or its saved application
    (.sjb bytes or path)"""
    mir = _mir(source)
    check_kernel(name, mir)
    g = _Writer(name, mir, mir.is_complex(), _labels(mir))
    for i in mir.code:
        g.emit(i)
    declarations = []
    if g.slots:
        declarations.append("double " + ", ".join(f"stack{k} = 0.0" for k in sorted(g.slots)) + ";")
    if g.regs:
        declarations.append("double " + ", ".join(f"{r} = 0.0" for r in declared(g.regs)) + ";")
    comment = f"/* the symjit kernel {name} ({len(mir.code)} MIR instructions), from its .plane.symjit by kernel_c.jl */"
    return kernel_function(name, comment, declarations, g.body, g.returns, g.has_ret)


# ---- the mixed-precision kernel (PAC_MIXED, pacops.h; Metal, kernel_msl.py) -------------------------------
#
# kernel_c_mixed(name, mir): the same kernel with each value an f32 or an f64, as sjb2ptx's `--mixed
# need` (symjit-utils src/ptx/mixed.rs) that pacuda's mixed kernels used: the divisions and
# reciprocals are f64, and so is every value a divisor or a reciprocal's operand depends on (the
# backward slices of the denominators, e.g. p² − m² + i m Γ); the bit operations (comparisons, masks,
# sign) stay f64 on their exact operands; every other value is f32. mem stays double: an input is
# rounded where an f32 value loads it, an output widened where it is saved.
#
# The kernel is first a list of steps (one value each: an expression of its operands, rendered in the
# step's type); a register or stack slot `x` has a wide `x` and a float `x_f`, and the current type
# of each name is followed through the straight-line code. The analysis does not follow control flow:
# before every label, jump, funclet call and return, the f32 values are widened (exactly) into their
# wide values, and after them every value is wide.
#
# The language (`lang`): "c" writes C, the wide type double (kernel_c_mixed); "msl" writes the Metal
# Shading Language, the wide type df64 (c/metal/df64.h; kernel_msl.py), straight-line only (its
# funclets inlined first: MSL has no goto). pac-julia's kernel_c.jl, the same text in both.

F32, F64 = "f32", "f64"


@dataclass
class _Step:
    dst: str          # the name it defines ("" for none: a store to mem, control)
    args: list        # the names it reads
    kind: str         # arith (f32 unless needed), f64 (always), move (its source's type unless needed),
                      # select (f64 when a branch is; its condition a mask, f64), load (from mem),
                      # save (to mem), ctrl (control: raw C)
    render: Callable  # (operand expressions, type) -> the expression (or, ctrl, the C)
    seeds: list       # the operands it needs in f64 (a divisor)


# ---- the spellings of the two languages: t is F32 or F64 (the wide type)

# a math function of type t: C sqrtf / sqrt; MSL sqrt / dd_sqrt (nearbyint is MSL's rint)
_MSL_WIDE_FN = {"sqrt": "dd_sqrt", "fabs": "dd_abs", "fma": "dd_fma", "nearbyint": "dd_rint", "floor": "dd_floor",
                "ceil": "dd_ceil", "trunc": "dd_trunc"}


def _mfn(lang: str, t: str, f: str) -> str:
    if lang == "c":
        return f + "f" if t == F32 else f   # sqrt → sqrtf
    if t == F32:
        return "rint" if f == "nearbyint" else f
    return _MSL_WIDE_FN[f]


def f32_msl(x) -> str:
    """a float32 as an MSL literal that reads back exactly: Julia's shortest repr, "0.5f0" → "0.5e0f",
    NaN by its bits, ±INFINITY"""
    x = np.float32(x)
    if np.isnan(x):
        return f"as_type<float>(0x{int(x.view(np.uint32)):x}u)"
    if np.isinf(x):
        return "INFINITY" if x > 0 else "(-INFINITY)"
    m, _, e = julia_repr_f32(x).partition("f")
    return f"{m}e{e}f"


def df64_pair(x: float) -> tuple:
    """a double as df64: (hi, lo), hi the nearest float32, lo the rest rounded (0 when hi is not finite)"""
    with np.errstate(over="ignore", invalid="ignore"):
        hi = np.float32(x)
        lo = np.float32(float(x) - float(hi)) if np.isfinite(hi) else np.float32(0.0)
    return hi, lo


def _mlit(lang: str, t: str, x: float) -> str:
    """a double in type t: C ((float)lit) / lit; MSL a float literal / a df64 (hi, lo) of the double"""
    if lang == "c":
        return f"((float){const_c(x)})" if t == F32 else const_c(x)
    if t == F32:
        with np.errstate(over="ignore"):
            return f32_msl(np.float32(x))
    hi, lo = df64_pair(x)
    return f"dd({f32_msl(hi)}, {f32_msl(lo)})"


def _wide_type(lang: str) -> str:
    return "double" if lang == "c" else "df64"


def _widen(lang: str, e: str) -> str:     # exact
    return f"(double){e}" if lang == "c" else f"dd_f({e})"


def _narrow(lang: str, e: str) -> str:    # rounded
    return f"(float){e}" if lang == "c" else f"dd_to_f({e})"


def _recip(lang: str, e: str) -> str:
    return f"1.0 / {e}" if lang == "c" else f"dd_f(1.0f) / {e}"


def _condition(lang: str, e: str) -> str:
    return f"{e} != 0.0" if lang == "c" else f"dd_true({e})"


# the f64 operations on their exact operands: comparisons and masks (C: BI_C, UNI_C)
BI_MSL: dict[str, Callable[[str, str], str]] = {
    "GreaterThan": lambda a, b: f"dd_mask({a} > {b})", "GreaterThanEqual": lambda a, b: f"dd_mask({a} >= {b})",
    "LittleThan": lambda a, b: f"dd_mask({a} < {b})", "LittleThanEqual": lambda a, b: f"dd_mask({a} <= {b})",
    "Equal": lambda a, b: f"dd_mask({a} == {b})", "NotEqual": lambda a, b: f"dd_mask({a} != {b})",
    "And": lambda a, b: f"dd_and({a}, {b})", "AndNot": lambda a, b: f"dd_andnot({a}, {b})",
    "Or": lambda a, b: f"dd_or({a}, {b})", "Xor": lambda a, b: f"dd_xor({a}, {b})",
}
UNI_MSL: dict[str, Callable[[str], str]] = {
    "Not": lambda x: f"dd_not({x})", "IsZero": lambda x: f"dd_mask(dd_is_zero({x}))",
    "IsNotZero": lambda x: f"dd_mask(!dd_is_zero({x}))", "Sign": lambda x: f"dd_sign({x})",
}


def _bi_wide(lang: str, op: str):
    return (BI_C if lang == "c" else BI_MSL).get(op)


def _uni_wide(lang: str, op: str):
    return (UNI_C if lang == "c" else UNI_MSL).get(op)


def _mixed_uni(lang: str) -> dict:
    """the unary operations that follow their value's type: (operands, type) -> expression"""
    return {
        "Neg": lambda a, t: f"-{a[0]}", "Abs": lambda a, t: f"{_mfn(lang, t, 'fabs')}({a[0]})",
        "Abs2": lambda a, t: f"{a[0]} * {a[0]}",
        "Root": lambda a, t: f"{_mfn(lang, t, 'sqrt')}({a[0]})", "RealRoot": lambda a, t: f"{_mfn(lang, t, 'sqrt')}({a[0]})",
        "Round": lambda a, t: f"{_mfn(lang, t, 'nearbyint')}({a[0]})", "Floor": lambda a, t: f"{_mfn(lang, t, 'floor')}({a[0]})",
        "Ceiling": lambda a, t: f"{_mfn(lang, t, 'ceil')}({a[0]})", "Trunc": lambda a, t: f"{_mfn(lang, t, 'trunc')}({a[0]})",
        "Half": lambda a, t: f"{a[0]} * {_mlit(lang, t, 0.5)}",
    }


def _mixed_fused(lang: str) -> dict:
    """the fused multiply-adds: (operands, type) -> expression"""
    return {
        "MulAdd": lambda a, t: f"{_mfn(lang, t, 'fma')}({a[0]}, {a[1]}, {a[2]})",
        "MulSub": lambda a, t: f"{_mfn(lang, t, 'fma')}({a[0]}, {a[1]}, -{a[2]})",
        "NegMulAdd": lambda a, t: f"{_mfn(lang, t, 'fma')}(-{a[0]}, {a[1]}, {a[2]})",
        "NegMulSub": lambda a, t: f"{_mfn(lang, t, 'fma')}(-{a[0]}, {a[1]}, -{a[2]})",
    }




def _copy(a, _):
    return a[0]


@dataclass
class _MixedWriter:
    """what _mixed_steps collects: the steps, and the funclets as _Writer does"""
    name: str
    mir: Mir
    complex: bool
    labels: set
    lang: str = "c"
    steps: list = field(default_factory=list)
    returns: int = 0
    has_ret: bool = False

    def __post_init__(self):
        self.uni = _mixed_uni(self.lang)
        self.fused = _mixed_fused(self.lang)

    def step(self, dst, args, kind, render, seeds=()):
        self.steps.append(_Step(dst, list(args), kind, render, list(seeds)))

    def arith(self, dst, args, render):
        self.step(dst, args, "arith", render)

    def move(self, dst, src):
        self.step(dst, [src], "move", _copy)

    def ctrl(self, c: str):
        self.step("", [], "ctrl", lambda _a, _t: c)

    def const(self, dst, x: float):
        lang = self.lang
        self.arith(dst, [], lambda _a, t: _mlit(lang, t, x))

    def load(self, dst, l: Loc, offset=0):
        """a load from a location: mem is wide (rounded there); a stack slot is a name"""
        n = loc_c(l, offset)
        lang = self.lang
        if l.kind == "Mem":
            self.step(dst, [], "load", lambda _a, t: _narrow(lang, n) if t == F32 else n)
        else:
            self.move(dst, n)

    def save(self, src, l: Loc, offset=0):
        """a save to a location: mem is wide (widened there); a stack slot is a name"""
        n = loc_c(l, offset)
        if l.kind == "Mem":
            self.step("", [src], "save", lambda a, _t: f"{n} = {a[0]}")
        else:
            self.move(n, src)

    def bi(self, op: str, dst, s1, s2):
        """a real binary operation: the division f64 with its divisor a seed, +, -, × following their
        value's type, the others (comparisons, masks: BI_C's, BI_MSL's) f64 on their exact operands"""
        if op == "Divide":
            self.step(dst, [s1, s2], "f64", lambda a, _t: f"{a[0]} / {a[1]}", seeds=[s2])
        elif op in ("Plus", "Minus", "Times"):
            o = {"Plus": "+", "Minus": "-", "Times": "*"}[op]
            self.arith(dst, [s1, s2], lambda a, _t: f"{a[0]} {o} {a[1]}")
        else:
            f = _bi_wide(self.lang, op) or refuse(f"{self.name}: the operation {op}")
            self.step(dst, [s1, s2], "f64", lambda a, _t: f(a[0], a[1]))

    def emit(self, i: Instruction) -> None:
        if isinstance(i, (Nop, End)):
            return
        if isinstance(i, Uni):
            d, s = reg_c(i.dst), reg_c(i.s1)
            lang = self.lang
            if i.op == "Recip":
                self.step(d, [s], "f64", lambda a, _t: _recip(lang, a[0]), seeds=[s])
            elif i.op in ("Real", "Conjugate"):
                self.move(d, s)
            elif i.op == "Imaginary":
                self.const(d, 0.0)
            elif i.op in self.uni:
                self.arith(d, [s], self.uni[i.op])
            else:
                f = _uni_wide(lang, i.op) or refuse(f"{self.name}: the operation {i.op}")
                self.step(d, [s], "f64", lambda a, _t: f(a[0]))
        elif isinstance(i, Bi):
            self.bi(i.op, reg_c(i.dst), reg_c(i.s1), reg_c(i.s2))
        elif isinstance(i, Fused):
            f = self.fused.get(i.op) or refuse(f"{self.name}: the operation {i.op}")
            self.arith(reg_c(i.dst), [reg_c(i.a), reg_c(i.b), reg_c(i.c)], f)
        elif isinstance(i, Mov):
            self.move(reg_c(i.dst), reg_c(i.s1))
        elif isinstance(i, Load):
            self.load(reg_c(i.dst), i.loc)
        elif isinstance(i, Save):
            self.save(reg_c(i.src), i.loc)
        elif isinstance(i, LoadConst):
            self.const(reg_c(i.dst), self.mir.consts[i.idx])
        elif isinstance(i, LoadComplex):
            self.load(reg_c(i.xd), i.loc)
            self.load(reg_c(i.yd), i.loc, 1)
        elif isinstance(i, SaveComplex):
            self.save(reg_c(i.xs), i.loc)
            self.save(reg_c(i.ys), i.loc, 1)
        elif isinstance(i, (LoadMath, LoadConstMath)):
            t = reg_c(Reg("Temp" if self.complex else "Ret"))
            if isinstance(i, LoadMath):
                self.load(t, i.loc)
            else:
                self.const(t, self.mir.consts[i.idx])
            self.bi(i.op, reg_c(i.dst), reg_c(i.s1), t)
        elif isinstance(i, ComplexBi):
            self.complex_bi(i)
        elif isinstance(i, IfElse):
            lang = self.lang
            self.step(reg_c(i.dst), [reg_c(i.true_val), reg_c(i.false_val), loc_c(i.cond, 0)], "select",
                      lambda a, _t: f"{_condition(lang, a[2])} ? {a[0]} : {a[1]}")
        elif isinstance(i, Label):
            self.ctrl(f"{label_c(i.label)}:;")
        elif isinstance(i, Branch):
            if i.label == ".ret":
                self.ctrl("goto L_return;")
                self.has_ret = True
            else:
                _known_label(self.labels, self.name, i.label)
                self.ctrl(f"goto {label_c(i.label)};")
        elif isinstance(i, BranchIf):
            _known_label(self.labels, self.name, i.label)
            neg, target = "" if i.is_else else "!", label_c(i.label)
            self.step("", [reg_c(i.cond)], "ctrl", lambda a, _t: f"if ({neg}({a[0]} == 0.0)) goto {target};")
        elif isinstance(i, Call):
            _funclet(self.labels, self.name, i)
            self.ctrl(f"ret = {self.returns}; goto {label_c(i.label)}; R_{self.returns}:;")
            self.returns += 1
        else:
            refuse(f"{self.name}: the instruction {i}")

    def complex_bi(self, i: ComplexBi) -> None:
        """as _Writer.complex_bi, one step per operation"""
        (xd, yd, x1, y1, x2, y2) = (reg_c(r) for r in (i.xd, i.yd, i.x1, i.y1, i.x2, i.y2))
        (xt, yt, t) = (reg_c(r) for r in (Reg("Gen", 2), Reg("Gen", 3), Reg("Temp")))
        if i.op in ("Plus", "Minus"):
            self.bi(i.op, xd, x1, x2)
            self.bi(i.op, yd, y1, y2)
        elif i.op == "Times":
            self.bi("Times", xt, y1, y2)
            self.arith(xt, [x1, x2, xt], self.fused["MulSub"])
            self.bi("Times", yt, x1, y2)
            self.arith(yd, [x2, y1, yt], self.fused["MulAdd"])
            self.move(xd, xt)
        elif i.op == "Divide":
            self.bi("Times", xt, y1, y2)
            self.arith(xt, [x1, x2, xt], self.fused["MulAdd"])
            self.bi("Times", yt, x1, y2)
            self.arith(yt, [x2, y1, yt], self.fused["MulSub"])
            self.bi("Times", t, x2, x2)
            self.arith(t, [y2, y2, t], self.fused["MulAdd"])
            self.bi("Divide", xd, xt, t)
            self.bi("Divide", yd, yt, t)
        else:
            refuse(f"{self.name}: the complex operation {i.op}")


def _needed_steps(steps: list) -> list[bool]:
    """backward: the steps whose value a denominator needs in f64 (the seeds' backward slices)"""
    needed: set = set()
    need = [False] * len(steps)
    for k in reversed(range(len(steps))):
        s = steps[k]
        if s.dst and s.dst in needed:
            need[k] = True
            needed.discard(s.dst)
            needed.update(s.args)
        needed.update(s.seeds)
    return need


@dataclass
class _MixedBody:
    """What the forward pass keeps: each name's current type (absent: f64, as declared), the names
    declared as wide and as float, the lines, and the counts for the kernel's comment."""
    lang: str = "c"
    cur: dict = field(default_factory=dict)
    doubles: set = field(default_factory=set)
    floats: set = field(default_factory=set)
    body: list = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {F64: 0, F32: 0, "conversions": 0})

    def current_type(self, n: str) -> str:
        return self.cur.get(n, F64)

    def ref(self, n: str, t: str) -> str:
        """the name `n` read as type `t`: its variable of its current type, converted when that differs"""
        if n.startswith("mem["):
            return _narrow(self.lang, n) if t == F32 else n
        have = self.current_type(n)
        (self.floats if have == F32 else self.doubles).add(n)
        if have == t:
            return f"{n}_f" if have == F32 else n
        return _widen(self.lang, f"{n}_f") if t == F64 else _narrow(self.lang, n)

    def is_conversion(self, r: str) -> bool:
        """a conversion in an operand (counted)"""
        if self.lang == "c":
            return r.startswith("(double)") or r.startswith("(float)")
        return r.startswith("dd_f(") or r.startswith("dd_to_f(")

    def sync(self) -> None:
        """before control flow: every f32 value widened (exactly) into its wide value; after it all are f64"""
        for n, t in sorted(self.cur.items()):
            if t != F32:
                continue
            self.body.append(f"{n} = {_widen(self.lang, n + '_f')};")
            self.doubles.add(n)
            self.counts["conversions"] += 1
        self.cur.clear()

    def step_type(self, s: _Step, needed: bool) -> str:
        """the type of a step's value: f64 when a denominator needs it"""
        if needed or s.kind == "f64":
            return F64
        if s.kind == "move":
            return self.current_type(s.args[0])
        if s.kind == "select":
            return F64 if any(self.current_type(a) == F64 for a in s.args[:2]) else F32
        if s.kind == "save":
            return F64
        return F32

    def emit(self, s: _Step, needed: bool) -> None:
        """one step: its operands converted where their current type differs from the step's"""
        if s.kind == "ctrl":
            if self.lang != "c":
                refuse("control flow in an MSL kernel (only straight-line kernels and inlined funclets)")
            self.sync()
            self.body.append(s.render([self.ref(a, F64) for a in s.args], F64))
            return
        t = self.step_type(s, needed)
        operands = []
        for j, a in enumerate(s.args):
            want = F64 if s.kind == "select" and j == 2 else t     # the condition: a mask (f64)
            r = self.ref(a, want)
            if self.is_conversion(r):
                self.counts["conversions"] += 1
            operands.append(r)
        if s.kind == "load" and t == F32:
            self.counts["conversions"] += 1
        expr = s.render(operands, t)
        if s.kind == "save":
            self.body.append(expr + ";")
            return
        if s.kind not in ("move", "select"):
            self.counts[t] += 1
        self.cur[s.dst] = t
        if t == F32:
            self.floats.add(s.dst)
            self.body.append(f"{s.dst}_f = {expr};")
        else:
            self.doubles.add(s.dst)
            self.body.append(f"{s.dst} = {expr};")


def mixed_body(name: str, mir: Mir, lang: str = "c"):
    """the typed body of a kernel in `lang` ("c" or "msl"): (declarations, body, counts, returns, has_ret)"""
    m = _MixedWriter(name, mir, mir.is_complex(), _labels(mir), lang)
    for i in mir.code:
        m.emit(i)
    need = _needed_steps(m.steps)
    b = _MixedBody(lang)
    for s, n in zip(m.steps, need):
        b.emit(s, n)
    declarations = []
    zero = "0.0" if lang == "c" else "dd_f(0.0f)"
    if b.doubles:
        declarations.append(_wide_type(lang) + " " + ", ".join(f"{n} = {zero}" for n in declared(b.doubles)) + ";")
    if b.floats:
        declarations.append("float " + ", ".join(f"{n}_f = 0.0f" for n in declared(b.floats)) + ";")
    return declarations, b.body, b.counts, m.returns, m.has_ret


def kernel_c_mixed(name: str, source: Source) -> str:
    """the kernel `name` in mixed precision (KERNEL_PRELUDE before it); same prototype as kernel_c's"""
    mir = _mir(source)
    check_kernel(name, mir)
    declarations, body, counts, returns, has_ret = mixed_body(name, mir)
    comment = (f"/* the symjit kernel {name} in mixed precision (kernel_c.jl's kernel_c_mixed): "
               f"{counts[F32]} f32 and {counts[F64]} f64 operations, {counts['conversions']} conversions */")
    return kernel_function(name, comment, declarations, body, returns, has_ret)

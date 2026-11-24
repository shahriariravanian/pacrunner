"""A model kernel in the mixed kernels (plans/F32-KERNEL.md): kernel_msl's kernel (kernels/kernel_msl.py:
kernel_c_mixed's steps and `need` rule, df64 for the wide type) written as MIR on simd/mir.py, so that
the f64 + double-double kernel, downgraded, computes the Metal runner's bits.

kernel_fragment(name, mir) -> a fragment (simd/library.py's format):
  inputs   x<k>.hi, x<k>.lo: the kernel's states (mem[k], df64 on Metal); w<i>.hi, w<i>.lo: the wide
           constant i (the kernel's constant pool entry i where a step needs it in df64: the assembler binds
           it to a dd constant, split at the downgrade as kernel_msl's literal dd(hi, lo))
  outputs  o<j>.hi, o<j>.lo: the kernel's outputs (mem[states + j], df64)

The typing is kernel_c_mixed's, not a copy of it: Writer records the same steps as kernel_c's
_MixedWriter (the kinds, operands and seeds; check_steps compares them) with their operations as data,
_needed_steps and _MixedBody.step_type type them, and emit renders each step as kernel_msl does:
  f32   the f32 operation (fma for Fused: one rounding);
  df64  df64.h's: + - × ÷ (AccurateDWPlusDW, DWTimesDW3, DWDivDW2), dd_fma = a × b + c, dd_sqrt, dd_abs,
        dd_floor/ceil/trunc/rint; comparisons as masks in both halves (dd_mask), the bit operations on
        both halves, a select's condition dd_true (either half nonzero).
A step's operand of the other type is converted: f32 to df64 exactly (dd_f), df64 to f32 rounded once
(dd_to_f). Funclets are inlined first (kernel_msl.inline_funclets); other control flow is refused.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..kernels.kernel_c import (F32, F64, _labels, _MixedBody, _MixedWriter, _needed_steps, _Step, check_kernel, loc_c,
                                reg_c, refuse)
from ..kernels.kernel_msl import inline_funclets
from ..kernels.mir import (Bi, ComplexBi, End, Fused, IfElse, Load, LoadComplex, LoadConst, LoadConstMath, LoadMath, Mov,
                           Nop, Reg, Save, SaveComplex, Uni)
from .mir import DDX, Mir, dd

BI_WIDE = {"GreaterThan", "GreaterThanEqual", "LittleThan", "LittleThanEqual", "Equal", "NotEqual", "And", "AndNot",
           "Or", "Xor"}
UNI_WIDE = {"Not", "IsZero", "IsNotZero", "Sign"}
UNI_TYPED = {"Neg", "Abs", "Abs2", "Root", "RealRoot", "Round", "Floor", "Ceiling", "Trunc", "Half"}


@dataclass
class Writer:
    """kernel_c's _MixedWriter, the steps' operations as data (sem: a tuple) instead of rendered text"""
    name: str
    mir: object
    complex: bool
    steps: list = field(default_factory=list)

    def step(self, dst, args, kind, sem, seeds=()):
        self.steps.append(_Step(dst, list(args), kind, sem, list(seeds)))

    def arith(self, dst, args, sem):
        self.step(dst, args, "arith", sem)

    def move(self, dst, src):
        self.step(dst, [src], "move", ("move",))

    def const(self, dst, k: int):
        self.arith(dst, [], ("const", k))

    def load(self, dst, l, offset=0):
        n = loc_c(l, offset)
        if l.kind == "Mem":
            self.step(dst, [], "load", ("load", l.index + offset))
        else:
            self.move(dst, n)

    def save(self, src, l, offset=0):
        n = loc_c(l, offset)
        if l.kind == "Mem":
            self.step("", [src], "save", ("save", l.index + offset))
        else:
            self.move(n, src)

    def bi(self, op, dst, s1, s2):
        if op == "Divide":
            self.step(dst, [s1, s2], "f64", ("div",), seeds=[s2])
        elif op in ("Plus", "Minus", "Times"):
            self.arith(dst, [s1, s2], ("bi", op))
        elif op in BI_WIDE:
            self.step(dst, [s1, s2], "f64", ("wbi", op))
        else:
            refuse(f"{self.name}: the operation {op}")

    def emit(self, i) -> None:
        if isinstance(i, (Nop, End)):
            return
        if isinstance(i, Uni):
            d, s = reg_c(i.dst), reg_c(i.s1)
            if i.op == "Recip":
                self.step(d, [s], "f64", ("recip",), seeds=[s])
            elif i.op in ("Real", "Conjugate"):
                self.move(d, s)
            elif i.op == "Imaginary":
                self.arith(d, [], ("lit", 0.0))
            elif i.op in UNI_TYPED:
                self.arith(d, [s], ("uni", i.op))
            elif i.op in UNI_WIDE:
                self.step(d, [s], "f64", ("wuni", i.op))
            else:
                refuse(f"{self.name}: the operation {i.op}")
        elif isinstance(i, Bi):
            self.bi(i.op, reg_c(i.dst), reg_c(i.s1), reg_c(i.s2))
        elif isinstance(i, Fused):
            if i.op not in ("MulAdd", "MulSub", "NegMulAdd", "NegMulSub"):
                refuse(f"{self.name}: the operation {i.op}")
            self.arith(reg_c(i.dst), [reg_c(i.a), reg_c(i.b), reg_c(i.c)], ("fused", i.op))
        elif isinstance(i, Mov):
            self.move(reg_c(i.dst), reg_c(i.s1))
        elif isinstance(i, Load):
            self.load(reg_c(i.dst), i.loc)
        elif isinstance(i, Save):
            self.save(reg_c(i.src), i.loc)
        elif isinstance(i, LoadConst):
            self.const(reg_c(i.dst), i.idx)
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
                self.const(t, i.idx)
            self.bi(i.op, reg_c(i.dst), reg_c(i.s1), t)
        elif isinstance(i, ComplexBi):
            (xd, yd, x1, y1, x2, y2) = (reg_c(r) for r in (i.xd, i.yd, i.x1, i.y1, i.x2, i.y2))
            (xt, yt, t) = (reg_c(r) for r in (Reg("Gen", 2), Reg("Gen", 3), Reg("Temp")))
            if i.op in ("Plus", "Minus"):
                self.bi(i.op, xd, x1, x2)
                self.bi(i.op, yd, y1, y2)
            elif i.op == "Times":
                self.bi("Times", xt, y1, y2)
                self.arith(xt, [x1, x2, xt], ("fused", "MulSub"))
                self.bi("Times", yt, x1, y2)
                self.arith(yd, [x2, y1, yt], ("fused", "MulAdd"))
                self.move(xd, xt)
            elif i.op == "Divide":
                self.bi("Times", xt, y1, y2)
                self.arith(xt, [x1, x2, xt], ("fused", "MulAdd"))
                self.bi("Times", yt, x1, y2)
                self.arith(yt, [x2, y1, yt], ("fused", "MulSub"))
                self.bi("Times", t, x2, x2)
                self.arith(t, [y2, y2, t], ("fused", "MulAdd"))
                self.bi("Divide", xd, xt, t)
                self.bi("Divide", yd, yt, t)
            else:
                refuse(f"{self.name}: the complex operation {i.op}")
        elif isinstance(i, IfElse):
            self.step(reg_c(i.dst), [reg_c(i.true_val), reg_c(i.false_val), loc_c(i.cond, 0)], "select", ("select",))
        else:
            refuse(f"{self.name}: control flow or the instruction {i} (the mixed kernels are straight-line)")


def prepared(name: str, mir):
    """the kernel's MIR as kernel_msl takes it: checked, its funclets inlined"""
    check_kernel(name, mir)
    return dataclasses.replace(mir, code=tuple(inline_funclets(name, mir.code)))


def steps(name: str, mir) -> list:
    w = Writer(name, mir, mir.is_complex())
    for i in mir.code:
        w.emit(i)
    return w.steps


def check_steps(name: str, mir) -> None:
    """the Writer's steps are kernel_c's _MixedWriter's (lang "msl"): the same names, operands, kinds, seeds"""
    ours = steps(name, mir)
    m = _MixedWriter(name, mir, mir.is_complex(), _labels(mir), "msl")
    for i in mir.code:
        m.emit(i)
    key = lambda s: (s.dst, tuple(s.args), s.kind, tuple(s.seeds))
    if [key(s) for s in ours] != [key(s) for s in m.steps]:
        raise AssertionError(f"{name}: the steps differ from kernel_c's _MixedWriter")


class _Emit:
    """_MixedBody's forward pass with Mir values: each name's current value, plain (f32) or dd (df64)"""

    def __init__(self, mir, m: Mir, d: DDX, states: int, wide_const):
        self.mir, self.m, self.d, self.states = mir, m, d, states
        self.cur: dict = {}                  # name -> ("f32", Val) | ("f64", dd)
        self.body = _MixedBody("msl")        # for step_type: its `cur` mirrors ours' types
        self.wide_const = wide_const         # constant index -> dd (the assembler's)
        self.inputs: dict = {}

    def input(self, k: int) -> dd:
        if k not in self.inputs:
            self.inputs[k] = self.d.input(f"x{k}")
        return self.inputs[k]

    def ref(self, n: str, t: str):
        """the name n as type t (an undefined name: 0, as kernel_msl's declarations)"""
        d, m = self.d, self.m
        have, v = self.cur.get(n, (F64, None))
        if v is None:
            v = d.exact(0.0)
        if have == t:
            return v
        return d.of(v) if t == F64 else d.to_plain(v)

    def mask(self, x: dd):
        """dd_true(x) as a mask: either half nonzero"""
        m = self.m
        z = m.const(0.0)
        return m.or_(m.ne(x.hi, z), m.ne(x.lo, z))

    def dd_mask(self, mask) -> dd:
        return dd(mask, mask)

    def lit(self, x: float, t: str):
        if t == F32:
            from ..codegen.compile_simd import fl32
            return self.m.const(fl32(x))
        hi_lo = self.wide_const(x)
        return hi_lo

    def emit(self, s: _Step, needed: bool) -> None:
        if s.kind == "ctrl":
            refuse("control flow in a mixed kernel (only straight-line kernels and inlined funclets)")
        t = self.body.step_type(s, needed)
        ops = [self.ref(a, F64 if (s.kind == "select" and j == 2) else t) for j, a in enumerate(s.args)]
        sem = s.render
        v = self.value(sem, ops, t, s)
        if s.kind == "save":
            return
        self.cur[s.dst] = (t, v)
        self.body.cur[s.dst] = t

    def value(self, sem, a, t, s):
        m, d = self.m, self.d
        wide = t == F64
        kind = sem[0]
        if kind == "move":
            return a[0]
        if kind == "const":
            return self.lit(self.mir.consts[sem[1]], t)
        if kind == "lit":
            return self.lit(sem[1], t)
        if kind == "load":
            x = self.input(sem[1])
            return x if wide else d.to_plain(x)
        if kind == "save":
            self.saves.append((sem[1], a[0]))
            return None
        if kind == "bi":
            op = sem[1]
            if wide:
                return {"Plus": d.add, "Minus": d.sub, "Times": d.mul}[op](a[0], a[1])
            return {"Plus": m.add, "Minus": m.sub, "Times": m.mul}[op](a[0], a[1])
        if kind == "div":
            return d.div(a[0], a[1])
        if kind == "recip":
            return d.div(d.exact(1.0), a[0])
        if kind == "fused":
            op = sem[1]
            if wide:          # dd_fma(a, b, c) = a × b + c, its operands negated as kernel_msl writes them
                x = d.neg(a[0]) if op in ("NegMulAdd", "NegMulSub") else a[0]
                c = d.neg(a[2]) if op in ("MulSub", "NegMulSub") else a[2]
                return d.add(d.mul(x, a[1]), c)
            return m.fused(op, a[0], a[1], a[2])
        if kind == "uni":
            op = sem[1]
            if op == "Neg":
                return d.neg(a[0]) if wide else m.neg(a[0])
            if op == "Abs":
                return d.abs(a[0]) if wide else m.uni("Abs", a[0])
            if op == "Abs2":
                return d.mul(a[0], a[0]) if wide else m.mul(a[0], a[0])
            if op in ("Root", "RealRoot"):
                return d.sqrt(a[0]) if wide else m.sqrt(a[0])
            if op == "Half":
                return d.mul(a[0], d.exact(0.5)) if wide else m.mul(a[0], m.const(0.5))
            if op in ("Round", "Floor", "Ceiling", "Trunc"):
                return self.rounding(op, a[0]) if wide else m.uni(op, a[0])
        if kind == "wbi":       # comparisons and masks, on exact (df64) operands
            op, x, y = sem[1], a[0], a[1]
            cmp = {"GreaterThan": lambda: d.gt(x, y), "GreaterThanEqual": lambda: d.le(y, x),
                   "LittleThan": lambda: d.lt(x, y), "LittleThanEqual": lambda: d.le(x, y),
                   "Equal": lambda: m.and_(m.eq(x.hi, y.hi), m.eq(x.lo, y.lo)), "NotEqual": lambda: d.ne(x, y)}
            if op in cmp:
                return self.dd_mask(cmp[op]())
            f = {"And": m.and_, "Or": m.or_, "Xor": m.xor, "AndNot": lambda p, q: m.bi("AndNot", p, q)}[op]
            return dd(f(x.hi, y.hi), f(x.lo, y.lo))
        if kind == "wuni":
            op, x = sem[1], a[0]
            if op == "Not":
                return dd(m.uni("Not", x.hi), m.uni("Not", x.lo))
            if op == "IsZero":
                return self.dd_mask(d.is_zero(x))
            if op == "IsNotZero":
                return self.dd_mask(m.uni("Not", d.is_zero(x)))
            if op == "Sign":
                sg = m.sign(x.hi)
                return dd(sg, sg)
        if kind == "select":
            c = self.mask(a[2])
            if wide:
                return d.select(c, a[0], a[1])
            return m.select(c, a[0], a[1])
        refuse(f"no mixed rendering of {sem}")

    def rounding(self, op, x: dd) -> dd:
        """df64.h's dd_floor, dd_ceil, dd_trunc, dd_rint"""
        d, m = self.d, self.m

        def fc(name, x):
            h = m.uni(name, x.hi)
            return d.select(m.ne(h, x.hi), dd(h, m.const(0.0)), d.fast_two_sum(h, m.uni(name, x.lo)))

        if op == "Floor":
            return fc("Floor", x)
        if op == "Ceiling":
            return fc("Ceiling", x)
        if op == "Trunc":
            return d.select(d.signbit_mask(x), fc("Ceiling", x), fc("Floor", x))
        f = fc("Floor", x)
        diff = d.sub(x, f)
        half = d.exact(0.5)
        one = d.exact(1.0)
        up = d.add(f, one)
        halved = fc("Floor", d.mul(f, half))
        twice = d.mul(halved, d.exact(2.0))
        tie = d.select(m.and_(m.eq(twice.hi, f.hi), m.eq(twice.lo, f.lo)), f, up)
        return d.select(d.lt(diff, half), f, d.select(d.gt(diff, half), up, tie))


def kernel_fragment(name: str, mir, inputs: int, outputs: int) -> tuple[dict, list]:
    """the kernel as a mixed fragment (see the module), and its wide constants: [(index, value)] in the
    order of its w<i> inputs"""
    mir = prepared(name, mir)
    check_steps(name, mir)
    ss = steps(name, mir)
    need = _needed_steps(ss)
    wide: dict = {}

    def wide_const(x):
        if x not in wide:
            wide[x] = len(wide)
        return ("w", wide[x])

    # pass 1: which wide constants are used (their inputs are named before the Mir is built)
    probe = _Probe(mir, inputs, wide_const)
    for s, n in zip(ss, need):
        probe.emit(s, n)
    names = [f"x{k}.{p}" for k in range(inputs) for p in ("hi", "lo")]
    names += [f"w{i}.{p}" for i in range(len(wide)) for p in ("hi", "lo")]
    m = Mir(names, 2 * outputs, f"kernel:{name}:mixed")
    d = DDX(m)
    ws = [d.input(f"w{i}") for i in range(len(wide))]
    e = _Emit(mir, m, d, inputs, lambda x: ws[wide[x]])
    e.saves = []
    for s, n in zip(ss, need):
        e.emit(s, n)
    written = {}
    for (k, v) in e.saves:
        written[k - inputs] = v
    for j in range(outputs):
        v = written.get(j)
        if v is None:
            v = d.exact(0.0)
        m.output(2 * j, v.hi)
        m.output(2 * j + 1, v.lo)
    return m.fragment(), sorted(wide, key=lambda x: wide[x])


class _Probe(_Emit):
    """pass 1 of kernel_fragment: the steps typed, only the wide constants recorded"""

    def __init__(self, mir, states, wide_const):
        self.mir, self.wide_const = mir, wide_const
        self.cur, self.body = {}, _MixedBody("msl")

    def emit(self, s: _Step, needed: bool) -> None:
        t = self.body.step_type(s, needed)
        if s.render[0] in ("const", "lit") and t == F64:
            x = self.mir.consts[s.render[1]] if s.render[0] == "const" else s.render[1]
            self.wide_const(x)
        if s.kind != "save":
            self.body.cur[s.dst] = t

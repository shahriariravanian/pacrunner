"""A model kernel split for the split kernels (plans/F32-KERNEL.md, "Split kernels"): its wide,
current-free steps (the backward slices of its divisors, as kernel_c_mixed's `need` rule finds them,
and its comparisons on constants) computed in f64 by kernel A, the rest in f32 by kernel B.

split(name, mir, kinds, inputs, outputs) -> Split: `kinds[k]` is "current" (a stub input from a
current: IN_L, IN_R, OUT) or "static" (a momentum, a scale, the factor) for each input k.
  - A side: the steps that do not depend on a current and are wide (needed by a divisor, or a
    comparison or mask), closed over their operands; a_fragment() computes them in f64: inputs x<k>
    (its static inputs), outputs the crossing values (the A values B reads), in `crossing` order.
  - B side: every other step, in plain arithmetic: b_fragment(), inputs x<k> (every input) then a<j>
    (the crossing values), outputs the kernel's outputs. Its constants are rounded to f32: it is
    compiled in f64 and downgraded.

The steps are kernel_c_mixed's: Writer records kernel_c's _MixedWriter's steps with their operations as
data (check_steps compares the two), and _needed_steps types them.
"""
from __future__ import annotations

import dataclasses
import struct
from dataclasses import dataclass, field

from ..kernels.kernel_c import _labels, _MixedWriter, _needed_steps, _Step, check_kernel, loc_c, reg_c, refuse
from ..kernels.kernel_msl import inline_funclets
from ..kernels.mir import (Bi, ComplexBi, End, Fused, IfElse, Load, LoadComplex, LoadConst, LoadConstMath, LoadMath, Mov,
                           Nop, Reg, Save, SaveComplex, Uni)
from .mir import Mir


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
            refuse(f"{self.name}: control flow or the instruction {i} (the split kernels are straight-line)")


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


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


@dataclass
class Split:
    name: str
    mir: object
    steps: list
    a: list            # step indices of the A side
    crossing: list     # step indices whose values B reads from A
    inputs: int
    outputs: int
    kinds: list

    def _render(self, m: Mir, sem, ops, f32: bool):
        """a step as plain arithmetic (f64, or f32 with its constants rounded)"""
        k = sem[0]
        if k == "move":
            return ops[0]
        if k in ("const", "lit"):
            x = self.mir.consts[sem[1]] if k == "const" else sem[1]
            return m.const(_f32(x) if f32 else x)
        if k == "bi":
            return {"Plus": m.add, "Minus": m.sub, "Times": m.mul}[sem[1]](ops[0], ops[1])
        if k == "div":
            return m.div(ops[0], ops[1])
        if k == "recip":
            return m.div(m.const(1.0), ops[0])
        if k == "fused":
            return m.fused(sem[1], *ops)
        if k == "uni":
            op = sem[1]
            if op == "Neg":
                return m.neg(ops[0])
            if op == "Abs2":
                return m.mul(ops[0], ops[0])
            if op == "Half":
                return m.mul(ops[0], m.const(0.5))
            if op == "RealRoot":
                return m.sqrt(ops[0])
            return m.uni(op, ops[0])
        if k == "wbi":
            return m.bi(sem[1], ops[0], ops[1])
        if k == "wuni":
            return m.uni(sem[1], ops[0])
        if k == "select":
            return m.select(ops[2], ops[0], ops[1])
        refuse(f"{self.name}: no plain rendering of {sem}")

    def _run(self, m: Mir, which: set, inputs: dict, given: dict, f32: bool):
        """the steps `which` in order; `inputs` k -> Val, `given` step index -> Val (values from elsewhere);
        returns (step index -> Val, saves: mem index -> Val)"""
        cur, out, saves = {}, {}, {}
        zero = None
        for i, s in enumerate(self.steps):
            if i in given:
                v = given[i]
            elif i in which:
                sem = s.render
                if sem[0] == "load":
                    v = inputs[sem[1]]
                else:
                    ops = []
                    for a in s.args:
                        if a in cur:
                            ops.append(cur[a])
                        else:
                            zero = zero or m.const(0.0)
                            ops.append(zero)
                    if sem[0] == "save":
                        saves[sem[1]] = ops[0]
                        continue
                    v = self._render(m, sem, ops, f32)
            else:
                if s.dst:
                    cur.pop(s.dst, None)
                continue
            if s.dst:
                cur[s.dst] = v
                out[i] = v
        return out, saves

    def a_fragment(self) -> dict:
        static = [k for k in range(self.inputs) if self.kinds[k] == "static"]
        m = Mir([f"x{k}" for k in static], len(self.crossing), f"kernel:{self.name}:A")
        ins = {k: m.input(f"x{k}") for k in static}
        vals, _ = self._run(m, set(self.a), ins, {}, f32=False)
        for j, i in enumerate(self.crossing):
            m.output(j, vals[i])
        return m.fragment()

    def b_fragment(self, f32: bool = True) -> dict:
        names = [f"x{k}" for k in range(self.inputs)] + [f"a{j}" for j in range(len(self.crossing))]
        m = Mir(names, self.outputs, f"kernel:{self.name}:B")
        ins = {k: m.input(f"x{k}") for k in range(self.inputs)}
        given = {i: m.input(f"a{j}") for j, i in enumerate(self.crossing)}
        b = set(range(len(self.steps))) - set(self.a)
        _, saves = self._run(m, b, ins, given, f32=f32)
        for j in range(self.outputs):
            v = saves.get(self.inputs + j)
            m.output(j, v if v is not None else m.const(0.0))
        return m.fragment()


def split(name: str, mir, kinds: list, inputs: int, outputs: int) -> Split:
    mir = prepared(name, mir)
    ss = steps(name, mir)
    need = _needed_steps(ss)
    # the producers of each step's operands (names are reused: follow the definitions in order)
    producer, prods = {}, []
    for i, s in enumerate(ss):
        prods.append([producer.get(a) for a in s.args])
        if s.dst:
            producer[s.dst] = i
    current = [False] * len(ss)
    for i, s in enumerate(ss):
        if s.render[0] == "load":
            current[i] = kinds[s.render[1]] == "current"
        else:
            current[i] = any(p is not None and current[p] for p in prods[i])
    wide = [(need[i] or s.kind == "f64") and not current[i] and s.kind != "save" for i, s in enumerate(ss)]
    a = set()
    stack = [i for i, w in enumerate(wide) if w]
    while stack:
        i = stack.pop()
        if i in a:
            continue
        a.add(i)
        stack.extend(p for p in prods[i] if p is not None and not current[p])
    crossing = sorted({p for i in range(len(ss)) if i not in a for p in prods[i] if p is not None and p in a})
    return Split(name, mir, ss, sorted(a), crossing, inputs, outputs, kinds)

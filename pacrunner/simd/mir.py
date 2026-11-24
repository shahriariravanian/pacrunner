"""Fragments written as symjit MIR directly (plans/F32-KERNEL.md): the mixed fragments need fused
multiply-adds (double-double arithmetic: two_prod), which symjit's expression trees cannot ask for
without fastmath, and fastmath would also contract and reorder the rest.

`Mir` builds a straight-line fragment over values (Val): loads of its inputs (Mem 0 .. n_in), constants,
Bi (Plus, Minus, Times, Divide), Fused (MulAdd a b + c, MulSub a b - c, NegMulAdd, NegMulSub), Neg, Root,
stores to its outputs (Mem n_in ..). Registers: Gen 0 .. 13 (ymm2 .. ymm15 / zmm2 .. zmm15; Ret and Temp
are the generators'), allocated in order of use; a value whose register is needed while it is still live
is spilled to one of the fragment's Stack slots (the furthest next use first) and reloaded when used.
The result is a library fragment (simd/library.py's format) that compile_simd splices like the others.

`DD` is double-double arithmetic on pairs of Vals, c/metal/df64.h's algorithms in its exact order of
operations (Joldes, Muller, Popescu 2017: AccurateDWPlusDW, DWTimesDW3, DWDivDW2; Karp-Markstein sqrt),
so that the fragment, downgraded to f32 (simd/downgrade.py), computes df64.h's bits.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

GENS = 14


def bits(x: float) -> str:
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]


@dataclass(frozen=True)
class Val:
    id: int


@dataclass
class _Op:
    kind: str                       # "load", "const", "bi", "fused", "uni", "save"
    dst: Optional[Val]
    args: tuple = ()
    op: str = ""
    loc: int = 0                    # Mem index (load, save) or the constant's index



class Mir:
    def __init__(self, inputs: list[str], outputs: int, name: str):
        self.name = name
        self.inputs = list(inputs)
        self.n_out = outputs
        self.ops: list[_Op] = []
        self.consts: list[str] = []
        self.count = 0
        self.saved = set()

    def _new(self) -> Val:
        self.count += 1
        return Val(self.count - 1)

    # ---- values ----
    def input(self, name: str) -> Val:
        v = self._new()
        self.ops.append(_Op("load", v, loc=self.inputs.index(name)))
        return v

    def const(self, x: float) -> Val:
        b = bits(x)
        if b not in self.consts:
            self.consts.append(b)
        v = self._new()
        self.ops.append(_Op("const", v, loc=self.consts.index(b)))
        return v

    def bi(self, op: str, a: Val, b: Val) -> Val:
        v = self._new()
        self.ops.append(_Op("bi", v, (a, b), op))
        return v

    def add(self, a, b): return self.bi("Plus", a, b)
    def sub(self, a, b): return self.bi("Minus", a, b)
    def mul(self, a, b): return self.bi("Times", a, b)
    def div(self, a, b): return self.bi("Divide", a, b)

    def fused(self, op: str, a: Val, b: Val, c: Val) -> Val:
        v = self._new()
        self.ops.append(_Op("fused", v, (a, b, c), op))
        return v

    def fma(self, a, b, c): return self.fused("MulAdd", a, b, c)       # a b + c
    def fms(self, a, b, c): return self.fused("MulSub", a, b, c)       # a b - c

    def uni(self, op: str, a: Val) -> Val:
        v = self._new()
        self.ops.append(_Op("uni", v, (a,), op))
        return v

    def neg(self, a): return self.uni("Neg", a)
    def sqrt(self, a): return self.uni("Root", a)
    def sign(self, a): return self.uni("Sign", a)              # the sign bit (x And -0.0)

    # masks: all bits set (true) or none, per lane
    def lt(self, a, b): return self.bi("LittleThan", a, b)
    def le(self, a, b): return self.bi("LittleThanEqual", a, b)
    def gt(self, a, b): return self.bi("GreaterThan", a, b)
    def eq(self, a, b): return self.bi("Equal", a, b)
    def ne(self, a, b): return self.bi("NotEqual", a, b)
    def and_(self, a, b): return self.bi("And", a, b)
    def or_(self, a, b): return self.bi("Or", a, b)
    def xor(self, a, b): return self.bi("Xor", a, b)

    def select(self, mask: Val, t: Val, f: Val) -> Val:
        """t where mask is set, else f (symjit's IfElse: the mask in a Stack slot)"""
        if t == f:
            return t
        v = self._new()
        self.ops.append(_Op("ifelse", v, (mask, t, f)))
        return v

    def output(self, k: int, v: Val) -> None:
        if not 0 <= k < self.n_out or k in self.saved:
            raise ValueError(f"{self.name}: output {k}")
        self.saved.add(k)
        self.ops.append(_Op("save", None, (v,), loc=len(self.inputs) + k))

    # ---- allocation and MIR ----
    def fragment(self) -> dict:
        if self.saved != set(range(self.n_out)):
            raise ValueError(f"{self.name}: outputs {sorted(set(range(self.n_out)) - self.saved)} not written")
        # dead values dropped: an op is kept when it stores an output or a kept op reads its value
        live, kept = set(), []
        for o in reversed(self.ops):
            if o.kind == "save" or o.dst in live:
                kept.append(o)
                live.update(o.args)
        ops = kept[::-1]
        uses: dict[Val, list[int]] = {}
        for i, o in enumerate(ops):
            for a in o.args:
                uses.setdefault(a, []).append(i)
        reg: dict[Val, int] = {}          # value -> Gen
        slot: dict[Val, int] = {}         # value -> Stack, once spilled
        free_slots: list[int] = []
        nslots = 0
        holder: dict[int, Val] = {}       # Gen -> value
        code = []

        def next_use(v, i):
            """the first op after op i that reads v (None: none)"""
            for u in uses.get(v, ()):
                if u > i:
                    return u
            return None

        def take(i, keep):
            """a Gen for op i: a free one, else the one whose value is read furthest ahead (not in
            `keep`), its value saved to a Stack slot first (once: a value's slot keeps it)"""
            nonlocal nslots
            for g in range(GENS):
                if g not in holder:
                    return g
            far, gv = -1, None
            for g, v in holder.items():
                if v in keep:
                    continue
                n = next_use(v, i)
                n = 1 << 30 if n is None else n
                if n > far:
                    far, gv = n, g
            v = holder.pop(gv)
            del reg[v]
            if v not in slot:
                if free_slots:
                    s = free_slots.pop()
                else:
                    s, nslots = nslots, nslots + 1
                slot[v] = s
                code.append({"Save": {"loc": {"Stack": s}, "src": {"Gen": gv}}})
            return gv

        def get(v, i, keep):
            if v in reg:
                return reg[v]
            if v not in slot:
                raise ValueError(f"{self.name}: value {v.id} used before it is defined")
            g = take(i, keep)
            code.append({"Load": {"dst": {"Gen": g}, "loc": {"Stack": slot[v]}}})
            holder[g], reg[v] = v, g
            return g

        def in_slot(v, i):
            """v's Stack slot, saving it there first if it is only in a register"""
            nonlocal nslots
            if v not in slot:
                if free_slots:
                    s = free_slots.pop()
                else:
                    s, nslots = nslots, nslots + 1
                slot[v] = s
                code.append({"Save": {"loc": {"Stack": s}, "src": {"Gen": get(v, i, {v})}}})
            return slot[v]

        for i, o in enumerate(ops):
            keep = set(o.args)
            cond = in_slot(o.args[0], i) if o.kind == "ifelse" else None
            srcs = [get(a, i, keep) for a in (o.args[1:] if o.kind == "ifelse" else o.args)]
            dead_slots = []
            for a in keep:                    # arguments read for the last time: their registers are free
                if next_use(a, i) is None:    # (the result may take one), their slots after the op
                    if a in reg:
                        del holder[reg.pop(a)]
                    if a in slot:
                        dead_slots.append(slot.pop(a))
            if o.kind == "save":
                code.append({"Save": {"loc": {"Mem": o.loc}, "src": {"Gen": srcs[0]}}})
                free_slots.extend(dead_slots)
                continue
            d = take(i, set())
            holder[d], reg[o.dst] = o.dst, d
            if o.kind == "load":
                code.append({"Load": {"dst": {"Gen": d}, "loc": {"Mem": o.loc}}})
            elif o.kind == "const":
                code.append({"LoadConst": {"dst": {"Gen": d}, "idx": o.loc}})
            elif o.kind == "bi":
                code.append({"Bi": {"op": o.op, "dst": {"Gen": d}, "s1": {"Gen": srcs[0]}, "s2": {"Gen": srcs[1]}}})
            elif o.kind == "ifelse":
                code.append({"IfElse": {"dst": {"Gen": d}, "true_val": {"Gen": srcs[0]}, "false_val": {"Gen": srcs[1]},
                                        "cond": {"Stack": cond}}})
            elif o.kind == "fused":
                code.append({"Fused": {"op": o.op, "dst": {"Gen": d}, "a": {"Gen": srcs[0]}, "b": {"Gen": srcs[1]},
                                       "c": {"Gen": srcs[2]}}})
            else:
                code.append({"Uni": {"op": o.op, "dst": {"Gen": d}, "s1": {"Gen": srcs[0]}}})
            free_slots.extend(dead_slots)
        return {"format": "pacrunner-simd-fragment", "format_version": 1, "name": self.name, "symjit_version": None,
                "inputs": self.inputs, "outputs": self.n_out, "count_stack": nslots + 2, "consts": self.consts,
                "code": code}


@dataclass(frozen=True)
class dd:
    hi: Val
    lo: Val


class DD:
    """double-double on a Mir: df64.h, operation for operation"""

    def __init__(self, m: Mir):
        self.m = m

    def input(self, name: str) -> dd:
        return dd(self.m.input(name + ".hi"), self.m.input(name + ".lo"))

    def exact(self, x: float) -> dd:
        """an exact constant (dd_f): (x, 0)"""
        return dd(self.m.const(x), self.m.const(0.0))

    def of(self, x: Val) -> dd:
        """a plain value widened exactly (dd_f)"""
        return dd(x, self.m.const(0.0))

    def to_plain(self, x: dd) -> Val:
        """dd_to_f: hi + lo, rounded once"""
        return self.m.add(x.hi, x.lo)

    def two_sum(self, a: Val, b: Val) -> dd:
        m = self.m
        s = m.add(a, b)
        bb = m.sub(s, a)
        return dd(s, m.add(m.sub(a, m.sub(s, bb)), m.sub(b, bb)))

    def fast_two_sum(self, a: Val, b: Val) -> dd:
        m = self.m
        s = m.add(a, b)
        return dd(s, m.sub(b, m.sub(s, a)))

    def two_prod(self, a: Val, b: Val) -> dd:
        m = self.m
        p = m.mul(a, b)
        return dd(p, m.fms(a, b, p))

    def neg(self, x: dd) -> dd:
        return dd(self.m.neg(x.hi), self.m.neg(x.lo))

    def add(self, x: dd, y: dd) -> dd:                   # AccurateDWPlusDW
        m = self.m
        s, t = self.two_sum(x.hi, y.hi), self.two_sum(x.lo, y.lo)
        v = self.fast_two_sum(s.hi, m.add(s.lo, t.hi))
        return self.fast_two_sum(v.hi, m.add(t.lo, v.lo))

    def sub(self, x: dd, y: dd) -> dd:                   # x + (-y)
        return self.add(x, self.neg(y))

    def mul(self, x: dd, y: dd) -> dd:                   # DWTimesDW3
        m = self.m
        ch = m.mul(x.hi, y.hi)
        cl1 = m.fms(x.hi, y.hi, ch)
        tl0 = m.mul(x.lo, y.lo)
        tl1 = m.fma(x.hi, y.lo, tl0)
        cl2 = m.fma(x.lo, y.hi, tl1)
        return self.fast_two_sum(ch, m.add(cl1, cl2))

    def div(self, x: dd, y: dd) -> dd:                   # DWDivDW2
        m = self.m
        th = m.div(x.hi, y.hi)
        rh = m.mul(y.hi, th)
        rl1 = m.fms(y.hi, th, rh)
        rl3 = m.fma(y.lo, th, rl1)
        r = self.fast_two_sum(rh, rl3)
        pi_h = m.sub(x.hi, r.hi)
        delta = m.add(pi_h, m.sub(x.lo, r.lo))
        return self.fast_two_sum(th, m.div(delta, y.hi))


class DDX(DD):
    """DD with df64.h's comparisons, selects and the functions the wavefunctions use (masks per lane:
    df64.h's branches become selects, both sides computed)"""

    def zero(self) -> dd:
        return self.exact(0.0)

    def select(self, mask: Val, x: dd, y: dd) -> dd:
        return dd(self.m.select(mask, x.hi, y.hi), self.m.select(mask, x.lo, y.lo))

    # comparisons (hi decides, lo breaks ties)
    def lt(self, a: dd, b: dd) -> Val:
        m = self.m
        return m.or_(m.lt(a.hi, b.hi), m.and_(m.eq(a.hi, b.hi), m.lt(a.lo, b.lo)))

    def le(self, a: dd, b: dd) -> Val:
        m = self.m
        return m.or_(m.lt(a.hi, b.hi), m.and_(m.eq(a.hi, b.hi), m.le(a.lo, b.lo)))

    def gt(self, a: dd, b: dd) -> Val:
        return self.lt(b, a)

    def ne(self, a: dd, b: dd) -> Val:
        m = self.m
        return m.or_(m.ne(a.hi, b.hi), m.ne(a.lo, b.lo))

    def is_zero(self, x: dd) -> Val:
        m = self.m
        z = m.const(0.0)
        return m.and_(m.eq(x.hi, z), m.eq(x.lo, z))

    def isnan(self, x: dd) -> Val:
        m = self.m
        return m.or_(m.ne(x.hi, x.hi), m.ne(x.lo, x.lo))

    def signbit_mask(self, x: dd) -> Val:
        """dd_signbit as a mask, for a number (not NaN): hi < 0, or hi = -0 (1 / hi < 0)"""
        m = self.m
        z = m.const(0.0)
        return m.or_(m.lt(x.hi, z), m.and_(m.eq(x.hi, z), m.lt(m.div(m.const(1.0), x.hi), z)))

    # sign-bit operations: exact, as df64.h's negations
    def abs(self, x: dd) -> dd:
        s = self.m.sign(x.hi)
        return dd(self.m.xor(x.hi, s), self.m.xor(x.lo, s))

    def copysign(self, x: dd, s: dd) -> dd:
        m = self.m
        d = m.xor(m.sign(x.hi), m.sign(s.hi))
        return dd(m.xor(x.hi, d), m.xor(x.lo, d))

    def sqrt(self, x: dd) -> dd:
        """Karp-Markstein: one correction of the plain root; (sqrt(hi), 0) where hi > 0 does not hold"""
        m = self.m
        s = m.sqrt(x.hi)
        r = self.sub(x, self.two_prod(s, s))
        good = self.fast_two_sum(s, m.div(r.hi, m.mul(m.const(2.0), s)))
        return self.select(m.gt(x.hi, m.const(0.0)), good, dd(s, m.const(0.0)))

    def max(self, x: dd, y: dd) -> dd:
        """Julia's max (NaN propagates as x + y; max(-0, 0) = 0)"""
        tie = self.select(self.signbit_mask(x), y, x)
        r = self.select(self.ne(x, y), self.select(self.gt(x, y), x, y), tie)
        return self.select(self.m.or_(self.isnan(x), self.isnan(y)), self.add(x, y), r)

    def min(self, x: dd, y: dd) -> dd:
        tie = self.select(self.signbit_mask(x), x, y)
        r = self.select(self.ne(x, y), self.select(self.lt(x, y), x, y), tie)
        return self.select(self.m.or_(self.isnan(x), self.isnan(y)), self.add(x, y), r)

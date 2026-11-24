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

    def output(self, k: int, v: Val) -> None:
        if not 0 <= k < self.n_out or k in self.saved:
            raise ValueError(f"{self.name}: output {k}")
        self.saved.add(k)
        self.ops.append(_Op("save", None, (v,), loc=len(self.inputs) + k))

    # ---- allocation and MIR ----
    def fragment(self) -> dict:
        if self.saved != set(range(self.n_out)):
            raise ValueError(f"{self.name}: outputs {sorted(set(range(self.n_out)) - self.saved)} not written")
        uses: dict[Val, list[int]] = {}
        for i, o in enumerate(self.ops):
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

        for i, o in enumerate(self.ops):
            keep = set(o.args)
            srcs = [get(a, i, keep) for a in o.args]
            for a in keep:                    # arguments read for the last time: their places are free
                if next_use(a, i) is None:
                    if a in reg:
                        del holder[reg.pop(a)]
                    if a in slot:
                        free_slots.append(slot.pop(a))
            if o.kind == "save":
                code.append({"Save": {"loc": {"Mem": o.loc}, "src": {"Gen": srcs[0]}}})
                continue
            if next_use(o.dst, i) is None:
                raise ValueError(f"{self.name}: value {o.dst.id} is never used")
            d = take(i, set())
            holder[d], reg[o.dst] = o.dst, d
            if o.kind == "load":
                code.append({"Load": {"dst": {"Gen": d}, "loc": {"Mem": o.loc}}})
            elif o.kind == "const":
                code.append({"LoadConst": {"dst": {"Gen": d}, "idx": o.loc}})
            elif o.kind == "bi":
                code.append({"Bi": {"op": o.op, "dst": {"Gen": d}, "s1": {"Gen": srcs[0]}, "s2": {"Gen": srcs[1]}}})
            elif o.kind == "fused":
                code.append({"Fused": {"op": o.op, "dst": {"Gen": d}, "a": {"Gen": srcs[0]}, "b": {"Gen": srcs[1]},
                                       "c": {"Gen": srcs[2]}}})
            else:
                code.append({"Uni": {"op": o.op, "dst": {"Gen": d}, "s1": {"Gen": srcs[0]}}})
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

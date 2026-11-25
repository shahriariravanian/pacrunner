"""The split kernels (plans/F32-KERNEL.md, "Split kernels"): a program as two SIMD kernels, A in f64 and B
in f32 (generated in f64 and downgraded by simd/downgrade.py), no double-double.

  A (SplitA): states the momenta (Mem 4 leg + j); outputs the transfers, in order:
      the forms (4 per form);
      1 / k·k per form of a propagator row (as the f64 propagator fragment computes it);
      each source's components (the wavefunction × the factor, or its embedding's entries; re, im);
      per model kernel row, its kernel's crossing values (simd/split_kernel.py) of its static inputs.
  B (SplitB): states the transfers (as f32: the wrapper rounds A's outputs); outputs the helicity
      outputs. The rows as compile_simd's Assembler writes them, with every constant folded in f32
      (rmul32 of rf: the Metal runner's f32 arithmetic), forms and sources read from the transfers,
      propagators × the transferred 1 / k·k, model kernels' B sides with their crossing values.

split_applications(prog) -> (A's application, B's application, transfers: the description of each).
"""
from __future__ import annotations

from ..simd import split_kernel
from ..simd.mir import Mir, Val
from ..uir import InputKind, Lowered, Opcode, PhysicalType, is_kernel, is_propagator, is_vertex, source_fields
from .compile_simd import PROPAGATOR, VERTEX, Assembler, called, fl32, library, rf, rmul32
from .common import kernel_info


class R(tuple):
    """pacops' rcplx: a pair of plain values with its operations"""

    m: Mir

    def __new__(cls, m, re, im):
        t = super().__new__(cls, (re, im))
        t.m = m
        return t

    @property
    def re(self): return self[0]

    @property
    def im(self): return self[1]

    def __add__(self, b): return R(self.m, self.m.add(self.re, b.re), self.m.add(self.im, b.im))       # radd
    def __sub__(self, b): return R(self.m, self.m.sub(self.re, b.re), self.m.sub(self.im, b.im))       # rsub
    def __neg__(self): return R(self.m, self.m.neg(self.re), self.m.neg(self.im))                      # rneg

    def __mul__(self, b):                                                                               # rmul
        m = self.m
        return R(m, m.sub(m.mul(self.re, b.re), m.mul(self.im, b.im)), m.add(m.mul(self.re, b.im), m.mul(self.im, b.re)))

    def mulr(self, x: Val):                                                                             # rmulr
        return R(self.m, self.m.mul(self.re, x), self.m.mul(self.im, x))

    def i(self):                                                                                        # rmul_i
        return R(self.m, self.m.neg(self.im), self.re)


class Plan:
    """the transfers of a program (A's outputs, B's states), and the model kernels' splits"""

    def __init__(self, prog: Lowered):
        self.prog = prog
        self.items = []                 # (kind, key, j): what each transfer is
        self.index = {}
        self.splits = {}                # (sha, kinds) -> Split
        p = prog
        for f in range(len(p.forms)):
            for j in range(4):
                self.add(("form", f, j))
        for op in p.ops:
            if is_propagator(op.op):
                self.add(("rnorm", p.edge_form(op.d), 0))
        for r, op in enumerate(p.ops):
            if op.op == Opcode.SOURCE:
                for (comp, _) in self.source_components(op):
                    self.add(("source", r, 2 * comp))
                    self.add(("source", r, 2 * comp + 1))
            elif is_kernel(op.op):
                sp, key = self.kernel_split(op)
                for j in range(len(sp.crossing)):
                    self.add(("kernel", key, j))

    def add(self, item):
        if item not in self.index:
            self.index[item] = len(self.items)
            self.items.append(item)
        return self.index[item]

    def source_components(self, op):
        """the components a source row writes: (component, wave component or None)"""
        (family, _, _, _) = source_fields(op)
        n = {PhysicalType.SCALAR: 1, PhysicalType.WEYL: 2}.get(family, 4)
        if op.l is None:
            return [(j, j) for j in range(n)]
        return [(comp, wv) for (comp, wv, _) in self.prog.embeddings[op.l]]

    def kernel_split(self, op):
        """the row's kernel split, and its key: the kernel and its static inputs (forms, scales, factor)"""
        p = self.prog
        (s, set_) = p.instances[op.t]
        stub = p.stubs[s]
        kinds = tuple("current" if i.kind in (InputKind.IN_L, InputKind.IN_R, InputKind.OUT) else "static"
                      for i in stub.inputs)
        k = (stub.kernel.sha256, kinds)
        if k not in self.splits:
            info = kernel_info(stub.kernel)
            self.splits[k] = split_kernel.split(stub.kernel.name, info.mir, list(kinds), info.inputs, info.outputs)
        static = []
        for i in stub.inputs:
            if i.kind == InputKind.MOMENTUM_U:
                static.append(("u", p.edge_form(op.l), i.n))
            elif i.kind == InputKind.MOMENTUM_V:
                static.append(("v", p.edge_form(op.r), i.n))
            elif i.kind == InputKind.SCALE:
                static.append(("s", p.scales[p.scalesets[set_][i.n]].real))
            elif i.kind == InputKind.FACTOR:
                z = p.factors[op.x]
                static.append(("f", z.imag if i.im else z.real))
        return self.splits[k], (k, tuple(static))


def _rnorm() -> dict:
    """1 / (((k0 k0 - k1 k1) - k2 k2) - k3 k3), as the f64 propagator fragment computes it"""
    m = Mir([f"k{j}" for j in range(4)], 1, "rnorm")
    k = [m.input(f"k{j}") for j in range(4)]
    norm = m.sub(m.sub(m.sub(m.mul(k[0], k[0]), m.mul(k[1], k[1])), m.mul(k[2], k[2])), m.mul(k[3], k[3]))
    m.output(0, m.div(m.const(1.0), norm))
    return m.fragment()


def _propagator_split(kind: str) -> dict:
    """the f64 propagator fragment (simd/library.py) with its 1 / k·k an input (r) and k rounded (kr)"""
    n = 4 if kind == "vector" else 2
    names = [f"c{j}.{q}" for j in range(n) for q in ("re", "im")] + [f"k{j}" for j in range(4)] + ["r", "f.re", "f.im"]
    m = Mir(names, 2 * n, f"propagator:{kind}:split")
    c = [R(m, m.input(f"c{j}.re"), m.input(f"c{j}.im")) for j in range(n)]
    k = [m.input(f"k{j}") for j in range(4)]
    s = R(m, m.input("f.re"), m.input("f.im")).mulr(m.input("r"))
    if kind == "vector":
        out = [cj * s for cj in c]
    else:
        c0, c1 = c
        if kind == "weyl_a":
            o0 = ((-(c0.mulr(k[3]).i()) - c1.mulr(k[1]).i()) + c1.mulr(k[2])) + c0.mulr(k[0]).i()
            o1 = ((-(c0.mulr(k[2])) - c0.mulr(k[1]).i()) + c1.mulr(k[0]).i()) + c1.mulr(k[3]).i()
        else:
            o0 = ((-(c1.mulr(k[2])) + c0.mulr(k[0]).i()) + c0.mulr(k[3]).i()) + c1.mulr(k[1]).i()
            o1 = ((-(c1.mulr(k[3]).i()) + c0.mulr(k[2])) + c0.mulr(k[1]).i()) + c1.mulr(k[0]).i()
        out = [o0 * s, o1 * s]
    for j, z in enumerate(out):
        m.output(2 * j, z.re)
        m.output(2 * j + 1, z.im)
    return m.fragment()


def a_called(name: str) -> bool:
    """A's subroutines: the wavefunctions (called()'s); the kernels' A sides are a few instructions,
    inlined (one may take no argument: symjit's LoadArgs needs at least one)"""
    return called(name) and not name.endswith(":A")


class SplitA(Assembler):
    """kernel A (f64): the forms, then each transfer into its output"""

    def __init__(self, prog: Lowered, plan: Plan):
        super().__init__(prog, a_called)
        self.plan = plan

    def count_obs(self) -> int:
        return len(self.plan.items)

    def out(self, h):
        return {"Mem": 4 * self.prog.legs + h}

    def frag(self, name):
        if name == "rnorm":
            if name not in self.frags:
                self.frags[name] = _rnorm()
            return self.frags[name]
        if name.startswith("kernel:") and name.endswith(":A"):
            return self.frags[name]
        return super().frag(name)

    def assemble(self):
        p, b, plan = self.prog, self.b, self.plan
        names = {self.frag_name(op) for op in p.ops if op.op == Opcode.SOURCE} - {None}
        for (k, sp) in plan.splits.items():
            self.frags[f"kernel:{k[0]}:{'.'.join(c[0] for c in k[1])}:A"] = sp.a_fragment()
        stacks = [self.frag(n)["count_stack"] for n in names | {"rnorm"} | {n for n in self.frags if n.endswith(":A")}]
        self.T = max([80, *stacks])
        self.zero([("arena", i) for i in range(b.size)])
        for i, f in enumerate(p.forms):          # as Assembler.assemble: 0 + coef × (sign ×) leg
            for j in range(4):
                self.loadc(0, 0.0)
                for (leg, coef) in f.terms:
                    if p.crossing is None:
                        self.load(1, self.momentum(leg, j))
                    else:
                        (cl, sign) = p.crossing[leg]
                        self.load(1, self.momentum(cl, j))
                        self.cmath("Times", 1, 1, float(sign))
                    self.cmath("Times", 1, 1, float(coef))
                    self.bi("Plus", 0, 0, 1)
                self.save(self.form(i, j), 0)
        done = set()
        for t, item in enumerate(plan.items):
            kind = item[0]
            if kind == "form":
                self.load(0, self.form(item[1], item[2]))
                self.save(self.out(t), 0)
            elif kind == "rnorm":
                self.inline("rnorm", {f"k{j}": self.form(item[1], j) for j in range(4)}, [self.out(t)])
        for r, op in enumerate(p.ops):
            if op.op == Opcode.SOURCE:          # Assembler.source into the arena, then each component out
                self.source(op, self.frag_name(op))
                for (comp, _) in plan.source_components(op):
                    re, im = self.edge(op.d, comp)
                    for q, x in ((0, re), (1, im)):
                        self.load(0, x)
                        self.save(self.out(plan.index[("source", r, 2 * comp + q)]), 0)
            elif is_kernel(op.op):
                sp, key = plan.kernel_split(op)
                if key in done or not sp.crossing:
                    continue
                done.add(key)
                k, static = key
                ins, si = {}, 0
                for kx, kind in enumerate(k[1]):
                    if kind != "static":
                        continue
                    s = static[si]
                    si += 1
                    ins[f"x{kx}"] = self.form(s[1], s[2]) if s[0] in ("u", "v") else float(s[1])
                name = f"kernel:{k[0]}:{'.'.join(c[0] for c in k[1])}:A"
                outs = [self.out(plan.index[("kernel", key, j)]) for j in range(len(sp.crossing))]
                self.inline(name, ins, outs)
            self.rows_done += 1
        self.finish()
        return self.application()


class SplitB(Assembler):
    """kernel B (f32 after the downgrade): the rows on the transfers, every constant folded in f32"""

    def __init__(self, prog: Lowered, plan: Plan):
        super().__init__(prog, called)
        self.plan = plan
        self.extra = len(plan.items)    # the transfers copied to Stack slots: a subroutine's arguments
                                        # are slot numbers (LoadArgs ultra), not Mem

    def count_states(self) -> int:
        return len(self.plan.items)

    def out(self, h):
        return {"Mem": len(self.plan.items) + h}

    def transfer(self, item):
        return ("extra", self.plan.index[item])

    def form(self, f, j):
        return self.transfer(("form", f, j))

    def row_factor(self, x) -> complex:
        return rf(self.prog.factors[x])

    def frag(self, name):
        if name.endswith(":split"):
            if name not in self.frags:
                self.frags[name] = _propagator_split(name.split(":")[1])
            return self.frags[name]
        if name.startswith("kernel:") and name.endswith(":B"):
            return self.frags[name]
        return super().frag(name)

    def frag_name(self, op):
        p, o = self.prog, op.op
        if is_propagator(o):
            return f"propagator:{PROPAGATOR[o]}:split"
        if o == Opcode.SOURCE:
            return None
        if is_kernel(o):
            sp, key = self.plan.kernel_split(op)
            k = key[0]
            name = f"kernel:{k[0]}:{'.'.join(c[0] for c in k[1])}:B"
            if name not in self.frags:
                self.frags[name] = sp.b_fragment()
            return name
        return super().frag_name(op)

    def assemble(self):
        p, b = self.prog, self.b
        stacks = [self.frag(n)["count_stack"] for n in self.needed()]
        self.T = max([80, *stacks])
        self.zero([("arena", i) for i in range(b.size)])
        self.zero([self.out(h) for h in range(p.output_count())])
        for t in range(len(self.plan.items)):
            self.load(0, {"Mem": t})
            self.save(("extra", t), 0)
        for r, op in enumerate(p.ops):
            self.r = r
            self.row(op)
            self.rows_done += 1
        (num, den) = p.group_factor
        if num != den:
            for h in range(p.output_count()):
                self.load(0, self.out(h))
                self.cmath("Times", 0, 0, float(num))
                self.cmath("Divide", 0, 0, float(den))
                self.save(self.out(h), 0)
        self.finish()
        return self.application()

    def row(self, op):
        p, o = self.prog, op.op
        name = self.frag_name(op)
        if is_vertex(o):
            nl, nr, no = library.VERTICES[VERTEX[o]]
            ins = {**self.current_inputs("l", op.l, nl), **self.current_inputs("r", op.r, nr)}
            s = rmul32(rf(p.scales[op.t]), rf(p.factors[op.x]))
            ins["s.re"], ins["s.im"] = s.real, s.imag
            if o == Opcode.V_THREE_VECTOR:
                fu, fv = p.edge_form(op.l), p.edge_form(op.r)
                for j in range(4):
                    ins[f"u{j}"], ins[f"v{j}"] = self.form(fu, j), self.form(fv, j)
                ins["two"] = 2.0
            if not op.set:
                ins.update(self.current_inputs("old", op.d, no))
            self.inline(name, ins, [t for j in range(no) for t in self.edge(op.d, j)])
        elif is_propagator(o):
            n = 4 if o == Opcode.P_VECTOR else 2
            ins = self.current_inputs("c", op.d, n)
            fd = p.edge_form(op.d)
            for j in range(4):
                ins[f"k{j}"] = self.form(fd, j)
            ins["r"] = self.transfer(("rnorm", fd, 0))
            f = rf(p.factors[op.x])
            if o == Opcode.P_VECTOR:
                f = rmul32(f, complex(0.0, -1.0))
            ins["f.re"], ins["f.im"] = f.real, f.imag
            self.inline(name, ins, [t for j in range(n) for t in self.edge(op.d, j)])
        elif o == Opcode.SOURCE:
            for (comp, wv) in self.plan.source_components(op):
                re, im = self.edge(op.d, comp)
                for q, x in ((0, re), (1, im)):
                    self.load(0, self.transfer(("source", self.r, 2 * comp + q)))
                    self.save(x, 0)
        elif o == Opcode.CLOSURE:
            entries = p.lists[op.t]
            if not entries:
                return
            n = len(entries)
            ins = {**self.current_inputs("l", op.l, n), **self.current_inputs("r", op.r, n)}
            f = rf(p.factors[op.x])
            for j, x in enumerate(entries):
                c = rmul32(rf(p.factors[x]), f)
                ins[f"c{j}.re"], ins[f"c{j}.im"] = c.real, c.imag
            ins["amp.re"], ins["amp.im"] = self.amp(op.d)
            self.inline(name, ins, list(self.amp(op.d)))
        elif o == Opcode.REDUCE:
            (are, aim) = self.amp(op.t)
            self.load(0, are)
            self.load(1, aim)
            if op.x is not None:
                s = rf(p.factors[op.x])
                self.cmath("Times", 2, 0, s.real)
                self.cmath("Times", 3, 1, s.imag)
                self.bi("Minus", 2, 2, 3)
                self.cmath("Times", 3, 0, s.imag)
                self.cmath("Times", 4, 1, s.real)
                self.bi("Plus", 3, 3, 4)
            else:
                self.emit({"Mov": {"dst": {"Gen": 2}, "s1": {"Gen": 0}}})
                self.emit({"Mov": {"dst": {"Gen": 3}, "s1": {"Gen": 1}}})
            self.bi("Times", 4, 3, 3)
            self.emit({"Fused": {"op": "MulAdd", "dst": {"Gen": 4}, "a": {"Gen": 2}, "b": {"Gen": 2}, "c": {"Gen": 4}}})
            self.cmath("Times", 4, 4, fl32(p.weights[op.n]))
            self.load(5, self.out(op.d))
            self.bi("Plus", 5, 5, 4)
            self.save(self.out(op.d), 5)
        elif is_kernel(o):
            self.split_kernel_row(op, name)
        else:
            super().row(op)

    def split_kernel_row(self, op, name):
        p, o = self.prog, op.op
        sp, key = self.plan.kernel_split(op)
        (s, set_) = p.instances[op.t]
        stub = p.stubs[s]
        ins = {}
        for k, i in enumerate(stub.inputs):
            im = 1 if i.im else 0
            if i.kind in (InputKind.IN_L, InputKind.IN_R, InputKind.OUT):
                e = {InputKind.IN_L: op.l, InputKind.IN_R: op.r, InputKind.OUT: op.d}[i.kind]
                ins[f"x{k}"] = ("arena", self.b.edge_at(p, e) + 2 * i.n + im)
            elif i.kind == InputKind.MOMENTUM_U:
                ins[f"x{k}"] = self.form(p.edge_form(op.l), i.n)
            elif i.kind == InputKind.MOMENTUM_V:
                ins[f"x{k}"] = self.form(p.edge_form(op.r), i.n)
            elif i.kind == InputKind.SCALE:
                ins[f"x{k}"] = fl32(p.scales[p.scalesets[set_][i.n]].real)
            else:
                z = p.factors[op.x]
                ins[f"x{k}"] = fl32(z.imag if im else z.real)
        for j in range(len(sp.crossing)):
            ins[f"a{j}"] = self.transfer(("kernel", key, j))
        d = self.frag(name)
        res = [("scratch", j) for j in range(d["outputs"])]
        self.scratch_used = max(self.scratch_used, d["outputs"])
        self.inline(name, ins, res)
        f = rf(p.factors[op.x])
        for n, comp in enumerate(stub.outputs):
            dre, dim = self.edge(op.d, comp)
            if o == Opcode.KERNEL_DIRECT or f == complex(1.0, 0.0):
                self.load(0, res[2 * n])
                self.load(1, res[2 * n + 1])
            else:
                self.cmul_const(res[2 * n], res[2 * n + 1], f, a_first=True)
            if not op.set:
                self.load(2, dre)
                self.load(3, dim)
                self.bi("Plus", 0, 2, 0)
                self.bi("Plus", 1, 3, 1)
            self.save(dre, 0)
            self.save(dim, 1)


def split_applications(prog: Lowered):
    """(A's application, B's application with its constants' roles, the transfers)"""
    plan = Plan(prog)
    a = SplitA(prog, plan)
    app_a = a.assemble()
    b = SplitB(prog, plan)
    app_b = b.assemble()
    return app_a, app_b, b.roles, plan.items

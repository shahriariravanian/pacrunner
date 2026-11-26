"""compile_simd(path, l): a program as one symjit kernel, for device "simd": the program's rows spliced
from the library of pacops operations (pacrunner/simd/: symjit fragments, pacops.c's bits) into one
symjit application (symjit's "symjit-application" JSON), compiled by symjit's JIT into its SIMD kernels,
AVX2 (4 points a call) and AVX-512 (8, where the building CPU has it), whose machine code simd/elf.py wraps
in object files (no relocations: the kernels call no function). The values are the straight-line C
runner's (codegen/compile.py) bit for bit.

The kernel: its states are the caller's momenta (E, px, py, pz per leg: Mem 0 .. 4 × legs), its outputs
the helicity outputs (Mem 4 × legs ..). runner.c's arena (forms, amplitudes, planes: ArenaIndex) is in
symjit stack slots, after the fragments' own slots, then a scratch area, the per-row constants and the
result area of the called fragments. The rows:

  - vertices, propagators, wavefunctions and model kernels are compress-mode subroutines (symjit's
    `topology.rs` and `statement.rs` layout): one body each after `@success`, entered at
    `@f_<name>_ultra` (SaveArgs ultra: the first 16 arguments passed as their stack slots' numbers,
    packed four to a register, and gathered into the argument area Stack(16 + k)), its outputs written
    to the result area and copied to their currents by the caller. A call site is LoadArgs (ultra),
    Call, the copies. An accumulation run (vertices adding into the current the call before wrote) is
    chained: the run's calls leave the sum in the result area and the next takes it from there (its body
    reads its old inputs at the result slots), so only the run's last call copies it out (Assembler.chain). The body's inputs that are the same constant at every call site are constants
    of the body; the per-row ones are stack slots filled at the start (ultra takes stack slots only).
    So the kernel's code stays small (straight-line code of a large program would be fetched from L3
    for every point: pacrunner's plans/CPU-KERNEL.md).
  - closures are inlined (their library fragments); forms, clears, copies, scatters, scales, source
    factors (plain and embedded), the reduction and the group factor are written here as MIR, in
    pacops' order (pacops.c is the reference).
  - a model kernel (KERNEL rows) is its application's MIR of real operations (kernels/mir.py's
    json-complex) rewritten for a real application as kernel_c.py runs it: LoadComplex/SaveComplex as
    two loads/stores, ComplexBi × and ÷ as kernel_c's fused sequences (through Gen(2), Gen(3), Temp),
    Real/Conjugate/Imaginary/Abs2 as their real meaning; its funclets stay (internal calls).

The folder: kernel.json (the application), runner.h (the program's sizes and names, as
compile.py's), and per lane count L in `lanes` kernel<L>.o and kernel<L>.h (the kernel pac_simd<L>).
"""
from __future__ import annotations

import copy
import json
import math
import struct
from pathlib import Path
from typing import Optional, Sequence

from ..simd import library
from ..uir import (InputKind, Lowered, Opcode, PhysicalType, is_kernel, is_propagator, is_vertex, source_fields,
                   validate)
from .common import ArenaIndex, CompileError, cstring

SQH = math.sqrt(0.5)
VERTEX = {Opcode.V_WEYL_VECTOR_A: "weyl_vector_a", Opcode.V_WEYL_VECTOR_B: "weyl_vector_b",
          Opcode.V_TENSOR_VECTOR: "tensor_vector", Opcode.V_VECTOR_WEDGE: "vector_wedge",
          Opcode.V_THREE_VECTOR: "three_vector"}
PROPAGATOR = {Opcode.P_WEYL_A: "weyl_a", Opcode.P_WEYL_B: "weyl_b", Opcode.P_VECTOR: "vector"}
READS = {"Load", "LoadMath", "LoadComplex"}
WRITES = {"Save", "SaveComplex"}
ARG_AREA = 16        # config.location(k) = Stack(SPILL_AREA + k): the compress-mode argument area
ARGS_CAP = 64        # COMPRESSED_ARGS_CAP: a body's arguments
PACKED = 16          # the arguments passed by slot number (the x86-64 entry: 16 vector registers)
LANES = (4, 8)


class SimdError(CompileError):
    pass


def bits(x: float) -> str:
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def ckey(x) -> object:
    """a constant's key in the pool: its bits"""
    return bits(x)


def cmul(a: complex, b: complex) -> complex:
    """pacops mul(a, b) on doubles (Python floats are IEEE doubles, no fma)"""
    return complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)


def G(n):
    return {"Gen": n}


def called(name: str) -> bool:
    """the fragments that are subroutines (the rest is inlined)"""
    return name.startswith(("vertex:", "propagator:", "wave:", "kernel:"))


# ---- model kernels: real MIR -------------------------------------------------------------------------------

def _loc_plus(loc: dict, k: int) -> dict:
    ((kind, i),) = loc.items()
    return {kind: i + k}


def kernel_fragment(k) -> dict:
    """a model kernel (uir Kernel) as a library-style fragment: its application's real MIR (json-complex)
    rewritten for a real application (see the module); inputs x0 .. (the stub's), outputs its pairs"""
    import symjit
    app = json.loads(symjit.mir_json(k.application, "json-complex"))
    prog = app["program"]
    consts = list(app["mir"]["consts"])

    def const(x):
        b = bits(x)
        if b not in consts:
            consts.append(b)
        return consts.index(b)

    out = []
    for ins in app["mir"]["code"]:
        if isinstance(ins, str):
            if ins in ("Nop", "End"):
                continue
            raise SimdError(f"kernel {k.name}: the instruction {ins}")
        ((ty, b),) = ins.items()
        if ty == "LoadComplex":
            out += [{"Load": {"dst": b["xd"], "loc": b["loc"]}}, {"Load": {"dst": b["yd"], "loc": _loc_plus(b["loc"], 1)}}]
        elif ty == "SaveComplex":
            out += [{"Save": {"loc": b["loc"], "src": b["xs"]}}, {"Save": {"loc": _loc_plus(b["loc"], 1), "src": b["ys"]}}]
        elif ty == "ComplexBi":
            xd, yd, x1, y1, x2, y2 = (b[f] for f in ("xd", "yd", "x1", "y1", "x2", "y2"))
            if b["op"] in ("Plus", "Minus"):
                out += [{"Bi": {"op": b["op"], "dst": xd, "s1": x1, "s2": x2}},
                        {"Bi": {"op": b["op"], "dst": yd, "s1": y1, "s2": y2}}]
            elif b["op"] == "Times":     # kernel_c: xt = y1 y2; xt = fma(x1, x2, -xt); yt = x1 y2; yd = fma(x2, y1, yt); xd = xt
                xt, yt = G(2), G(3)
                out += [{"Bi": {"op": "Times", "dst": xt, "s1": y1, "s2": y2}},
                        {"Fused": {"op": "MulSub", "dst": xt, "a": x1, "b": x2, "c": xt}},
                        {"Bi": {"op": "Times", "dst": yt, "s1": x1, "s2": y2}},
                        {"Fused": {"op": "MulAdd", "dst": yd, "a": x2, "b": y1, "c": yt}},
                        {"Mov": {"dst": xd, "s1": xt}}]
            elif b["op"] == "Divide":    # kernel_c's ÷ (t: Temp)
                xt, yt, t = G(2), G(3), "Temp"
                out += [{"Bi": {"op": "Times", "dst": xt, "s1": y1, "s2": y2}},
                        {"Fused": {"op": "MulAdd", "dst": xt, "a": x1, "b": x2, "c": xt}},
                        {"Bi": {"op": "Times", "dst": yt, "s1": x1, "s2": y2}},
                        {"Fused": {"op": "MulSub", "dst": yt, "a": x2, "b": y1, "c": yt}},
                        {"Bi": {"op": "Times", "dst": t, "s1": x2, "s2": x2}},
                        {"Fused": {"op": "MulAdd", "dst": t, "a": y2, "b": y2, "c": t}},
                        {"Bi": {"op": "Divide", "dst": xd, "s1": xt, "s2": t}},
                        {"Bi": {"op": "Divide", "dst": yd, "s1": yt, "s2": t}}]
            else:
                raise SimdError(f"kernel {k.name}: the complex operation {b['op']}")
        elif ty == "Uni" and b["op"] in ("Real", "Conjugate"):
            out.append({"Mov": {"dst": b["dst"], "s1": b["s1"]}})
        elif ty == "Uni" and b["op"] == "Imaginary":
            out.append({"LoadConst": {"dst": b["dst"], "idx": const(0.0)}})
        elif ty == "Uni" and b["op"] == "Abs2":
            out.append({"Bi": {"op": "Times", "dst": b["dst"], "s1": b["s1"], "s2": b["s1"]}})
        elif ty == "Call" and b["num_args"] != 0:
            raise SimdError(f"kernel {k.name}: a call of {b['label']} (math functions are not supported)")
        elif ty in ("LoadArgs", "SaveArgs"):
            raise SimdError(f"kernel {k.name}: a compressed kernel")
        else:
            out.append({ty: b})
    text = json.dumps(out)
    if '"Static"' in text or '"Param"' in text:
        raise SimdError(f"kernel {k.name}: static registers or parameters")
    stack = [int(x) for x in __import__("re").findall(r'"Stack": (\d+)', text)]
    return {"name": f"kernel:{k.sha256}", "inputs": [f"x{i}" for i in range(prog["count_states"])],
            "outputs": prog["count_obs"], "count_stack": max([0, *stack]) + 2, "consts": consts, "code": out}


# ---- the assembler -----------------------------------------------------------------------------------------

class CallSite:
    def __init__(self, name, inputs, outputs):
        self.name, self.inputs, self.outputs = name, inputs, outputs
        self.keep = False       # chained: the outputs stay in the result area for the next call (no copies)


def old_inputs(inputs) -> list:
    """an add fragment's `old` inputs (old<j>.re, old<j>.im), in component order"""
    olds = [n for n in inputs if n.startswith("old")]
    return sorted(olds, key=lambda n: (int(n[3:].split(".")[0]), n.endswith(".im")))


class Assembler:
    """one program's kernel (see the module): rows() walks the program as compile.py's compile_op,
    finish() lays out the calls and the bodies, application() is the JSON"""

    def __init__(self, prog: Lowered, call=called):
        validate(prog)
        self.prog, self.b = prog, ArenaIndex.of(prog)
        self.code, self.consts, self.cidx = [], [], {}
        self.frags = {}
        self.rows_done = 0
        self.scratch_used = 0
        self.call = call
        self.sites = {}
        self.cslots = {}
        self.results = 0
        self.bodies = []
        self.T = None
        self.extra = 0          # slots after the arena (compile_split's kernel B: the transfers)

    # ---- slots and constants ----
    def const(self, x) -> int:
        k = ckey(x)
        if k not in self.cidx:
            self.cidx[k] = len(self.consts)
            self.consts.append(bits(x))
        return self.cidx[k]

    def count_states(self) -> int:
        return 4 * self.prog.legs

    def count_obs(self) -> int:
        return self.prog.output_count()

    def momentum(self, leg, j):
        return {"Mem": 4 * leg + j}

    def out(self, h):
        return {"Mem": 4 * self.prog.legs + h}

    def loc(self, t):
        if isinstance(t, dict):
            return t
        kind, i = t
        base = self.T
        if kind == "arena":
            return {"Stack": base + i}
        base += self.b.size
        if kind == "extra":
            return {"Stack": base + i}
        base += self.extra
        if kind == "scratch":
            return {"Stack": base + i}
        base += self.scratch_used
        if kind == "cslot":
            return {"Stack": base + i}
        base += len(self.cslots)
        if kind == "result":
            return {"Stack": base + i}
        raise KeyError(kind)

    def frag(self, name):
        if name not in self.frags:
            if name.startswith("kernel:"):
                k = next(s.kernel for s in self.prog.stubs if s.kernel.sha256 == name[len("kernel:"):])
                self.frags[name] = kernel_fragment(k)
            else:
                self.frags[name] = library.fragment(name)
        return self.frags[name]

    def row_factor(self, x) -> complex:
        """the factor x of a scale, copy or scatter row"""
        return self.prog.factors[x]

    # ---- raw MIR ----
    def emit(self, ins):
        self.code.append(ins)

    def load(self, r, t):
        self.emit({"Load": {"dst": G(r), "loc": self.loc(t)}})

    def save(self, t, r):
        self.emit({"Save": {"loc": self.loc(t), "src": G(r)}})

    def loadc(self, r, x):
        self.emit({"LoadConst": {"dst": G(r), "idx": self.const(x)}})

    def bi(self, op, d, a, b):
        self.emit({"Bi": {"op": op, "dst": G(d), "s1": G(a), "s2": G(b)}})

    def cmath(self, op, d, a, x):
        """d = a op x (x a constant)"""
        self.emit({"LoadConstMath": {"op": op, "dst": G(d), "s1": G(a), "idx": self.const(x)}})

    def zero(self, targets):
        self.loadc(0, 0.0)
        for t in targets:
            self.save(t, 0)

    def cmul_const(self, re, im, z: complex, a_first: bool):
        """(g0, g1) = mul(z, w) (a_first: mul(w, z)) of the value w at targets (re, im) and the constant z,
        in pacops' order: re = a.re b.re - a.im b.im, im = a.re b.im + a.im b.re"""
        self.load(4, re)
        self.load(5, im)
        if a_first:     # mul(w, z): w.re z.re - w.im z.im, w.re z.im + w.im z.re
            self.cmath("Times", 0, 4, z.real)
            self.cmath("Times", 1, 5, z.imag)
            self.bi("Minus", 0, 0, 1)
            self.cmath("Times", 1, 4, z.imag)
            self.cmath("Times", 2, 5, z.real)
            self.bi("Plus", 1, 1, 2)
        else:           # mul(z, w): z.re w.re - z.im w.im, z.re w.im + z.im w.re
            self.cmath("Times", 0, 4, z.real)
            self.cmath("Times", 1, 5, z.imag)
            self.bi("Minus", 0, 0, 1)
            self.cmath("Times", 1, 5, z.real)
            self.cmath("Times", 2, 4, z.imag)
            self.bi("Plus", 1, 1, 2)

    # ---- fragments ----
    def inline(self, name, inputs: dict, outputs: list):
        """the fragment `name` at this row: inputs name -> a target (("arena", i), {"Mem": k}) or a float
        constant; outputs: the targets of its outputs in order"""
        d = self.frag(name)
        names, nout = d["inputs"], d["outputs"]
        if len(outputs) != nout or set(names) - set(inputs):
            raise SimdError(f"{name}: {len(outputs)} outputs for {nout}, missing inputs {set(names) - set(inputs)}")
        if self.call(name):
            site = CallSite(name, inputs, list(outputs))
            self.code.append(site)
            self.sites.setdefault(name, []).append(site)
            return
        targets = [inputs[n] for n in names] + list(outputs)
        if not self._splice(d, targets, check=True):
            # an in-place hazard (an output read again after a write): the outputs to scratch, then copied
            scratch = [("scratch", j) for j in range(nout)]
            self.scratch_used = max(self.scratch_used, nout)
            self._splice(d, [inputs[n] for n in names] + scratch, check=False)
            for j, t in enumerate(outputs):
                self.load(0, ("scratch", j))
                self.save(t, 0)

    def _splice(self, d, targets, check, dest=None, prefix=None):
        nin = len(d["inputs"])
        cmap = {}
        for k, c in enumerate(d["consts"]):
            if c not in self.cidx:
                self.cidx[c] = len(self.consts)
                self.consts.append(c)
            cmap[k] = self.cidx[c]
        prefix = prefix or f"r{self.rows_done}_"
        out, written = [], set()
        for ins in d["code"]:
            if isinstance(ins, str):
                if ins in ("End", "Nop"):
                    continue
                out.append(ins)
                continue
            ((ty, body),) = ins.items()
            body = copy.deepcopy(body)
            if ty in ("LoadConst", "LoadConstMath"):
                body["idx"] = cmap[body["idx"]]
            if "label" in body and not body["label"].startswith("."):
                body["label"] = prefix + body["label"]
            for key in ("loc", "cond"):
                if isinstance(body.get(key), dict) and "Mem" in body[key]:
                    i = body[key]["Mem"]
                    t = targets[i]
                    if isinstance(t, float):
                        if ty == "Load":
                            ty, body = "LoadConst", {"dst": body["dst"], "idx": self.const(t)}
                        elif ty == "LoadMath":
                            ty, body = "LoadConstMath", {"op": body["op"], "dst": body["dst"], "s1": body["s1"],
                                                         "idx": self.const(t)}
                        else:
                            raise SimdError(f"{d['name']}: a constant input in {ty}")
                        break
                    key_t = tuple(self.loc(t).items())[0]
                    if (ty in READS or key == "cond") and check and i < nin and key_t in written:
                        return False
                    if ty in WRITES:
                        written.add(key_t)
                    body[key] = self.loc(t)
            out.append({ty: body})
        (self.code if dest is None else dest).extend(out)
        return True

    # ---- the program ----
    def edge(self, e, j):
        q = self.b.edge_at(self.prog, e) + 2 * j
        return ("arena", q), ("arena", q + 1)

    def form(self, f, j):
        return ("arena", self.b.form_at(f) + j)

    def amp(self, a):
        q = self.b.amplitude_at(a)
        return ("arena", q), ("arena", q + 1)

    def current_inputs(self, name, e, n):
        d = {}
        for j in range(n):
            d[f"{name}{j}.re"], d[f"{name}{j}.im"] = self.edge(e, j)
        return d

    def needed(self):
        names = {self.frag_name(op) for op in self.prog.ops}
        return sorted(n for n in names if n)

    def frag_name(self, op):
        p, o = self.prog, op.op
        if is_vertex(o):
            return f"vertex:{VERTEX[o]}:{'set' if op.set else 'add'}"
        if is_propagator(o):
            return f"propagator:{PROPAGATOR[o]}"
        if o == Opcode.CLOSURE:
            n = len(p.lists[op.t])
            return f"closure:{n}" if n else None
        if o == Opcode.SOURCE:
            return self.wave_name(op)
        if is_kernel(o):
            return f"kernel:{p.stubs[p.instances[op.t][0]].kernel.sha256}"
        return None

    def wave_name(self, op):
        (family, anti, h, ch) = source_fields(op)
        mass = self.prog.scales[op.t].real
        a = "anti" if anti else "part"
        if family == PhysicalType.WEYL:
            return f"wave:weyl:{a}:{h}:{ch}"
        if family == PhysicalType.DIRAC:
            return f"wave:dirac:{a}:{h}:{'massless' if abs(mass) < 1.0e-8 else 'massive'}"
        if family == PhysicalType.VECTOR:
            return f"wave:gluon:{h}" if mass == 0 else f"wave:massive_vector:{h}"
        if family == PhysicalType.SCALAR:
            return None
        raise SimdError(f"a source of family {family}")

    def assemble(self):
        p, b = self.prog, self.b
        stacks = [self.frag(n)["count_stack"] for n in self.needed()]
        self.T = max([80, *stacks])     # the fragments' own slots, from Stack(80) (16 + ARGS_CAP)
        # run(): the arena and the outputs at zero, then the forms
        self.zero([("arena", i) for i in range(b.size)])
        self.zero([self.out(h) for h in range(p.output_count())])
        for i, f in enumerate(p.forms):
            for j in range(4):
                self.loadc(0, 0.0)                      # 0.0 + coef × (sign × momenta[leg]), in term order
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
        for op in p.ops:
            self.row(op)
            self.rows_done += 1
        (num, den) = p.group_factor
        if num != den:                                  # group(): x × num ÷ den
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
            s = cmul(p.scales[op.t], p.factors[op.x])
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
            f = p.factors[op.x]
            if o == Opcode.P_VECTOR:
                f = cmul(f, complex(0.0, -1.0))
            ins["f.re"], ins["f.im"], ins["one"] = f.real, f.imag, 1.0
            self.inline(name, ins, [t for j in range(n) for t in self.edge(op.d, j)])
        elif o == Opcode.CLEAR:
            self.zero([t for j in range(op.n) for t in self.edge(op.d, j)])
        elif o == Opcode.CLEAR_AMPLITUDE:
            self.zero(list(self.amp(op.d)))
        elif o == Opcode.SCALE_ROW:          # c[j] = mul(f, c[j])
            f = self.row_factor(op.x)
            for j in range(op.n):
                re, im = self.edge(op.d, j)
                self.cmul_const(re, im, f, a_first=False)
                self.save(re, 0)
                self.save(im, 1)
        elif o == Opcode.COPY:               # out[j] = src[j] × f (0 for a zero factor)
            f = self.row_factor(op.x)
            n = p.edge_layout(op.d).components
            for j in range(n):
                dre, dim = self.edge(op.d, j)
                if f == 0:
                    self.zero([dre, dim])
                    continue
                sre, sim = self.edge(op.l, j)
                self.cmul_const(sre, sim, f, a_first=True)
                self.save(dre, 0)
                self.save(dim, 1)
        elif o == Opcode.SCATTER:
            self.scatter(op)
        elif o == Opcode.CLOSURE:
            entries = p.lists[op.t]
            if not entries:
                return
            n = len(entries)
            ins = {**self.current_inputs("l", op.l, n), **self.current_inputs("r", op.r, n)}
            f = p.factors[op.x]
            for j, x in enumerate(entries):
                c = cmul(p.factors[x], f)
                ins[f"c{j}.re"], ins[f"c{j}.im"] = c.real, c.imag
            ins["amp.re"], ins["amp.im"] = self.amp(op.d)
            self.inline(name, ins, list(self.amp(op.d)))
        elif o == Opcode.SOURCE:
            self.source(op, name)
        elif is_kernel(o):
            self.kernel_row(op, name)
        elif o == Opcode.REDUCE:
            # out[d] = out[d] + weight × fma(a.re, a.re, a.im × a.im), a = amplitude × scale
            (are, aim) = self.amp(op.t)
            self.load(0, are)
            self.load(1, aim)
            if op.x is not None:
                s = p.factors[op.x]
                self.cmath("Times", 2, 0, s.real)
                self.cmath("Times", 3, 1, s.imag)
                self.bi("Minus", 2, 2, 3)
                self.cmath("Times", 3, 0, s.imag)
                self.cmath("Times", 4, 1, s.real)
                self.bi("Plus", 3, 3, 4)
            else:
                self.emit({"Mov": {"dst": G(2), "s1": G(0)}})
                self.emit({"Mov": {"dst": G(3), "s1": G(1)}})
            self.bi("Times", 4, 3, 3)
            self.emit({"Fused": {"op": "MulAdd", "dst": G(4), "a": G(2), "b": G(2), "c": G(4)}})
            self.cmath("Times", 4, 4, float(p.weights[op.n]))
            self.load(5, self.out(op.d))
            self.bi("Plus", 5, 5, 4)
            self.save(self.out(op.d), 5)
        else:
            raise SimdError(f"the opcode {o.name}")

    def scatter(self, op):
        """out[j] (mode)= scatter_value(src[j], f) with pacops' shortcuts (f is a constant)"""
        p = self.prog
        f = self.row_factor(op.x)
        n = p.edge_layout(op.d).components
        for j in range(n):
            sre, sim = self.edge(op.l, j)
            dre, dim = self.edge(op.d, j)
            self.load(4, sre)
            self.load(5, sim)
            if f.imag == 0:
                if f.real == 1:
                    self.emit({"Mov": {"dst": G(0), "s1": G(4)}})
                    self.emit({"Mov": {"dst": G(1), "s1": G(5)}})
                elif f.real == -1:
                    self.emit({"Uni": {"op": "Neg", "dst": G(0), "s1": G(4)}})
                    self.emit({"Uni": {"op": "Neg", "dst": G(1), "s1": G(5)}})
                else:
                    self.cmath("Times", 0, 4, f.real)
                    self.cmath("Times", 1, 5, f.real)
            elif f.real == 0:                # (-v.im × f.im, v.re × f.im)
                self.emit({"Uni": {"op": "Neg", "dst": G(0), "s1": G(5)}})
                self.cmath("Times", 0, 0, f.imag)
                self.cmath("Times", 1, 4, f.imag)
            else:                            # rmul(v, f)
                self.cmul_const(sre, sim, f, a_first=True)
            if not op.set:                   # rcombine: old + value
                self.load(2, dre)
                self.load(3, dim)
                self.bi("Plus", 0, 2, 0)
                self.bi("Plus", 1, 3, 1)
            self.save(dre, 0)
            self.save(dim, 1)

    def source(self, op, name):
        p = self.prog
        (family, anti, h, ch) = source_fields(op)
        mass = p.scales[op.t].real
        n = {PhysicalType.SCALAR: 1, PhysicalType.WEYL: 2}.get(family, 4)
        if name is None:                     # a scalar: the wavefunction (1, 0), the products constants
            wave = [complex(1.0, 0.0)]
            w = None
        else:
            fd = p.edge_form(op.d)
            ins = {f"k{j}": self.form(fd, j) for j in range(4)}
            ins.update({"zero": 0.0, "one": 1.0, "two": 2.0, "half": 0.5, "sqh": SQH, "-sqh": -SQH, "mass": mass,
                        "h": float(h), "-h": float(-h)})
            for tag, sign in (("+", 1), ("-", -1)):
                hh = sign * h        # gluon: hel = h for E > 0, -h else; massive vector: -k, -h for E < 0
                ins.update({f"hel{tag}": float(hh), f"-hel{tag}": -float(hh), f"nsvahl{tag}": float(abs(hh)),
                            f"-nsvahl{tag}": -float(abs(hh)), f"hel0{tag}": 1.0 - abs(float(hh))})
                nsf = (-1 if tag == "+" else 1) if anti else (1 if tag == "+" else -1)
                nh = nsf * h
                ins.update({f"sf1{tag}": (1 + nsf + (1 - nsf) * nh) * 0.5, f"sf2{tag}": (1 + nsf - (1 - nsf) * nh) * 0.5,
                            f"nh{tag}": float(nh), f"-nh{tag}": -float(nh)})
            names = self.frag(name)["inputs"]
            w = [("scratch", j) for j in range(2 * n)]
            self.scratch_used = max(self.scratch_used, 2 * n)
            self.inline(name, {k: ins[k] for k in names}, w)
            wave = None
        if op.l is None:                     # out[j] = mul(f, w[j])
            f = p.factors[op.x]
            for j in range(n):
                dre, dim = self.edge(op.d, j)
                if w is None:
                    z = cmul(f, wave[j])
                    self.loadc(0, z.real)
                    self.loadc(1, z.imag)
                else:
                    self.cmul_const(w[2 * j], w[2 * j + 1], f, a_first=False)
                self.save(dre, 0)
                self.save(dim, 1)
        else:                                # embedded: out[components[e]] = mul(factors[e], w[waves[e]]) or 0
            for (comp, wv, x) in p.embeddings[op.l]:
                dre, dim = self.edge(op.d, comp)
                if wv is None:
                    self.zero([dre, dim])
                    continue
                f = p.factors[x]
                if w is None:
                    z = cmul(f, wave[wv])
                    self.loadc(0, z.real)
                    self.loadc(1, z.imag)
                else:
                    self.cmul_const(w[2 * wv], w[2 * wv + 1], f, a_first=False)
                self.save(dre, 0)
                self.save(dim, 1)

    def kernel_row(self, op, name):
        """adapter_<kernel>: the stub's inputs, the kernel, its output pairs (× factor unless direct) into
        out with the mode"""
        p, o = self.prog, op.op
        (s, set_) = p.instances[op.t]
        stub = p.stubs[s]
        scaleset = p.scalesets[set_]
        d = self.frag(name)
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
                ins[f"x{k}"] = float(p.scales[scaleset[i.n]].real)
            else:                            # FACTOR
                z = p.factors[op.x]
                ins[f"x{k}"] = float(z.imag if im else z.real)
        pairs = d["outputs"] // 2
        res = [("scratch", j) for j in range(2 * pairs)]
        self.scratch_used = max(self.scratch_used, 2 * pairs)
        self.inline(name, ins, res)
        f = p.factors[op.x]
        for n, comp in enumerate(stub.outputs):
            dre, dim = self.edge(op.d, comp)
            if o == Opcode.KERNEL_DIRECT:
                self.load(0, res[2 * n])
                self.load(1, res[2 * n + 1])
            else:                            # mul(pair, f)
                self.cmul_const(res[2 * n], res[2 * n + 1], f, a_first=True)
            if not op.set:                   # combine: old + value
                self.load(2, dre)
                self.load(3, dim)
                self.bi("Plus", 0, 2, 0)
                self.bi("Plus", 1, 3, 1)
            self.save(dre, 0)
            self.save(dim, 1)

    # ---- the calls ----
    def chain(self) -> int:
        """the accumulation runs: a call site whose old inputs are exactly the outputs of the call site before it
        (nothing between the two calls; none of its other inputs among those outputs) takes them from the result
        area, where that call left them (its result copies dropped: keep). Returns the number of chained sites."""
        links, prev = 0, None
        for item in self.code:
            if not isinstance(item, CallSite):
                prev = None
                continue
            olds = old_inputs(item.inputs)
            if prev is not None and olds and [item.inputs[n] for n in olds] == prev.outputs \
                    and not any(item.inputs[n] in prev.outputs for n in item.inputs if n not in olds):
                prev.keep = True
                for j, n in enumerate(olds):
                    item.inputs[n] = ("result", j)
                links += 1
            prev = item
        return links

    def reads_old_first(self, name) -> bool:
        """the body of fragment `name` reads its old inputs before it writes any output, so they can be its
        result slots (a trial splice with the old inputs and the outputs on the same slots)"""
        d = self.frag(name)
        olds = old_inputs(d["inputs"])
        targets = [("result", olds.index(n)) if n in olds else {"Mem": 1_000_000 + k} for k, n in enumerate(d["inputs"])]
        targets += [("result", j) for j in range(d["outputs"])]
        saved = self.consts, self.cidx, self.T
        self.consts, self.cidx, self.T = list(self.consts), dict(self.cidx), self.T or 0
        try:
            return self._splice(d, targets, check=True, dest=[])
        finally:
            self.consts, self.cidx, self.T = saved

    def finish(self):
        """the call sites (LoadArgs ultra, Call, the result copies), the bodies after @success, the
        per-row constant slots filled at the start. The accumulation runs are chained (chain); a body whose
        old inputs are the result slots at every call site, and reads them before writing, takes them there
        (not as arguments)."""
        self.links = self.chain()
        plans = {}
        for name, sites in self.sites.items():
            d = self.frag(name)
            baked, args = {}, []
            in_place = None
            for n in d["inputs"]:
                vals = {ckey(s.inputs[n]) if isinstance(s.inputs[n], float) else
                        s.inputs[n] if isinstance(s.inputs[n], tuple) and s.inputs[n][0] == "result" else None
                        for s in sites}
                if len(vals) == 1 and None not in vals and isinstance(sites[0].inputs[n], tuple):
                    if in_place is None:
                        in_place = self.reads_old_first(name)
                    if in_place:
                        baked[n] = sites[0].inputs[n]
                    else:
                        args.append(n)
                elif len(vals) == 1 and None not in vals:
                    baked[n] = sites[0].inputs[n]
                else:
                    args.append(n)
            args.sort(key=lambda n: isinstance(sites[0].inputs[n], float))   # the operands first (by slot number)
            if len(args) > ARGS_CAP:
                raise SimdError(f"{name}: {len(args)} arguments, symjit's subroutines take {ARGS_CAP}")
            plans[name] = (baked, args)
            for s in sites:
                for n in args:
                    if isinstance(s.inputs[n], float):
                        self.cslots.setdefault(ckey(s.inputs[n]), (len(self.cslots), s.inputs[n]))
        self.results = max((self.frag(n)["outputs"] for n in self.sites), default=0)
        code = []
        for (i, x) in self.cslots.values():
            code.append({"LoadConst": {"dst": G(0), "idx": self.const(x)}})
            code.append({"Save": {"loc": self.loc(("cslot", i)), "src": G(0)}})
        for item in self.code:
            if not isinstance(item, CallSite):
                code.append(item)
                continue
            baked, args = plans[item.name]
            locs = [self.loc(("cslot", self.cslots[ckey(t)][0]) if isinstance(t, float) else t)
                    for t in (item.inputs[n] for n in args)]
            code.append({"LoadArgs": {"locs": locs, "complex": False, "ultra": True}})
            code.append({"Call": {"label": self.label(item.name) + "_ultra", "num_args": 0}})
            if item.keep:
                continue
            for j, t in enumerate(item.outputs):
                code.append({"Load": {"dst": G(0), "loc": self.loc(("result", j))}})
                code.append({"Save": {"loc": self.loc(t), "src": G(0)}})
        bodies = []
        for name in sorted(self.sites):
            d = self.frag(name)
            baked, args = plans[name]
            pos = {n: k for k, n in enumerate(args)}
            targets = [baked[n] if n in baked else {"Stack": ARG_AREA + pos[n]} for n in d["inputs"]]
            targets += [self.loc(("result", j)) for j in range(d["outputs"])]
            label = self.label(name)
            bodies.append({"Label": {"label": label + "_ultra"}})
            bodies.append({"SaveArgs": {"num_args": len(args), "complex": False, "ultra": True}})
            bodies.append({"Label": {"label": label}})
            bodies.append({"SaveArgs": {"num_args": len(args), "complex": False, "ultra": False}})
            self._splice(d, targets, check=False, dest=bodies, prefix=label + "_")
            bodies.append({"Branch": {"label": ".ret"}})
        self.code = code
        self.bodies = bodies

    @staticmethod
    def label(name):
        return "@f_" + name.replace(":", "_").replace("-", "m")

    def application(self):
        p = self.prog
        stack = self.T + self.b.size + self.extra + self.scratch_used + len(self.cslots) + self.results
        cfg = library.config()
        opts = cfg["options"]
        opts.update({"use_threads": False, "fastmath": False, "compress": bool(self.bodies), "ultra": bool(self.bodies),
                     "simd_branch": True})
        tail = ([{"Branch": {"label": "@success"}}] + self.bodies) if self.bodies else []
        version = library.fragment(library.catalogue()[0])["symjit_version"]
        return {
            "format": "symjit-application", "format_version": 1, "symjit_version": version,
            "program": {"config": cfg, "count_states": self.count_states(), "count_params": 0, "count_obs": self.count_obs(),
                        "count_diffs": 0, "count_loops": 0,
                        "builder": {"consts": self.consts, "count_loops": 0, "count_stack": stack, "ft": []}},
            "prepared": {"compiled": True, "fast": False, "simd": False},
            "mir": {"complexified": False, "code": self.code + tail + ["End"], "consts": self.consts, "labels": {}},
            "reals": [],
        }

    def stats(self) -> dict:
        return {"rows": len(self.prog.ops), "instructions": len(self.code) + len(self.bodies),
                "bodies": len(self.bodies), "calls": sum(len(v) for v in self.sites.values()),
                "constants": len(self.consts),
                "stack": self.T + self.b.size + self.extra + self.scratch_used + len(self.cslots) + self.results}


# ---- f32 folding (the split kernels' kernel B: codegen/compile_split.py) --------------------------------

def fl32(x: float) -> float:
    """x rounded to f32 (to nearest)"""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def rf(z: complex) -> complex:
    """a table value as the Metal runner's f32 copy (rscales, rfactors)"""
    return complex(fl32(z.real), fl32(z.imag))


def rmul32(a: complex, b: complex) -> complex:
    """pacops.metal rmul in f32: each operation rounded once (in double, then to f32: correctly rounded,
    a double holding more than twice a float's digits)"""
    re = fl32(fl32(a.real * b.real) - fl32(a.imag * b.imag))
    im = fl32(fl32(a.real * b.imag) + fl32(a.imag * b.real))
    return complex(re, im)


def simd_application(prog: Lowered, call=called) -> tuple[dict, dict]:
    """the program's kernel as a symjit-application (JSON as a dict), and its sizes"""
    a = Assembler(prog, call)
    app = a.assemble()
    return app, a.stats()


# ---- the folder --------------------------------------------------------------------------------------------

ISAS = {"simd-avx2": "avx2", "simd-avx512": "avx512", "simd-neon": "neon"}   # device -> instruction set
LANES = {"avx2": 4, "avx512": 8, "neon": 2}                                 # its f64 lanes
NEEDS = {"avx2": "an x86-64 CPU with AVX2 and FMA", "avx512": "an x86-64 CPU with AVX-512 (F, DQ)",
         "neon": "an aarch64 CPU"}


def host_isa() -> Optional[str]:
    """the instruction set of this machine's SIMD device (_env.simd_device), or None"""
    from .. import _env
    d = _env.simd_device()
    return ISAS[d] if d else None


def _isa(isa: Optional[str]) -> str:
    """`isa` (default host_isa()), checked against the SIMD devices this machine builds: symjit's JIT
    compiles the kernels for its own CPU"""
    from .. import _env
    isa = isa or host_isa()
    if isa not in LANES:
        raise SimdError(f"no SIMD device on this machine ({', '.join(ISAS)}: {', '.join(NEEDS.values())})"
                        if isa is None else f"instruction set {isa!r}: one of {', '.join(LANES)}")
    if f"simd-{isa}" not in _env.simd_devices():
        raise SimdError(f"device simd-{isa} needs {NEEDS[isa]} to build on (symjit's JIT compiles the kernels)")
    return isa


def runner_h(prog: Lowered, isa: str, split: bool = False) -> str:
    n = LANES[isa]
    if split:      # the split kernels: A (f64) n lanes, B (f32) 2 n
        what = f"kernelA.h, kernelB.h: pac_kernel_a (f64, {n} lanes), pac_kernel_b (f32, {2 * n} lanes); A: states " \
               "the momenta, outputs the transfers; B: states the transfers, outputs the helicity outputs"
    else:
        what = f"kernel.h: pac_kernel ({n} lanes); states the momenta, outputs the helicity outputs"
    return f"""\
/* {prog.process} {prog.colour}: the program as SIMD kernels (simd-{isa}), by PacRunner's compile_simd
 * ({what}). */
#ifndef RUNNER_H
#define RUNNER_H

#define RUNNER_PROCESS {cstring(prog.process)}
#define RUNNER_COLOUR {cstring(prog.colour)}
#define RUNNER_LEGS {prog.legs}
#define RUNNER_OUTPUTS {prog.output_count()}
#define RUNNER_ARENA_SIZE {ArenaIndex.of(prog).size}
#define PAC_SIMD_ISA_{isa.upper()} 1
#define PAC_SIMD_DEVICE "simd-{isa}{'-split' if split else ''}"
#define PAC_SIMD_NEEDS "{NEEDS[isa]}"
{'#define PAC_SIMD_SPLIT 1' if split else ''}
#endif
"""


# the instruction sets whose kernels call their subroutines in the plain convention (plain_calls)
PLAIN_CALLS = {"neon"}


def plain_calls(app: dict) -> dict:
    """the application with its compress-mode calls in symjit's plain convention: ultra off, each LoadArgs
    loading the argument values into vector registers, each Call to the body's plain entry (its SaveArgs
    stores them to the argument area), instead of slot numbers packed 4 to an integer register. On aarch64
    the packing costs a movz/movk per argument; plain calls run 7-11% fewer instructions, 1-3% faster on
    the M4 (plans/pacrunner-python/F32-KERNEL.md)"""
    a = json.loads(json.dumps(app))
    a["program"]["config"]["options"]["ultra"] = False
    for ins in a["mir"]["code"]:
        if isinstance(ins, dict) and "LoadArgs" in ins:
            ins["LoadArgs"]["ultra"] = False
        if isinstance(ins, dict) and "Call" in ins and ins["Call"]["label"].endswith("_ultra"):
            ins["Call"]["label"] = ins["Call"]["label"][:-len("_ultra")]
    return a


def jit_kernel(app: dict, isa: str, what: str, folder) -> bytes:
    """the SIMD kernel of `app` for `isa` from symjit's JIT: `dump(.., what)`, "simd" or "simd-f32" (its
    f32 twin: symjit >= 2.29.2)"""
    import symjit
    a = json.loads(json.dumps(app))
    a["program"]["config"]["options"]["enable_simd512"] = isa == "avx512"
    f = symjit.load_json(json.dumps(a))
    path = Path(folder) / "k.bin"
    try:
        f.dump(str(path), what)
    except ValueError as e:
        if what == "simd-f32":
            raise SimdError(f"symjit {getattr(symjit, '__version__', '')} cannot dump an f32 kernel (simd-f32: "
                            "symjit >= 2.29.2; its error, if any, is on stderr)") from e
        raise SimdError(f"symjit dumps no SIMD kernel: {e}") from e
    code = path.read_bytes()
    path.unlink()
    return code


def compile_simd(path, prog: Lowered, isa: Optional[str] = None, precision: str = "f64"):
    """the folder of `prog` (see the module) for the instruction set `isa` ("avx2", "avx512", "neon";
    default host_isa()); returns `path`. The kernel: kernel.o (x86-64) or kernel.S + kernel.bin (aarch64),
    pac_kernel, kernel.h. precision "split": the split kernels (compile_simd_split)"""
    if precision in ("split", "mixed"):
        return compile_simd_split(path, prog, isa)
    if precision != "f64":
        raise SimdError(f"precision {precision!r}: f64 or split")
    from ..simd import elf
    isa = _isa(isa)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    app, _ = simd_application(prog)
    if isa in PLAIN_CALLS:
        app = plain_calls(app)
    elf.no_relocations(app["mir"]["code"])
    (path / "kernel.json").write_text(json.dumps(app))
    (path / "runner.h").write_text(runner_h(prog, isa))
    code = jit_kernel(app, isa, "simd", path)
    elf.write_object(path, "kernel", "pac_kernel", code, isa)
    n = LANES[isa]
    (path / "kernel.h").write_text(elf.header("pac_kernel", n, app["program"]["count_states"], app["program"]["count_obs"],
                                              elf.stack_bytes(app, n), f"{prog.process} {prog.colour}, simd-{isa}"))
    return path


def compile_simd_split(path, prog: Lowered, isa: Optional[str] = None):
    """compile_simd(precision = "split"): the split kernels (codegen/compile_split.py) for `isa`: kernel A
    in f64 (kernelA, pac_kernel_a: the momenta to the transfers, its f64 lanes) and kernel B in f32
    (kernelB, pac_kernel_b: the transfers to the outputs, twice the lanes: symjit's f32 twin of its f64
    kernel, `dump(.., "simd-f32")`); runner.h defines PAC_SIMD_SPLIT"""
    from ..simd import elf
    from .compile_split import split_applications
    isa = _isa(isa)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    app_a, app_b, _, items = split_applications(prog)
    if isa in PLAIN_CALLS:
        app_a, app_b = plain_calls(app_a), plain_calls(app_b)
    for app in (app_a, app_b):
        elf.no_relocations(app["mir"]["code"])
    (path / "kernelA.json").write_text(json.dumps(app_a))
    (path / "kernelB.json").write_text(json.dumps(app_b))
    (path / "runner.h").write_text(runner_h(prog, isa, split=True))
    what = f"{prog.process} {prog.colour}, simd-{isa}-split"
    n = LANES[isa]
    for tag, app, dump, m in (("A", app_a, "simd", n), ("B", app_b, "simd-f32", 2 * n)):
        code = jit_kernel(app, isa, dump, path)
        symbol = f"pac_kernel_{tag.lower()}"
        elf.write_object(path, f"kernel{tag}", symbol, code, isa)
        (path / f"kernel{tag}.h").write_text(elf.header(
            symbol, m, app["program"]["count_states"], app["program"]["count_obs"], elf.stack_bytes(app, n),
            what + (", kernel A (f64)" if tag == "A" else ", kernel B (f32: floats, call with idx / 2)")))
    return path

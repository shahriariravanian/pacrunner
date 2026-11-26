"""Hand-optimized SIMD fragments: library.py's expression trees (pacops' order of operations) compiled by a
small fragment compiler of our own instead of symjit's, for a register budget per ISA.

  - pacops' operations, in pacops' order: the outputs are lib/'s bit for bit;
  - negations folded where IEEE gives the same bits: a + (-b) = a - b, (-a) + b = b - a, (-a)·b = -(a·b),
    (-a)/b = -(a/b), --a = a; a Neg is emitted only for (-a) + (-b) (as (-a) - b: -(a + b) differs at a = -b,
    a signed zero) and for a negative output;
  - common subexpressions shared (Plus and Times operands in a canonical order: commutative, exact);
  - registers allocated over the ISA's pool (POOLS) by next use: an input is loaded once and reloaded from its
    Mem rather than spilled; a value used once may stay a memory operand (LoadMath); computed values spill to
    the fragment's own stack (Stack 80 + k) only when the pool runs out;
  - the outputs are stored at the end (every input read before any output is written: inlined and chained
    fragments may alias them).

Only the vertices, propagators and closures (+, ×, ÷ only: library.HAND_KINDS); the wavefunctions stay
lib/'s. The kernels of an ISA take these (library.fragment(name, isa)); PACRUNNER_SIMD_LIB=lib keeps lib/'s.

    python -m pacrunner.simd.handgen [isa...]     writes lib_hand_<isa>/ (default: every ISA of POOLS)
"""
from __future__ import annotations

import json
import sys

from . import library as L

# the logical registers a fragment may use (MIR Gen numbers), per ISA:
#   avx512: %n = zmm(n+2), zmm2..zmm31; avx2: ymm2..ymm15; neon (symjit's FMAP): %0..%13 = v2-v7, v16-v23,
#   %16..%18 = v26-v28, %19..%26 = v8-v15 (saved by the compress-mode prologue); v24, v25 and v29-v31 not used
POOLS = {
    "avx512": list(range(30)),
    "avx2": list(range(14)),
    "neon": list(range(14)) + list(range(16, 27)),
}
SPILL_BASE = 80          # the fragments' own stack slots (compile_simd: T = max(80, count_stack))


handled = L.hand_compiled


class Dag:
    """the outputs' trees as a DAG of positive values with sign flags folded (see the module)"""

    def __init__(self):
        self.nodes = {}          # key -> (op, a, b) ; ("in", k) for the inputs
        self.order = []          # keys in creation order

    def mk(self, op, a, b=None):
        if op in ("Plus", "Times") and b is not None and repr(b) < repr(a):
            a, b = b, a
        key = (op, a, b)
        if key not in self.nodes:
            self.nodes[key] = key
            self.order.append(key)
        return key

    def conv(self, t):
        """(key, negative) of a tree"""
        ty = t["type"]
        if ty == "Var":
            return ("in", int(t["name"][1:]), None), False
        if ty == "Const":
            raise ValueError(f"a constant {t['val']} outside times(-1, x)")
        op, args = t["op"], t["args"]
        if op == "times":
            if args[0]["type"] == "Const" and args[0]["val"] == -1.0:
                k, s = self.conv(args[1])
                return k, not s
            (ka, sa), (kb, sb) = self.conv(args[0]), self.conv(args[1])
            return self.mk("Times", ka, kb), sa != sb
        if op == "divide":
            (ka, sa), (kb, sb) = self.conv(args[0]), self.conv(args[1])
            return self.mk("Divide", ka, kb), sa != sb
        if op == "plus":
            (ka, sa), (kb, sb) = self.conv(args[0]), self.conv(args[1])
            if not sa and not sb:
                return self.mk("Plus", ka, kb), False
            if not sa and sb:
                return self.mk("Minus", ka, kb), False
            if sa and not sb:
                return self.mk("Minus", kb, ka), False
            return self.mk("Minus", self.mk("Neg", ka), kb), False
        raise ValueError(f"operation {op}")

    def output(self, t):
        k, s = self.conv(t)
        return self.mk("Neg", k) if s else k


def compile_trees(n_in: int, outs: list, pool: list) -> tuple[list, int]:
    """MIR for the outputs (Mem n_in + j) of trees over the inputs (Mem 0 .. n_in-1): (code, spill slots)"""
    dag = Dag()
    roots = [dag.output(t) for t in outs]

    # the schedule: post-order from the outputs, in their order (each value once)
    sched, seen = [], set()

    def visit(k):
        if k in seen or k[0] == "in":
            return
        seen.add(k)
        for c in k[1:]:
            if c is not None:
                visit(c)
        sched.append(k)

    for r in roots:
        visit(r)

    # uses: per value, the schedule positions reading it; the outputs are read at the end
    END = len(sched)
    uses = {}
    for p, k in enumerate(sched):
        for c in k[1:]:
            if c is not None:
                uses.setdefault(c, []).append(p)
    for r in roots:
        uses.setdefault(r, []).append(END)

    def next_use(k, p):
        for q in uses.get(k, []):
            if q >= p:
                return q
        return None

    code = []
    reg = {}                 # value -> register
    free = list(pool)
    spilled = {}             # computed value -> stack slot
    nslots = [0]

    def G(r):
        return {"Gen": r}

    def evict(p, keep):
        """free a register: the resident value used furthest ahead (inputs first: no store)"""
        best = None
        for k, r in reg.items():
            if k in keep:
                continue
            q = next_use(k, p)
            score = (q is None, k[0] == "in", q if q is not None else 0)
            if best is None or score > best[0]:
                best = (score, k)
        k = best[1]
        r = reg.pop(k)
        if k[0] != "in" and next_use(k, p) is not None and k not in spilled:
            spilled[k] = nslots[0]
            nslots[0] += 1
            code.append({"Save": {"loc": {"Stack": SPILL_BASE + spilled[k]}, "src": G(r)}})
        return r

    def alloc(p, keep):
        return free.pop(0) if free else evict(p, keep)

    def fetch(k, p, keep):
        """k in a register (loaded / reloaded)"""
        if k in reg:
            return reg[k]
        r = alloc(p, keep)
        if k[0] == "in":
            code.append({"Load": {"dst": G(r), "loc": {"Mem": k[1]}}})
        else:
            code.append({"Load": {"dst": G(r), "loc": {"Stack": SPILL_BASE + spilled[k]}}})
        reg[k] = r
        return r

    def release(k, p):
        if k in reg and next_use(k, p + 1) is None:
            free.insert(0, reg.pop(k))

    for p, k in enumerate(sched):
        op, a, b = k
        keep = {a, b}
        if op == "Neg":
            ra = fetch(a, p, keep)
            release(a, p)
            rd = alloc(p, keep)
            code.append({"Uni": {"op": "Neg", "dst": G(rd), "s1": G(ra)}})
        else:
            # a memory operand (LoadMath: dst = s1 op Mem): an input used only here, not in a register
            mem = None
            if b[0] == "in" and b not in reg and len(uses[b]) == 1 and op != "Divide":
                mem = b
            elif op in ("Plus", "Times") and a[0] == "in" and a not in reg and len(uses[a]) == 1 and a != b:
                a, b, mem = b, a, a
            ra = fetch(a, p, keep)
            rb = None if mem else fetch(b, p, keep)
            release(a, p)
            if not mem:
                release(b, p)
            rd = alloc(p, keep)
            if mem:
                code.append({"LoadMath": {"op": op, "dst": G(rd), "s1": G(ra), "loc": {"Mem": mem[1]}}})
            else:
                code.append({"Bi": {"op": op, "dst": G(rd), "s1": G(ra), "s2": G(rb)}})
        reg[k] = rd
        # a value nobody reads (cannot happen for a schedule from the outputs, but keep the pool honest)
        if not uses.get(k):
            free.insert(0, reg.pop(k))

    # the outputs, at the end
    for j, r in enumerate(roots):
        rr = fetch(r, END, set(roots[j:]))
        code.append({"Save": {"loc": {"Mem": n_in + j}, "src": G(rr)}})
    return code, nslots[0]


def hand_fragment(name: str, isa: str) -> dict:
    f, outs = L.build(name)
    code, nslots = compile_trees(len(f.names), outs, POOLS[isa])
    lib = json.loads(L.file_of(name).read_text())
    return {"format": L.FORMAT, "format_version": L.FORMAT_VERSION, "name": name,
            "symjit_version": lib["symjit_version"], "inputs": f.names, "outputs": len(outs),
            "count_stack": SPILL_BASE + nslots, "consts": [], "code": code + ["End"],
            "hand": {"isa": isa, "spills": nslots}}


def write(isa: str, quiet=False):
    out = L.hand_dir(isa)
    out.mkdir(exist_ok=True)
    for name in L.catalogue():
        if handled(name):
            d = hand_fragment(name, isa)
            (out / L.file_of(name).name).write_text(json.dumps(d, separators=(",", ":")) + "\n")
            if not quiet:
                lib = json.loads(L.file_of(name).read_text())
                print(f"{name:32} MIR {len(lib['code']):4} -> {len(d['code']):4}, spill slots {d['hand']['spills']}")
    return out


if __name__ == "__main__":
    for isa in sys.argv[1:] or list(POOLS):
        print(f"== {isa}")
        write(isa)

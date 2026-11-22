"""The library of the SIMD kernels (device "simd"): pacops.c's operations as symjit fragments, shipped in
pacrunner/simd/lib/ (one JSON file each) and spliced into a program's kernel by codegen/compile_simd.py.

A fragment is a symjit function of named inputs (its Mem 0 .. n_in) and outputs (Mem n_in ..), written as
symjit expression trees (structure.py's format, given to compile_func as they are) in pacops' exact
order of operations, in real (re, im) pairs, compiled with fastmath off: its bits are pacops.c's
(-ffp-contract=off). Every constant an operation uses (couplings × factors, 0.0, 1.0, 2.0, sqrt(0.5),
± the helicity, ...) is an input, so symjit folds nothing; the assembler turns those inputs into
kernel constants. The trees bypass sympy, which would reorder sums and products and turn a relation on
a Piecewise into an ITE.

The fragments:
  vertex:<kind>:<set|add>     pacops vertex(): out[j] (mode)= o[j] × s, s = coupling × factor (an input)
  propagator:<weyl_a|weyl_b|vector>   out = o × (factor × 1/(k·k)) (the vector's factor × (0, -1) folded)
  closure:<n>                 amplitude += Σ_j (l[j] r[j]) × c[j], c[j] = component factor × factor (folded)
  wave:weyl:<part|anti>:<h>:<ch>          the Weyl spinors (weyl_particle, weyl_antiparticle)
  wave:dirac:<part|anti>:<h>:<massless|massive>   quark/antiquark_dirac(_massive)
  wave:gluon:<h>, wave:massive_vector:<h>
The wavefunctions are bare (no factor): the source's factor, and an embedded source's entries, are
written by the assembler. Branches on the momentum (E > 0, a momentum on the z axis, pt = 0) are
selects, so the lanes of a SIMD kernel may take different ones.

Known differences from pacops (none in any test): Julia's max/min of a NaN (jmax(NaN, 0) is 0 here),
min(0.0, -0.0) (0.0 here, -0.0 in pacops), copysign(sqh, -0.0) at pt = 0 (+ here).

    python -m pacrunner.simd.library        regenerates lib/ (needs symjit >= 2.29)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LIB = Path(__file__).with_name("lib")
FORMAT = "pacrunner-simd-fragment"
FORMAT_VERSION = 1


class LibraryError(RuntimeError):
    pass


# ---- exact-order arithmetic: symjit expression trees -------------------------------------------------

def _x(a):
    if isinstance(a, dict):
        return a
    if isinstance(a, (int, float)):
        return {"type": "Const", "val": float(a)}
    raise TypeError(a)


def node(op, *args):
    return {"type": "Tree", "op": op, "args": [_x(a) for a in args]}


def A(a, b): return node("plus", a, b)
def S(a, b): return node("plus", a, node("times", -1, b))   # a - b (symjit: Minus, bit for bit)
def M(a, b): return node("times", a, b)
def N(a): return node("times", -1, a)                        # -a (Neg)
def D(a, b): return node("divide", a, b)
def SQ(a): return node("root", a)
def ABS(a): return node("abs", a)
def IF(cond, x, y): return node("ifelse", cond, x, y)
def AND(*args):
    out = args[0]
    for a in args[1:]:
        out = node("and", out, a)
    return out
def OR(a, b): return node("or", a, b)
def EQ(a, b): return node("eq", a, b)
def NE(a, b): return node("neq", a, b)
def LT(a, b): return node("lt", a, b)
def LE(a, b): return node("leq", a, b)
def GT(a, b): return node("gt", a, b)
def GE(a, b): return node("geq", a, b)


def cadd(a, b): return (A(a[0], b[0]), A(a[1], b[1]))
def csub(a, b): return (S(a[0], b[0]), S(a[1], b[1]))
def cneg(a): return (N(a[0]), N(a[1]))
def cmul(a, b): return (S(M(a[0], b[0]), M(a[1], b[1])), A(M(a[0], b[1]), M(a[1], b[0])))
def cmulr(a, x): return (M(a[0], x), M(a[1], x))
def cmul_i(a): return (N(a[1]), a[0])


def sel(cond, a, b):
    """cond ? a : b, per component"""
    return tuple(IF(cond, x, y) for x, y in zip(a, b))


def jmax0(x, zero):
    return IF(GT(x, 0), x, zero)          # Julia's max(x, 0.0) but for a NaN x


def jmin(x, y):
    return IF(LE(x, y), x, y)             # Julia's min(x, y) but for NaN and (0.0, -0.0)


def copysign(c, x, negc):
    return IF(GE(x, 0), c, negc)          # copysign(c, x) but for x = -0.0


class Frag:
    """the inputs of a fragment while it is written (each a symjit variable x<k>, with its name)"""

    def __init__(self):
        self.names = []

    def real(self, name):
        if name in self.names:
            return {"type": "Var", "name": f"x{self.names.index(name)}"}
        self.names.append(name)
        return {"type": "Var", "name": f"x{len(self.names) - 1}"}

    def cplx(self, name):
        return (self.real(name + ".re"), self.real(name + ".im"))

    def current(self, name, n):
        return [self.cplx(f"{name}{j}") for j in range(n)]

    def vec4(self, name):
        return [self.real(f"{name}{j}") for j in range(4)]


# ---- vertices ------------------------------------------------------------------------------------------

def _dot(a, b):
    return csub(csub(csub(cmul(a[0], b[0]), cmul(a[1], b[1])), cmul(a[2], b[2])), cmul(a[3], b[3]))


def _dot_real(a, v):
    return csub(csub(csub(cmulr(a[0], v[0]), cmulr(a[1], v[1])), cmulr(a[2], v[2])), cmulr(a[3], v[3]))


VERTICES = {   # kind: (l components, r components, out components)
    "weyl_vector_a": (2, 4, 2), "weyl_vector_b": (2, 4, 2), "tensor_vector": (6, 4, 4),
    "vector_wedge": (4, 4, 6), "three_vector": (4, 4, 4),
}


def _vertex_o(kind, l, r, u, v, two):
    if kind == "weyl_vector_a":
        l0, l1 = l
        r0, r1, r2, r3 = r
        return [cadd(cadd(csub(cneg(cmul_i(cmul(l0, r3))), cmul_i(cmul(l1, r1))), cmul(l1, r2)), cmul_i(cmul(l0, r0))),
                cadd(cadd(csub(cneg(cmul(l0, r2)), cmul_i(cmul(l0, r1))), cmul_i(cmul(l1, r0))), cmul_i(cmul(l1, r3)))]
    if kind == "weyl_vector_b":
        l0, l1 = l
        r0, r1, r2, r3 = r
        return [cadd(cadd(cadd(cneg(cmul(l1, r2)), cmul_i(cmul(l0, r0))), cmul_i(cmul(l0, r3))), cmul_i(cmul(l1, r1))),
                cadd(cadd(cadd(cneg(cmul_i(cmul(l1, r3))), cmul(l0, r2)), cmul_i(cmul(l0, r1))), cmul_i(cmul(l1, r0)))]
    if kind == "tensor_vector":
        l0, l1, l2, l3, l4, l5 = l
        r0, r1, r2, r3 = r
        return [cadd(cadd(cmul(l0, r1), cmul(l1, r2)), cmul(l2, r3)),
                cadd(cadd(cmul(l0, r0), cmul(l3, r2)), cmul(l4, r3)),
                cadd(csub(cmul(l1, r0), cmul(l3, r1)), cmul(l5, r3)),
                csub(csub(cmul(l2, r0), cmul(l4, r1)), cmul(l5, r2))]
    if kind == "vector_wedge":
        l0, l1, l2, l3 = l
        r0, r1, r2, r3 = r
        return [csub(cmul(l0, r1), cmul(l1, r0)), csub(cmul(l0, r2), cmul(l2, r0)), csub(cmul(l0, r3), cmul(l3, r0)),
                csub(cmul(l1, r2), cmul(l2, r1)), csub(cmul(l1, r3), cmul(l3, r1)), csub(cmul(l2, r3), cmul(l3, r2))]
    if kind == "three_vector":
        lr, lv, ru = _dot(l, r), _dot_real(l, v), _dot_real(r, u)
        return [cadd(cmulr(lr, S(u[j], v[j])), cmulr(csub(cmul(lv, r[j]), cmul(ru, l[j])), two)) for j in range(4)]
    raise KeyError(kind)


def vertex(kind, mode):
    nl, nr, no = VERTICES[kind]
    f = Frag()
    l, r = f.current("l", nl), f.current("r", nr)
    u, v = (f.vec4("u"), f.vec4("v")) if kind == "three_vector" else (None, None)
    two = f.real("two") if kind == "three_vector" else None
    s = f.cplx("s")
    old = f.current("old", no) if mode == "add" else None
    o = _vertex_o(kind, l, r, u, v, two)
    out = []
    for j in range(no):
        x = cmul(o[j], s)
        out += list(cadd(old[j], x) if mode == "add" else x)
    return f, out


# ---- propagators ---------------------------------------------------------------------------------------

def propagator(kind):
    f = Frag()
    n = 4 if kind == "vector" else 2
    c = f.current("c", n)
    k = f.vec4("k")
    fac = f.cplx("f")
    one = f.real("one")
    rn = D(one, S(S(S(M(k[0], k[0]), M(k[1], k[1])), M(k[2], k[2])), M(k[3], k[3])))
    s = cmulr(fac, rn)
    if kind == "vector":
        return f, [x for j in range(4) for x in cmul(c[j], s)]
    c0, c1 = c
    if kind == "weyl_a":
        o0 = cadd(cadd(csub(cneg(cmul_i(cmulr(c0, k[3]))), cmul_i(cmulr(c1, k[1]))), cmulr(c1, k[2])), cmul_i(cmulr(c0, k[0])))
        o1 = cadd(cadd(csub(cneg(cmulr(c0, k[2])), cmul_i(cmulr(c0, k[1]))), cmul_i(cmulr(c1, k[0]))), cmul_i(cmulr(c1, k[3])))
    else:
        o0 = cadd(cadd(cadd(cneg(cmulr(c1, k[2])), cmul_i(cmulr(c0, k[0]))), cmul_i(cmulr(c0, k[3]))), cmul_i(cmulr(c1, k[1])))
        o1 = cadd(cadd(cadd(cneg(cmul_i(cmulr(c1, k[3]))), cmulr(c0, k[2])), cmul_i(cmulr(c0, k[1]))), cmul_i(cmulr(c1, k[0])))
    return f, [*cmul(o0, s), *cmul(o1, s)]


# ---- closure -------------------------------------------------------------------------------------------

def closure(n):
    f = Frag()
    l, r = f.current("l", n), f.current("r", n)
    c = [f.cplx(f"c{j}") for j in range(n)]
    re, im = f.real("amp.re"), f.real("amp.im")
    for j in range(n):
        t = cmul(cmul(l[j], r[j]), c[j])
        re, im = A(re, t[0]), A(im, t[1])
    return f, [re, im]


# ---- wavefunctions (bare) ------------------------------------------------------------------------------

def _weyl_ab(f, anti):
    """(a, b) of weyl_particle (anti: weyl_antiparticle) for E > 0 and for E <= 0, as quark_dirac and
    antiquark_dirac share them"""
    E, px, py, pz = f.vec4("k")
    zero, two, hc, hm = f.real("zero"), f.real("two"), f.real("h"), f.real("-h")
    out = {}
    for pos in (True, False):
        axis = AND(EQ(px, 0), EQ(py, 0), LT(pz, 0) if pos else GT(pz, 0))
        if pos:
            root = SQ(jmax0(A(E, pz), zero))
            big = M(hm, SQ(M(two, E)))
        else:
            root = SQ(jmax0(N(A(E, pz)), zero))
            big = M(hm, SQ(M(two, ABS(E))))
        sgn_root = (N(root) if anti else root) if pos else (root if anti else N(root))
        s = IF(axis, zero, sgn_root)
        if not anti:
            other = (D(M(hc, px), s), D(N(py), s)) if pos else (D(M(hc, px), s), D(py, s))
        else:
            other = (D(M(hm, px), s), D(py, s)) if pos else (D(M(hm, px), s), D(N(py), s))
        out[pos] = ((s, zero), sel(EQ(s, 0), (big, zero), other))
    return E, zero, out


def wave_weyl(anti, h, ch):
    f = Frag()
    E, zero, ab = _weyl_ab(f, anti)
    Z = (zero, zero)
    res = {}
    for pos in (True, False):
        a, b = ab[pos]
        if not anti:
            pick = {(1, 1): (a, b), (-1, -1): (b, a)} if pos else {(-1, 1): (a, b), (1, -1): (b, a)}
        else:
            pick = {(1, 1): (b, a), (-1, -1): (a, b)} if pos else {(-1, 1): (b, a), (1, -1): (a, b)}
        res[pos] = pick.get((h, ch), (Z, Z))
    pos = GT(E, 0)
    return f, [x for j in range(2) for x in sel(pos, res[True][j], res[False][j])]


def wave_dirac_massless(anti, h):
    f = Frag()
    E, zero, ab = _weyl_ab(f, anti)
    Z = (zero, zero)
    res = {}
    for pos in (True, False):
        a, b = ab[pos]
        if not anti:
            upper = h == (1 if pos else -1)
            res[pos] = (a, b, Z, Z) if upper else (Z, Z, b, a)
        else:
            lower = h == (-1 if pos else 1)
            res[pos] = (Z, Z, a, b) if lower else (b, a, Z, Z)
    pos = GT(E, 0)
    return f, [x for j in range(4) for x in sel(pos, res[True][j], res[False][j])]


def wave_dirac_massive(anti, h):
    """dirac_massive: per sign of E its own nsf, nh, sf1, sf2 (inputs)"""
    f = Frag()
    E, px, py, pz = f.vec4("k")
    zero, one, half, two, mass = f.real("zero"), f.real("one"), f.real("half"), f.real("two"), f.real("mass")
    pp = SQ(A(A(M(px, px), M(py, py)), M(pz, pz)))
    omega1 = SQ(A(ABS(E), pp))
    omega2 = D(mass, omega1)
    res = {}
    for pos in (True, False):
        tag = "+" if pos else "-"
        nsf = (-1 if pos else 1) if anti else (1 if pos else -1)
        nh = nsf * h
        sf1, sf2, nhc, mnh = f.real(f"sf1{tag}"), f.real(f"sf2{tag}"), f.real(f"nh{tag}"), f.real(f"-nh{tag}")
        spx, spy, spz = (px, py, pz) if pos else (N(px), N(py), N(pz))
        pp3 = jmax0(A(pp, spz), zero)
        chi1 = sel(EQ(pp, 0), (one, zero), (SQ(D(M(pp3, half), pp)), zero))
        d = SQ(M(M(two, pp), pp3))
        chi2 = sel(OR(EQ(pp3, 0), EQ(pp, 0)), (mnh, zero), (D(M(nhc, spx), d), D(spy if anti else N(spy), d)))
        if anti:
            up1, lo1, up2, lo2 = M(sf1, omega2), M(sf2, omega1), M(sf1, omega1), M(sf2, omega2)
        else:
            up1, lo1, up2, lo2 = M(sf2, omega1), M(sf1, omega2), M(sf2, omega2), M(sf1, omega1)
        if nh > 0:
            res[pos] = (cmulr(chi1, up1), cmulr(chi2, up1), cmulr(chi1, lo1), cmulr(chi2, lo1))
        else:
            res[pos] = (cmulr(chi2, up2), cmulr(chi1, up2), cmulr(chi2, lo2), cmulr(chi1, lo2))
    pos = GT(E, 0)
    return f, [x for j in range(4) for x in sel(pos, res[True][j], res[False][j])]


def wave_gluon(h):
    f = Frag()
    E, px, py, pz = f.vec4("k")
    zero, sqh, nsqh = f.real("zero"), f.real("sqh"), f.real("-sqh")
    res = {}
    for pos in (True, False):
        tag = "+" if pos else "-"
        hel, mhel = f.real(f"hel{tag}"), f.real(f"-hel{tag}")
        pp = E if pos else N(E)
        pt = SQ(A(M(px, px), M(py, py)))
        wf3 = (M(D(M(hel, pt), pp), sqh), zero)
        if pos:
            pzpt = M(M(D(pz, M(pp, pt)), sqh), hel)
            w1 = (M(N(px), pzpt), M(D(N(py), pt), sqh))
            w2 = (M(N(py), pzpt), M(D(px, pt), sqh))
            cs = copysign(sqh, pz, nsqh)
        else:
            pzpt = M(M(D(N(pz), M(pp, pt)), sqh), hel)
            w1 = (M(px, pzpt), M(D(py, pt), sqh))
            w2 = (M(py, pzpt), M(D(N(px), pt), sqh))
            cs = N(copysign(sqh, pz, nsqh))
        nz = NE(pt, 0)
        res[pos] = ((zero, zero), sel(nz, w1, (M(mhel, sqh), zero)), sel(nz, w2, (zero, cs)), wf3)
    pos = GT(E, 0)
    return f, [x for j in range(4) for x in sel(pos, res[True][j], res[False][j])]


def _massive_vector_at(f, E, px, py, pz, tag):
    """massive_vector for E >= 0, the helicity constants of this sign of E"""
    sqh, nsqh, zero, mass = f.real("sqh"), f.real("-sqh"), f.real("zero"), f.real("mass")
    hel, mhel = f.real(f"hel{tag}"), f.real(f"-hel{tag}")
    nsv, mnsv, hel0 = f.real(f"nsvahl{tag}"), f.real(f"-nsvahl{tag}"), f.real(f"hel0{tag}")
    pt2 = A(M(px, px), M(py, py))
    pp = jmin(E, SQ(A(pt2, M(pz, pz))))
    pt = jmin(pp, SQ(pt2))
    emp = D(E, M(mass, pp))
    wf0 = (D(M(hel0, pp), mass), zero)
    wf3 = (A(M(M(hel0, pz), emp), M(D(M(hel, pt), pp), sqh)), zero)
    nz = NE(pt, 0)
    pzpt = M(M(D(pz, M(pp, pt)), sqh), hel)
    wf1 = sel(nz, (S(M(M(hel0, px), emp), M(px, pzpt)), M(D(M(mnsv, py), pt), sqh)), (M(mhel, sqh), zero))
    wf2 = sel(nz, (S(M(M(hel0, py), emp), M(py, pzpt)), M(D(M(nsv, px), pt), sqh)),
              (zero, M(nsv, copysign(sqh, pz, nsqh))))
    rest = [(zero, zero), (M(mhel, sqh), zero), (zero, M(nsv, sqh)), (hel0, zero)]
    z = EQ(pp, 0)
    return [sel(z, rest[j], w) for j, w in enumerate([wf0, wf1, wf2, wf3])]


def wave_massive_vector(h):
    f = Frag()
    E, px, py, pz = f.vec4("k")
    P = _massive_vector_at(f, E, px, py, pz, "+")
    Q = _massive_vector_at(f, N(E), N(px), N(py), N(pz), "-")   # E < 0: massive_vector(-k, -h)
    neg = LT(E, 0)
    return f, [x for j in range(4) for x in sel(neg, Q[j], P[j])]


# ---- the catalogue -------------------------------------------------------------------------------------

def catalogue():
    names = [f"vertex:{k}:{m}" for k in VERTICES for m in ("set", "add")]
    names += [f"propagator:{k}" for k in ("weyl_a", "weyl_b", "vector")]
    names += [f"closure:{n}" for n in range(1, 7)]
    names += [f"wave:weyl:{a}:{h}:{ch}" for a in ("part", "anti") for h in (-1, 1) for ch in (-1, 1)]
    names += [f"wave:dirac:{a}:{h}:{m}" for a in ("part", "anti") for h in (-1, 1) for m in ("massless", "massive")]
    names += [f"wave:gluon:{h}" for h in (-1, 1)]
    names += [f"wave:massive_vector:{h}" for h in (-1, 0, 1)]
    return names


def build(name):
    p = name.split(":")
    if p[0] == "vertex":
        return vertex(p[1], p[2])
    if p[0] == "propagator":
        return propagator(p[1])
    if p[0] == "closure":
        return closure(int(p[1]))
    if p[0] == "wave":
        if p[1] == "weyl":
            return wave_weyl(p[2] == "anti", int(p[3]), int(p[4]))
        if p[1] == "dirac":
            anti, h = p[2] == "anti", int(p[3])
            return wave_dirac_massive(anti, h) if p[4] == "massive" else wave_dirac_massless(anti, h)
        if p[1] == "gluon":
            return wave_gluon(int(p[2]))
        if p[1] == "massive_vector":
            return wave_massive_vector(int(p[2]))
    raise LibraryError(f"no fragment {name!r}")


def file_of(name: str) -> Path:
    return LIB / (name.replace(":", "_").replace("-", "m") + ".json")


def compile_fragment(name):
    """the fragment `name` compiled by symjit: its shipped form"""
    import sympy
    import symjit
    f, outs = build(name)
    syms = [sympy.Symbol(f"x{k}", real=True) for k in range(len(f.names))]
    fn = symjit.compile_func(syms, outs, fastmath=False, use_threads=False, opt_level=2)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fn.dump(os.path.join(d, "f.json"), "json")
        app = json.loads(Path(d, "f.json").read_text())
    return {"format": FORMAT, "format_version": FORMAT_VERSION, "name": name, "symjit_version": app["symjit_version"],
            "inputs": f.names, "outputs": len(outs), "count_stack": app["program"]["builder"]["count_stack"],
            "consts": app["mir"]["consts"], "code": app["mir"]["code"]}, app["program"]["config"]


_cache: dict = {}


def fragment(name: str) -> dict:
    """the shipped fragment `name` (lib/), cached"""
    if name not in _cache:
        path = file_of(name)
        if not path.is_file():
            raise LibraryError(f"the SIMD library has no fragment {name!r} ({path}); regenerate it: "
                               "python -m pacrunner.simd.library")
        d = json.loads(path.read_text())
        if d.get("format") != FORMAT or d.get("format_version") != FORMAT_VERSION:
            raise LibraryError(f"{path}: not a {FORMAT} v{FORMAT_VERSION} file")
        _cache[name] = d
    return _cache[name]


def config() -> dict:
    """the symjit configuration the fragments were compiled with (the kernels' template)"""
    return json.loads((LIB / "config.json").read_text())


def write_library(names=None, quiet=False) -> None:
    LIB.mkdir(exist_ok=True)
    cfg = None
    for name in names or catalogue():
        d, cfg = compile_fragment(name)
        file_of(name).write_text(json.dumps(d, separators=(",", ":")) + "\n")
        if not quiet:
            print(f"{name}: {len(d['code'])} MIR instructions", flush=True)
    (LIB / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    _cache.clear()


if __name__ == "__main__":
    write_library(sys.argv[1:] or None)

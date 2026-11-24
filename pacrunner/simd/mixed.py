"""The mixed fragments (plans/F32-KERNEL.md): c/metal/pacops.metal's operations, written as MIR
(simd/mir.py) in f64 + double-double, so that the kernel downgraded to f32 (simd/downgrade.py) computes
the Metal runner's bits (f32 + df64). Every operation is pacops.metal's, in its order; a plain value
here is pacops.metal's float, a dd its df64.

  propagator:<weyl_a|weyl_b|vector>:mixed
      inputs  c<j>.re, c<j>.im (the current, plain), k<j>.hi, k<j>.lo (the momentum, dd),
              f.re, f.im (the row's factor, plain; the vector's already × (0, -1), as the Metal
              runner's rmul(factor, (0, -1)) computes it in f32)
      outputs the current's components (re, im), plain
      1 / (k·k) in dd, rounded once; the numerators take k rounded (kr).

  wave:scalar:mixed
  wave:weyl:<part|anti>:<h>:<ch>:mixed
  wave:dirac:<part|anti>:<h>:<massless|massive>:mixed
  wave:gluon:<h>:mixed, wave:massive_vector:<h>:mixed
      inputs  k<j>.hi, k<j>.lo (the momentum, dd); mass.hi, mass.lo (massive Dirac, massive vector)
      outputs per component re.hi, re.lo, im.hi, im.lo: the bare wavefunction in dd (the source's factor
              and rounding into the planes are the assembler's: pacops.metal source(), uirvm source_row)
      pacops.metal's branches (on E > 0, a momentum on the z axis, pt = 0, ...) are selects: both sides
      computed, per lane. The Dirac choice by the mass (|mass| < 1e-8: the massless spinors) is the
      assembler's, the mass being the row's constant.
"""
from __future__ import annotations

from .mir import DD, DDX, Mir, Val, dd


class R(tuple):
    """pacops.metal's rcplx: a pair of plain values with its operations"""

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


def _current(m: Mir, n: int) -> list[R]:
    return [R(m, m.input(f"c{j}.re"), m.input(f"c{j}.im")) for j in range(n)]


def _put(m: Mir, values: list[R]) -> None:
    for j, z in enumerate(values):
        m.output(2 * j, z.re)
        m.output(2 * j + 1, z.im)


def propagator(kind: str) -> dict:
    n = 4 if kind == "vector" else 2
    inputs = [f"c{j}.{p}" for j in range(n) for p in ("re", "im")] + \
             [f"k{j}.{p}" for j in range(4) for p in ("hi", "lo")] + ["f.re", "f.im"]
    m = Mir(inputs, 2 * n, f"propagator:{kind}:mixed")
    d = DD(m)
    c = _current(m, n)
    k = [d.input(f"k{j}") for j in range(4)]
    f = R(m, m.input("f.re"), m.input("f.im"))
    # reciprocal_norm: D(1) / (((k0 k0 - k1 k1) - k2 k2) - k3 k3), rounded once
    norm = d.sub(d.sub(d.sub(d.mul(k[0], k[0]), d.mul(k[1], k[1])), d.mul(k[2], k[2])), d.mul(k[3], k[3]))
    r = d.to_plain(d.div(d.exact(1.0), norm))
    s = f.mulr(r)
    if kind == "vector":
        _put(m, [cj * s for cj in c])
        return m.fragment()
    kr = [d.to_plain(kj) for kj in k]
    c0, c1 = c
    if kind == "weyl_a":
        o0 = ((-(c0.mulr(kr[3]).i()) - c1.mulr(kr[1]).i()) + c1.mulr(kr[2])) + c0.mulr(kr[0]).i()
        o1 = ((-(c0.mulr(kr[2])) - c0.mulr(kr[1]).i()) + c1.mulr(kr[0]).i()) + c1.mulr(kr[3]).i()
    elif kind == "weyl_b":
        o0 = ((-(c1.mulr(kr[2])) + c0.mulr(kr[0]).i()) + c0.mulr(kr[3]).i()) + c1.mulr(kr[1]).i()
        o1 = ((-(c1.mulr(kr[3]).i()) + c0.mulr(kr[2])) + c0.mulr(kr[1]).i()) + c1.mulr(kr[0]).i()
    else:
        raise ValueError(f"no propagator {kind!r}")
    _put(m, [o0 * s, o1 * s])
    return m.fragment()


# ---- wavefunctions -----------------------------------------------------------------------------------

class CD(tuple):
    """pacops.metal's cdf64: (re, im) in dd"""

    def __new__(cls, re: dd, im: dd):
        return super().__new__(cls, (re, im))

    @property
    def re(self): return self[0]

    @property
    def im(self): return self[1]


def _wave(name: str, n: int, mass: bool, body) -> dict:
    inputs = [f"k{j}.{p}" for j in range(4) for p in ("hi", "lo")] + (["mass.hi", "mass.lo"] if mass else [])
    m = Mir(inputs, 4 * n, name)
    d = DDX(m)
    k = [d.input(f"k{j}") for j in range(4)]
    out = body(d, k, d.input("mass") if mass else None)
    assert len(out) == n, name
    for j, z in enumerate(out):
        for q, x in enumerate((z.re.hi, z.re.lo, z.im.hi, z.im.lo)):
            m.output(4 * j + q, x)
    return m.fragment()


def _creal(d: DDX, x: dd) -> CD:
    return CD(x, d.zero())


def _czero(d: DDX) -> CD:
    return CD(d.zero(), d.zero())


def _csel(d: DDX, mask: Val, a: CD, b: CD) -> CD:
    return CD(d.select(mask, a.re, b.re), d.select(mask, a.im, b.im))


def _spinor(d: DDX, k, h: int, anti: bool, pos: bool):
    """(a, b) of pacops.metal's Weyl and massless Dirac spinors, for E > 0 (pos) or not"""
    E, px, py, pz = k
    zero, hd = d.zero(), d.exact(float(h))
    on_axis = d.m.and_(d.m.and_(d.is_zero(px), d.is_zero(py)), d.lt(pz, zero) if pos else d.gt(pz, zero))
    root = d.sqrt(d.max(d.add(E, pz), zero)) if pos else d.sqrt(d.max(d.neg(d.add(E, pz)), zero))
    if pos != anti:     # particle E > 0, antiparticle E <= 0: +root
        s = d.select(on_axis, zero, root)
    else:
        s = d.select(on_axis, zero, d.neg(root))
    a = _creal(d, s)
    two_e = d.mul(d.exact(2.0), E if pos else d.abs(E))
    special = _creal(d, d.mul(d.exact(float(-h)), d.sqrt(two_e)))
    if not anti:
        general = CD(d.div(d.mul(hd, px), s), d.div(d.neg(py), s) if pos else d.div(py, s))
    else:
        mh = d.neg(hd)
        general = CD(d.div(d.mul(mh, px), s), d.div(py, s) if pos else d.div(d.neg(py), s))
    b = _csel(d, d.is_zero(s), special, general)
    return a, b


def wave_weyl(anti: bool, h: int, ch: int) -> dict:
    def body(d, k, _):
        z = _czero(d)
        branches = []
        for pos in (True, False):
            a, b = _spinor(d, k, h, anti, pos)
            if not anti:
                if pos:
                    o = [a, b] if (h, ch) == (1, 1) else [b, a] if (h, ch) == (-1, -1) else [z, z]
                else:
                    o = [a, b] if (h, ch) == (-1, 1) else [b, a] if (h, ch) == (1, -1) else [z, z]
            else:
                if pos:
                    o = [b, a] if (h, ch) == (1, 1) else [a, b] if (h, ch) == (-1, -1) else [z, z]
                else:
                    o = [b, a] if (h, ch) == (-1, 1) else [a, b] if (h, ch) == (1, -1) else [z, z]
            branches.append(o)
        pos = d.gt(k[0], d.zero())
        return [_csel(d, pos, x, y) for x, y in zip(*branches)]
    return _wave(f"wave:weyl:{'anti' if anti else 'part'}:{h}:{ch}:mixed", 2, False, body)


def wave_dirac_massless(anti: bool, h: int) -> dict:
    def body(d, k, _):
        z = _czero(d)
        branches = []
        for pos in (True, False):
            a, b = _spinor(d, k, h, anti, pos)
            if not anti:
                upper = h == (1 if pos else -1)
                o = [a, b, z, z] if upper else [z, z, b, a]
            else:
                lower = h == (-1 if pos else 1)
                o = [z, z, a, b] if lower else [b, a, z, z]
            branches.append(o)
        pos = d.gt(k[0], d.zero())
        return [_csel(d, pos, x, y) for x, y in zip(*branches)]
    return _wave(f"wave:dirac:{'anti' if anti else 'part'}:{h}:massless:mixed", 4, False, body)


def wave_dirac_massive(anti: bool, h: int) -> dict:
    def body(d, k, mass):
        E, px, py, pz = k
        zero = d.zero()
        branches = []
        for pos in (True, False):
            nsf = (-1 if pos else 1) if anti else (1 if pos else -1)
            nh = nsf * h
            pp = d.sqrt(d.add(d.add(d.mul(px, px), d.mul(py, py)), d.mul(pz, pz)))
            omega1 = d.sqrt(d.add(d.abs(E), pp))
            omega2 = d.div(mass, omega1)
            half = d.exact(0.5)
            sf1 = d.mul(d.exact(float(1 + nsf + (1 - nsf) * nh)), half)
            sf2 = d.mul(d.exact(float(1 + nsf - (1 - nsf) * nh)), half)
            spx, spy, spz = (px, py, pz) if pos else (d.neg(px), d.neg(py), d.neg(pz))
            pp3 = d.max(d.add(pp, spz), zero)
            pp_zero = d.is_zero(pp)
            chi1 = _csel(d, pp_zero, _creal(d, d.exact(1.0)), _creal(d, d.sqrt(d.div(d.mul(pp3, d.exact(0.5)), pp))))
            den = d.sqrt(d.mul(d.mul(d.exact(2.0), pp), pp3))
            num = CD(d.mul(d.exact(float(nh)), spx), spy if anti else d.neg(spy))
            chi2 = _csel(d, d.m.or_(d.is_zero(pp3), pp_zero), _creal(d, d.exact(float(-nh))),
                         CD(d.div(num.re, den), d.div(num.im, den)))
            if anti:
                up1, lo1, up2, lo2 = d.mul(sf1, omega2), d.mul(sf2, omega1), d.mul(sf1, omega1), d.mul(sf2, omega2)
            else:
                up1, lo1, up2, lo2 = d.mul(sf2, omega1), d.mul(sf1, omega2), d.mul(sf2, omega2), d.mul(sf1, omega1)

            def mulr(c, w):
                return CD(d.mul(c.re, w), d.mul(c.im, w))
            if nh > 0:
                branches.append([mulr(chi1, up1), mulr(chi2, up1), mulr(chi1, lo1), mulr(chi2, lo1)])
            else:
                branches.append([mulr(chi2, up2), mulr(chi1, up2), mulr(chi2, lo2), mulr(chi1, lo2)])
        pos = d.gt(E, zero)
        return [_csel(d, pos, x, y) for x, y in zip(*branches)]
    return _wave(f"wave:dirac:{'anti' if anti else 'part'}:{h}:massive:mixed", 4, True, body)


def wave_gluon(h: int) -> dict:
    def body(d, k, _):
        E, px, py, pz = k
        zero = d.zero()
        sqh = d.sqrt(d.exact(0.5))
        pt = d.sqrt(d.add(d.mul(px, px), d.mul(py, py)))
        pt_zero = d.is_zero(pt)
        branches = []
        for pos in (True, False):
            hel = d.exact(float(h if pos else -h))
            pp = E if pos else d.neg(E)
            wf3 = _creal(d, d.mul(d.div(d.mul(hel, pt), pp), sqh))
            pzpt = d.mul(d.mul(d.div(pz if pos else d.neg(pz), d.mul(pp, pt)), sqh), hel)
            if pos:
                g1 = CD(d.mul(d.neg(px), pzpt), d.mul(d.div(d.neg(py), pt), sqh))
                g2 = CD(d.mul(d.neg(py), pzpt), d.mul(d.div(px, pt), sqh))
                s2 = d.copysign(sqh, pz)
            else:
                g1 = CD(d.mul(px, pzpt), d.mul(d.div(py, pt), sqh))
                g2 = CD(d.mul(py, pzpt), d.mul(d.div(d.neg(px), pt), sqh))
                s2 = d.neg(d.copysign(sqh, pz))
            wf1 = _csel(d, pt_zero, _creal(d, d.mul(d.neg(hel), sqh)), g1)
            wf2 = _csel(d, pt_zero, CD(zero, s2), g2)
            branches.append([_czero(d), wf1, wf2, wf3])
        pos = d.gt(E, zero)
        return [_csel(d, pos, x, y) for x, y in zip(*branches)]
    return _wave(f"wave:gluon:{h}:mixed", 4, False, body)


def wave_massive_vector(h: int) -> dict:
    def body(d, k, mass):
        zero = d.zero()
        branches = []
        for flip in (False, True):          # pacops.c recurses with (-k, -h) for E < 0
            E, px, py, pz = (d.neg(x) for x in k) if flip else k
            hh = -h if flip else h
            sqh = d.sqrt(d.exact(0.5))
            hel, nsvahl = d.exact(float(hh)), d.exact(float(abs(hh)))
            pt2 = d.add(d.mul(px, px), d.mul(py, py))
            pp = d.min(E, d.sqrt(d.add(pt2, d.mul(pz, pz))))
            pt = d.min(pp, d.sqrt(pt2))
            hel0 = d.sub(d.exact(1.0), d.abs(hel))
            at_rest = [_czero(d), _creal(d, d.mul(d.neg(hel), sqh)), CD(zero, d.mul(nsvahl, sqh)), _creal(d, hel0)]
            emp = d.div(E, d.mul(mass, pp))
            wf0 = _creal(d, d.div(d.mul(hel0, pp), mass))
            wf3 = _creal(d, d.add(d.mul(d.mul(hel0, pz), emp), d.mul(d.div(d.mul(hel, pt), pp), sqh)))
            pzpt = d.mul(d.mul(d.div(pz, d.mul(pp, pt)), sqh), hel)
            g1 = CD(d.sub(d.mul(d.mul(hel0, px), emp), d.mul(px, pzpt)), d.mul(d.div(d.mul(d.neg(nsvahl), py), pt), sqh))
            g2 = CD(d.sub(d.mul(d.mul(hel0, py), emp), d.mul(py, pzpt)), d.mul(d.div(d.mul(nsvahl, px), pt), sqh))
            pt_zero = d.is_zero(pt)
            wf1 = _csel(d, pt_zero, _creal(d, d.mul(d.neg(hel), sqh)), g1)
            wf2 = _csel(d, pt_zero, CD(zero, d.mul(nsvahl, d.copysign(sqh, pz))), g2)
            pp_zero = d.is_zero(pp)
            branches.append([_csel(d, pp_zero, x, y) for x, y in zip(at_rest, [wf0, wf1, wf2, wf3])])
        neg = d.lt(k[0], zero)
        return [_csel(d, neg, y, x) for x, y in zip(*branches)]
    return _wave(f"wave:massive_vector:{h}:mixed", 4, True, body)


def wave_scalar() -> dict:
    return _wave("wave:scalar:mixed", 1, False, lambda d, k, _: [_creal(d, d.exact(1.0))])


# ---- the rows' dd arithmetic (uirvm.metal) ------------------------------------------------------------

def _dd_inputs(names):
    return [f"{n}.{p}" for n in names for p in ("hi", "lo")]


def closure(n: int) -> dict:
    """closure:<n>:mixed: amplitude += Σ_j rmul(rmul(l_j, r_j), c_j) (c_j = rmul(cf_j, f), the
    assembler's f32 constant), the products in f32, each added to the dd amplitude as dd_f (pacops.metal
    closure)"""
    inputs = [f"{a}{j}.{p}" for a in ("l", "r", "c") for j in range(n) for p in ("re", "im")]
    inputs += _dd_inputs(["amp.re", "amp.im"])
    m = Mir(inputs, 4, f"closure:{n}:mixed")
    d = DD(m)
    re, im = d.input("amp.re"), d.input("amp.im")
    for j in range(n):
        lj = R(m, m.input(f"l{j}.re"), m.input(f"l{j}.im"))
        rj = R(m, m.input(f"r{j}.re"), m.input(f"r{j}.im"))
        cj = R(m, m.input(f"c{j}.re"), m.input(f"c{j}.im"))
        t = (lj * rj) * cj
        re = d.add(re, d.of(t.re))
        im = d.add(im, d.of(t.im))
    for k, x in enumerate((re.hi, re.lo, im.hi, im.lo)):
        m.output(k, x)
    return m.fragment()


def form(terms: int, crossing: bool) -> dict:
    """form:<terms>:<plain|crossed>:mixed: q = 0 + Σ coefficient_t × leg_t in dd, in term order (a crossed
    leg: sign × the caller's leg); outputs q (hi, lo) and q rounded (the three-vector vertex's)"""
    names = [f"m{t}" for t in range(terms)] + [f"c{t}" for t in range(terms)]
    names += [f"s{t}" for t in range(terms)] if crossing else []
    m = Mir(_dd_inputs(names), 3, f"form:{terms}:{'crossed' if crossing else 'plain'}:mixed")
    d = DD(m)
    v = d.exact(0.0)
    for t in range(terms):
        x = d.input(f"m{t}")
        if crossing:
            x = d.mul(d.input(f"s{t}"), x)
        v = d.add(v, d.mul(d.input(f"c{t}"), x))
    m.output(0, v.hi)
    m.output(1, v.lo)
    m.output(2, d.to_plain(v))
    return m.fragment()


def _cmul(d: DD, a, b):
    """cdd_mul: (a.re b.re - a.im b.im, a.re b.im + a.im b.re) in dd"""
    return (d.sub(d.mul(a[0], b[0]), d.mul(a[1], b[1])), d.add(d.mul(a[0], b[1]), d.mul(a[1], b[0])))


def cmul_round() -> dict:
    """cmul_round:mixed: rof(cdd_mul(f, v)): a source component (the factor × the wavefunction's, in dd)
    rounded once into the plane"""
    m = Mir(_dd_inputs(["f.re", "f.im", "v.re", "v.im"]), 2, "cmul_round:mixed")
    d = DD(m)
    re, im = _cmul(d, (d.input("f.re"), d.input("f.im")), (d.input("v.re"), d.input("v.im")))
    m.output(0, d.to_plain(re))
    m.output(1, d.to_plain(im))
    return m.fragment()


def reduce(scaled: bool) -> dict:
    """reduce:<plain|scaled>:mixed: acc + weight × (a.re a.re + a.im a.im), a = the amplitude (× its scale),
    all in dd (pacops.metal reduce_term)"""
    names = ["amp.re", "amp.im"] + (["s.re", "s.im"] if scaled else []) + ["w", "acc"]
    m = Mir(_dd_inputs(names), 2, f"reduce:{'scaled' if scaled else 'plain'}:mixed")
    d = DD(m)
    a = (d.input("amp.re"), d.input("amp.im"))
    if scaled:
        a = _cmul(d, a, (d.input("s.re"), d.input("s.im")))
    w, acc = d.input("w"), d.input("acc")
    v = d.add(acc, d.mul(w, d.add(d.mul(a[0], a[0]), d.mul(a[1], a[1]))))
    m.output(0, v.hi)
    m.output(1, v.lo)
    return m.fragment()


def group() -> dict:
    """group:mixed: x × numerator ÷ denominator in dd (both dd_int)"""
    m = Mir(_dd_inputs(["x", "n", "d"]), 2, "group:mixed")
    d = DD(m)
    v = d.div(d.mul(d.input("x"), d.input("n")), d.input("d"))
    m.output(0, v.hi)
    m.output(1, v.lo)
    return m.fragment()


def fragment(name: str) -> dict:
    """the mixed fragment `name` (…:mixed)"""
    p = name.split(":")
    if p[-1] != "mixed":
        raise ValueError(f"{name}: not a mixed fragment")
    p = p[:-1]
    if p[0] == "propagator":
        return propagator(p[1])
    if p[0] == "closure":
        return closure(int(p[1]))
    if p[0] == "form":
        return form(int(p[1]), p[2] == "crossed")
    if p[0] == "cmul_round":
        return cmul_round()
    if p[0] == "reduce":
        return reduce(p[1] == "scaled")
    if p[0] == "group":
        return group()
    if p[0] == "kout":
        return kout(p[1], p[2] == "add")
    if p[0] == "wave":
        if p[1] == "scalar":
            return wave_scalar()
        if p[1] == "weyl":
            return wave_weyl(p[2] == "anti", int(p[3]), int(p[4]))
        if p[1] == "dirac":
            anti, h = p[2] == "anti", int(p[3])
            return wave_dirac_massive(anti, h) if p[4] == "massive" else wave_dirac_massless(anti, h)
        if p[1] == "gluon":
            return wave_gluon(int(p[2]))
        if p[1] == "massive_vector":
            return wave_massive_vector(int(p[2]))
    raise ValueError(f"no mixed fragment {name!r}")


def kout(scaled, add: bool) -> dict:
    """kout:<direct|scaled>:<set|add>:mixed: a kernel row's output pair into its component (uirvm.metal
    uir_output): the pair (dd) × the row's factor (dd) unless direct, added to the old component widened
    (add), rounded once into the plane. A factor of exactly 1 is "direct", of exactly -1 "neg": cdd_mul by
    (±1, 0) gives ±the pair (its products by 0 and ±1 exact, the normalized pair kept by fast_two_sum),
    the same up to the sign of a zero"""
    kind = scaled if isinstance(scaled, str) else ("scaled" if scaled else "direct")
    names = ["p.re", "p.im"] + (["f.re", "f.im"] if kind == "scaled" else [])
    inputs = _dd_inputs(names) + (["old.re", "old.im"] if add else [])
    m = Mir(inputs, 2, f"kout:{kind}:{'add' if add else 'set'}:mixed")
    d = DD(m)
    v = (d.input("p.re"), d.input("p.im"))
    if kind == "scaled":
        v = _cmul(d, v, (d.input("f.re"), d.input("f.im")))
    elif kind == "neg":          # cdd_mul(p, (-1, 0)) = -p: the products by 0 and ±1 are exact
        v = (d.neg(v[0]), d.neg(v[1]))
    if add:
        v = (d.add(d.of(m.input("old.re")), v[0]), d.add(d.of(m.input("old.im")), v[1]))
    m.output(0, d.to_plain(v[0]))
    m.output(1, d.to_plain(v[1]))
    return m.fragment()

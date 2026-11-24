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
"""
from __future__ import annotations

from .mir import DD, Mir, Val


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

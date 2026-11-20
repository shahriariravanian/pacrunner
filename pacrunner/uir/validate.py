"""The tables shared by the readers that build a Lowered (the vertex, propagator and source types),
the interning of table values, and `validate`, the checks of a Lowered: pac-julia's src/validate.jl.
"""

from __future__ import annotations

import struct
from typing import Hashable, Optional

from .types import (TYPE_COMPONENTS, LOp, Lowered, Opcode, PhysicalType, InputKind, is_kernel, is_propagator, is_vertex,
                    no_destination, overlaps, source_fields)

O, T = Opcode, PhysicalType

# vertex type → (opcode, l, r, out types)
VERTEX_OPS = {
    "weyl_vector_a": (O.V_WEYL_VECTOR_A, T.WEYL, T.VECTOR, T.WEYL),
    "weyl_vector_b": (O.V_WEYL_VECTOR_B, T.WEYL, T.VECTOR, T.WEYL),
    "tensor_vector": (O.V_TENSOR_VECTOR, T.TENSOR, T.VECTOR, T.VECTOR),
    "vector_wedge": (O.V_VECTOR_WEDGE, T.VECTOR, T.VECTOR, T.TENSOR),
    "three_vector": (O.V_THREE_VECTOR, T.VECTOR, T.VECTOR, T.VECTOR),
}
# propagator type → (opcode, the current's type)
PROPAGATOR_OPS = {"weyl_a": (O.P_WEYL_A, T.WEYL), "weyl_b": (O.P_WEYL_B, T.WEYL), "vector": (O.P_VECTOR, T.VECTOR)}
# source family → type
FAMILY_TYPES = {"scalar": T.SCALAR, "weyl": T.WEYL, "dirac": T.DIRAC, "vector": T.VECTOR}

OPCODE_SHAPES = {op: (l, r, out) for (op, l, r, out) in VERTEX_OPS.values()}
PROPAGATOR_TYPES = {op: t for (op, t) in PROPAGATOR_OPS.values()}


class InvalidLowered(ValueError):
    """a Lowered that fails `validate`"""


def cname(c: Optional[int]) -> str:
    """a current's name, for messages"""
    return f"c{c}"


# ---- interning ----------------------------------------------------------------------------------------

def float_key(x: float) -> int:
    """a double's bit pattern: values equal bit for bit (-0.0 and 0.0 differ, a NaN equals itself)"""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def complex_key(z: complex) -> tuple[int, int]:
    return (float_key(z.real), float_key(z.imag))


class Table:
    """A table whose values are interned: `add(x)` gives x's index, appending it if new. `key` maps a
    value to what decides equality (by default the value itself; float_key, complex_key: bit for
    bit), and the indices are in first-insertion order, as pac-julia's intern! and intern_value!."""

    def __init__(self, key=None, values=()):
        self.key = key or (lambda x: x)
        self.values: list = []
        self.index: dict[Hashable, int] = {}
        for v in values:
            self.add(v)

    def add(self, x) -> int:
        k = self.key(x)
        i = self.index.get(k)
        if i is None:
            i = self.index[k] = len(self.values)
            self.values.append(x)
        return i

    def __len__(self) -> int:
        return len(self.values)


# ---- validate -----------------------------------------------------------------------------------------

def _inrange(i: Optional[int], n: int) -> bool:
    return i is not None and 0 <= i < n


def _opt_inrange(i: Optional[int], n: int) -> bool:
    return i is None or 0 <= i < n


def _components(l: Lowered, e: int) -> int:
    return l.edge_layout(e).components


def _table_problems(out: list[str], l: Lowered) -> None:
    """the layout (even planes, inside the arena, a typed current's component count), the edges and the embeddings"""
    nc, nf = len(l.currents), len(l.forms)
    for c, cur in enumerate(l.currents):
        if cur.start % 2 != 0:
            out.append(f"{cname(c)} starts at odd plane {cur.start}")
        if not (cur.start >= 0 and cur.start + cur.components <= l.planes):
            out.append(f"{cname(c)} lies outside the {l.planes} planes")
        if cur.type != T.UNTYPED and TYPE_COMPONENTS[cur.type] != cur.components:
            out.append(f"{cname(c)} is {cur.type.name} with {cur.components} components")
    for e, edge in enumerate(l.edges):
        if not _inrange(edge.current, nc):
            out.append(f"edge {e}: no current {edge.current}")
        if not _opt_inrange(edge.form, nf):
            out.append(f"edge {e}: no form {edge.form}")
        if e < nc and edge.current != e:
            out.append(f"edge {e} is not current {cname(e)}'s own edge")
    if len(set(l.edges)) != len(l.edges):
        out.append("an edge is listed twice")
    for k, entries in enumerate(l.embeddings):
        for (j, w, x) in entries:
            if not (j >= 0 and (w is None or w >= 0) and _inrange(x, len(l.factors))):
                out.append(f"embedding {k}: entry ({j}, {w}, {x}) out of range")
        if len({j for (j, _, _) in entries}) != len(entries):
            out.append(f"embedding {k} writes a component twice")


def _row_edges_ok(l: Lowered, op: LOp) -> bool:
    """the edges each opcode names are in range: d (not for a closure, clear_amplitude or reduce); l
    and r (vertex, closure; a kernel's may be None); l (copy, scatter); a source's l is its embedding"""
    ne = len(l.edges)
    if not (no_destination(op) or _inrange(op.d, ne)):
        return False
    if is_vertex(op.op) or op.op == O.CLOSURE:
        return _inrange(op.l, ne) and _inrange(op.r, ne)
    if is_kernel(op.op):
        return _opt_inrange(op.l, ne) and _opt_inrange(op.r, ne)
    if op.op in (O.COPY, O.SCATTER):
        return _inrange(op.l, ne) and op.r is None
    if op.op == O.SOURCE:
        return _opt_inrange(op.l, len(l.embeddings)) and op.r is None
    return op.l is None and op.r is None


def _row_factor_ok(l: Lowered, op: LOp) -> bool:
    """the factor: none for clear and clear_amplitude, optional for reduce (its amplitude's scale) and
    an embedded source (its entries carry theirs)"""
    if op.op in (O.CLEAR, O.CLEAR_AMPLITUDE):
        return op.x is None
    if op.op == O.REDUCE:
        return _opt_inrange(op.x, len(l.factors))
    if op.op == O.SOURCE and op.l is not None:
        return op.x is None
    return _inrange(op.x, len(l.factors))


def _vertex_problems(out: list[str], l: Lowered, op: LOp, at: str, open_: Optional[int]) -> Optional[int]:
    """A vertex row: its operands' types, the momenta a three-vector vertex reads, its coupling, and
    its place in a fused run (SET only on the run's first row; the run's rows on one destination; no
    row after the first reads the destination's planes). `open_`: the destination edge of the open
    run (None: none); returns it after this row."""
    (tl, tr, tout) = OPCODE_SHAPES[op.op]
    for (e, t, what) in ((op.l, tl, "l"), (op.r, tr, "r"), (op.d, tout, "out")):
        have = l.edge_layout(e).type
        if have != t:
            out.append(f"{at}: {what} is {cname(l.edge_current(e))}, a {have.name}, not a {t.name}")
    if op.op == O.V_THREE_VECTOR and (l.edge_form(op.l) is None or l.edge_form(op.r) is None):
        out.append(f"{at}: an operand without a momentum")
    if not _inrange(op.t, len(l.scales)):
        out.append(f"{at}: no coupling {op.t}")
    if op.first:
        if open_ is not None:
            out.append(f"{at}: a run starts inside another run")
        open_ = op.d
    else:
        if open_ != op.d:
            out.append(f"{at}: continues a run on another destination")
        if op.set:
            out.append(f"{at}: SET on a row that does not start its run")
        dest = l.plane_range(op.d)
        if any(overlaps(l.plane_range(e), dest) for e in (op.l, op.r)):
            out.append(f"{at}: reads the planes of its run's destination")
    return None if op.last else open_


def _propagator_problems(out: list[str], l: Lowered, op: LOp, at: str) -> None:
    have = l.edge_layout(op.d).type
    if have != PROPAGATOR_TYPES[op.op]:
        out.append(f"{at}: {cname(l.edge_current(op.d))} is a {have.name}")
    if l.edge_form(op.d) is None:
        out.append(f"{at}: the current has no momentum")


def _kernel_problems(out: list[str], l: Lowered, op: LOp, at: str) -> None:
    """a kernel row: every input of its stub against the row's operands, and its outputs"""
    if not _inrange(op.t, len(l.instances)):
        out.append(f"{at}: no binding instance {op.t}")
        return
    (s, set_) = l.instances[op.t]
    stub, scales = l.stubs[s], l.scalesets[set_]
    for i in stub.inputs:
        if i.kind in (InputKind.IN_L, InputKind.IN_R, InputKind.MOMENTUM_U, InputKind.MOMENTUM_V):
            e = op.l if i.kind in (InputKind.IN_L, InputKind.MOMENTUM_U) else op.r
            if e is None:
                out.append(f"{at}: input {i.kind.name} of a null operand")
                continue
            if i.kind in (InputKind.IN_L, InputKind.IN_R):
                if not i.n < _components(l, e):
                    out.append(f"{at}: input {i.kind.name}[{i.n}] beyond {_components(l, e)} components")
            else:
                if l.edge_form(e) is None:
                    out.append(f"{at}: input {i.kind.name} of an operand without a momentum")
                if not i.n < 4:
                    out.append(f"{at}: momentum component {i.n}")
        elif i.kind == InputKind.OUT:
            if not i.n < _components(l, op.d):
                out.append(f"{at}: input out[{i.n}] beyond {_components(l, op.d)} components")
        elif i.kind == InputKind.SCALE:
            if not (_inrange(i.n, len(scales)) and _inrange(scales[i.n], len(l.scales))):
                out.append(f"{at}: scale slot {i.n}")
    if not all(0 <= j < _components(l, op.d) for j in stub.outputs):
        out.append(f"{at}: an output beyond {_components(l, op.d)} components")


def _closure_problems(out: list[str], l: Lowered, op: LOp, at: str) -> None:
    if not _inrange(op.d, l.amplitudes):
        out.append(f"{at}: no amplitude {op.d}")
    if not _inrange(op.t, len(l.lists)):
        out.append(f"{at}: no component-factor list {op.t}")
        return
    n = len(l.lists[op.t])
    if not n <= min(_components(l, op.l), _components(l, op.r)):
        out.append(f"{at}: {n} components, beyond its operands'")
    if not all(_inrange(x, len(l.factors)) for x in l.lists[op.t]):
        out.append(f"{at}: a component factor out of range")


def _source_problems(out: list[str], l: Lowered, op: LOp, at: str) -> None:
    try:
        (family, _, h, ch) = source_fields(op)
    except ValueError:
        out.append(f"{at}: template {op.n}")
        return
    if family not in (T.SCALAR, T.WEYL, T.DIRAC, T.VECTOR):
        out.append(f"{at}: family {family.name}")
    if not (-2 <= h <= 2 and -1 <= ch <= 1):
        out.append(f"{at}: helicity {h}, chirality {ch}")
    if not _inrange(op.t, len(l.scales)):
        out.append(f"{at}: no mass scale {op.t}")
    if l.edge_form(op.d) is None:
        out.append(f"{at}: its edge has no momentum")
    if op.l is None:
        have = l.edge_layout(op.d).type
        if have != family:
            out.append(f"{at}: {family.name} into a {have.name}")
    elif family in TYPE_COMPONENTS:
        n = TYPE_COMPONENTS[family]
        nd = _components(l, op.d)
        if not all(j < nd and (w is None or w < n) for (j, w, _) in l.embeddings[op.l]):
            out.append(f"{at}: embedding {op.l} beyond the current's {nd} or the wavefunction's {n} components")


def _reduce_problems(out: list[str], l: Lowered, op: LOp, at: str) -> None:
    if not _inrange(op.d, l.output_count()):
        out.append(f"{at}: no output {op.d}")
    if not _inrange(op.t, l.amplitudes):
        out.append(f"{at}: no amplitude {op.t}")
    if not _inrange(op.n, len(l.weights)):
        out.append(f"{at}: no weight {op.n}")


def _row_problems(out: list[str], l: Lowered, op: LOp, at: str) -> None:
    """a row that is not a vertex, by its opcode"""
    if is_propagator(op.op):
        _propagator_problems(out, l, op, at)
    elif is_kernel(op.op):
        _kernel_problems(out, l, op, at)
    elif op.op == O.CLOSURE:
        _closure_problems(out, l, op, at)
    elif op.op in (O.SCALE_ROW, O.CLEAR):
        if not (op.n is not None and 0 <= op.n <= _components(l, op.d)):
            out.append(f"{at}: {op.n} components, beyond {_components(l, op.d)}")
    elif op.op == O.SOURCE:
        _source_problems(out, l, op, at)
    elif op.op == O.CLEAR_AMPLITUDE:
        if not _inrange(op.d, l.amplitudes):
            out.append(f"{at}: no amplitude {op.d}")
    elif op.op in (O.COPY, O.SCATTER):
        if not _components(l, op.l) >= _components(l, op.d):
            out.append(f"{at}: copies {_components(l, op.d)} components from {_components(l, op.l)}")
    elif op.op == O.REDUCE:
        _reduce_problems(out, l, op, at)


def problems(l: Lowered) -> list[str]:
    """The problems of `l`, empty when it is valid: the layout (even planes, inside the arena), the
    edges, the sources, each row's operands against its opcode (edges, physical types, component
    counts, the momenta it reads, table indices) and the fused runs (SET only on a run's first row; a
    run is vertex rows on one destination, and no row after the first reads the destination's planes)."""
    out: list[str] = []
    _table_problems(out, l)
    open_ = None   # the destination edge of the open fused run
    for k, op in enumerate(l.ops):
        at = f"row {k} ({op.op.name})"
        if op.op == O.END:
            out.append(f"{at}: an END row (the encoding's, not a Lowered's)")
            continue
        if not _row_edges_ok(l, op):
            out.append(f"{at}: an edge out of range")
            continue
        if not _row_factor_ok(l, op):
            out.append(f"{at}: factor {op.x}")
        if op.set and not (is_vertex(op.op) or is_kernel(op.op) or op.op == O.SCATTER):
            out.append(f"{at}: SET on a row without a mode")
        if (op.first or op.last) and not is_vertex(op.op):
            out.append(f"{at}: FIRST/LAST on a row that is not a vertex")
        if is_vertex(op.op):
            open_ = _vertex_problems(out, l, op, at, open_)
        else:
            if open_ is not None:
                out.append(f"{at}: inside the fused run on edge {open_}")
            _row_problems(out, l, op, at)
    if open_ is not None:
        out.append(f"the fused run on edge {open_} does not end")
    return out


def validate(l: Lowered) -> Lowered:
    """`l`, or InvalidLowered with its first problems"""
    p = problems(l)
    if not p:
        return l
    raise InvalidLowered(f"invalid Lowered ({l.process}, {l.colour}): {len(p)} problems\n  " + "\n  ".join(p[:20]))

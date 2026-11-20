"""encode(l) -> Encoded: the device form of UIR v0.8 (design doc, "The 8-byte encoding"), pac-julia's
src/encode.jl: one stream of 64-bit words, the rows in order (SOURCE rows among them), then one END;
and the edge table as 32-bit entries. The other tables (scales, factors, component-factor lists,
embeddings, stubs, scale sets, binding instances, forms, weights) are the Lowered's.

An index field holds the index; all ones means none (None: the null operand of a kernel row, ...).

Word 0 of a row (no field crosses bit 32, so each half decodes with 32-bit bit-field operations):
  bits 0-4 op, 5 EXT (word 1 follows), 6 SET, 7 FIRST, 8 LAST,
  9-21 D (13), 22-31 T (10), 32-44 L (13), 45-57 R (13), 32-47 N (16, on L and R's bits), 58-63 X (6)
Word 1 of a wide row (EXT), each field's high bits:
  bit 0 EXT (another word: none in v0.8), 1-7 D (+7 → 20), 8-14 L (+7), 15-21 R (+7),
  22-27 T (+6 → 16), 8-23 N (+16 → 32, on L and R's bits), 32-41 X (+10 → 16)
A row is wide when one of its values does not fit word 0 (an index ≥ 2^width − 1, all ones being none).

An edge-table entry (32 bits): bits 0-14 start plane / 2, 15-22 form (8; 255 none), 23-26
components, 27-30 physical type (PhysicalType's number), 31 spare.
A SOURCE row: D its edge (the current and the form), T its mass scale, X its factor (none with an
embedding), N its template (source_template: bits 0-2 family, 3 antiparticle, 4-6 helicity + 2, 7-8
chirality + 1; no high bits), E its embedding at bits 48-57 (+10 at bits 8-17 of word 1; none: all
ones). A COPY or SCATTER row: D, L edges and X (the count is D's components). A REDUCE row: D the
output, T the amplitude, X the amplitude's scale (none: all ones), N the weight (an index here; its
high bits 8-21 of word 1, below T's).
The stream ends with one END row (opcode 31, no fields): a device runs the rows until the op is END.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

from .types import LCurrent, LEdge, LOp, Lowered, Opcode, PhysicalType, is_kernel, is_propagator, is_vertex
from .validate import validate


class EncodeError(ValueError):
    """a Lowered that does not fit UIR v0.8's encoding"""


@dataclass(frozen=True)
class Encoded:
    version: str
    rows: tuple[int, ...]    # the rows' 64-bit words, then END's
    edges: tuple[int, ...]   # 32-bit entries


class Field(NamedTuple):
    """a field: (shift, width) in word 0, and (shift, width) of its high bits in word 1"""
    shift: int
    width: int
    hishift: int
    hiwidth: int


FD = Field(9, 13, 1, 7)
FT = Field(22, 10, 22, 6)
FL = Field(32, 13, 8, 7)
FR = Field(45, 13, 15, 7)
FN = Field(32, 16, 8, 16)
FX = Field(58, 6, 32, 10)
FTEMPLATE = Field(32, 16, 8, 0)   # a SOURCE's N: word 0 only (its high bits would cross T's in word 1)
FE = Field(48, 10, 8, 10)         # a SOURCE's embedding (its L): above its template
FW = Field(32, 16, 8, 14)         # a REDUCE's weight (its N): high bits below T's in word 1
EXT = 1 << 5


def _none(f: Field, wide: bool) -> int:
    return (1 << (f.width + (f.hiwidth if wide else 0))) - 1


def _mask(n: int) -> int:
    return (1 << n) - 1


_VERTEX_FIELDS = (("d", FD, True), ("t", FT, True), ("l", FL, True), ("r", FR, True), ("x", FX, True))
_FIELDS = {
    Opcode.P_WEYL_A: (("d", FD, True), ("x", FX, True)),
    Opcode.SCALE_ROW: (("d", FD, True), ("n", FN, False), ("x", FX, True)),
    Opcode.SOURCE: (("d", FD, True), ("t", FT, True), ("n", FTEMPLATE, False), ("l", FE, True), ("x", FX, True)),
    Opcode.CLEAR: (("d", FD, True), ("n", FN, False)),
    Opcode.CLEAR_AMPLITUDE: (("d", FD, True),),
    Opcode.COPY: (("d", FD, True), ("l", FL, True), ("x", FX, True)),
    Opcode.REDUCE: (("d", FD, True), ("t", FT, True), ("n", FW, True), ("x", FX, True)),
    Opcode.END: (),
}


def fields(op: Opcode) -> tuple[tuple[str, Field, bool], ...]:
    """the fields each opcode uses: (name, field, whether it holds an index (all ones: none) or a count)"""
    if is_vertex(op) or is_kernel(op) or op == Opcode.CLOSURE:
        return _VERTEX_FIELDS
    if is_propagator(op):
        return _FIELDS[Opcode.P_WEYL_A]
    if op == Opcode.SCATTER:
        return _FIELDS[Opcode.COPY]
    return _FIELDS[op]


def _encoded_value(v: Optional[int], index: bool, f: Field, wide: bool) -> int:
    """a field's value in the encoding: an index (None: all ones), a count as it is"""
    if index and v is None:
        return _none(f, wide)
    return v


def _fits(v: Optional[int], index: bool, f: Field, wide: bool) -> bool:
    if index:
        return v is None or 0 <= v < _none(f, wide)
    return v is not None and 0 <= v < (1 << (f.width + (f.hiwidth if wide else 0)))


def _encode_row(words: list[int], op: LOp) -> None:
    """the words of a row, appended to `words`"""
    fs = fields(op.op)
    for (s, f, index) in fs:
        if not _fits(getattr(op, s), index, f, True):
            raise EncodeError(f"{op.op.name}: {s} = {getattr(op, s)} does not fit UIR v0.8")
    for s in ("d", "t", "l", "r", "n", "x"):
        if getattr(op, s) is not None and all(s != name for (name, _, _) in fs):
            raise EncodeError(f"{op.op.name}: {s} = {getattr(op, s)} is not one of its fields")
    wide = not all(_fits(getattr(op, s), index, f, False) for (s, f, index) in fs)
    w0 = int(op.op) | (EXT if wide else 0) | int(op.set) << 6 | int(op.first) << 7 | int(op.last) << 8
    w1 = 0
    for (s, f, index) in fs:
        v = _encoded_value(getattr(op, s), index, f, wide)
        w0 |= (v & _mask(f.width)) << f.shift
        if wide:
            w1 |= (v >> f.width) << f.hishift
    words.append(w0)
    if wide:
        words.append(w1)


def _decode_row(words: Sequence[int], k: int) -> tuple[LOp, int]:
    """the row at words[k] and the index of the next row"""
    w0 = words[k]
    op = Opcode(w0 & 31)
    wide = w0 & EXT != 0
    w1 = words[k + 1] if wide else 0
    if wide and w1 & 1:
        raise EncodeError(f"row at word {k}: a third word (not in UIR v0.8)")
    vals = {}
    for (s, f, index) in fields(op):
        v = (w0 >> f.shift) & _mask(f.width)
        if wide:
            v |= ((w1 >> f.hishift) & _mask(f.hiwidth)) << f.width
        vals[s] = (None if v == _none(f, wide) else v) if index else v
    o = LOp(op, w0 >> 6 & 1 == 1, w0 >> 7 & 1 == 1, w0 >> 8 & 1 == 1, **vals)
    return (o, k + (2 if wide else 1))


EDGE_NOFORM = 255


def encode_edge(l: Lowered, e: int) -> int:
    c = l.edge_layout(e)
    f = l.edges[e].form
    if not (c.start % 2 == 0 and c.start // 2 < 2 ** 15):
        raise EncodeError(f"edge {e}: start plane {c.start} does not fit UIR v0.8")
    if not (f is None or f < EDGE_NOFORM):
        raise EncodeError(f"edge {e}: form {f} does not fit UIR v0.8")
    if not c.components < 16:
        raise EncodeError(f"edge {e}: {c.components} components")
    form = EDGE_NOFORM if f is None else f
    return c.start // 2 | form << 15 | c.components << 23 | int(c.type) << 27


def decode_edge(x: int) -> tuple[int, Optional[int], int, PhysicalType]:
    """an edge-table entry: (start plane, form (None: none), components, physical type)"""
    form = (x >> 15) & 0xFF
    return ((x & 0x7FFF) * 2, None if form == EDGE_NOFORM else form, (x >> 23) & 0xF, PhysicalType((x >> 27) & 0xF))


def encode(l: Lowered) -> Encoded:
    validate(l)
    words: list[int] = []
    for op in l.ops:
        _encode_row(words, op)
    _encode_row(words, LOp(Opcode.END))
    return Encoded(l.version, tuple(words), tuple(encode_edge(l, e) for e in range(len(l.edges))))


def decode_rows(words: Sequence[int]) -> list[LOp]:
    """the rows of an encoded stream (no END)"""
    ops = []
    k = 0
    while k < len(words):
        (o, k) = _decode_row(words, k)
        ops.append(o)
    return ops


def decode_stream(e: Encoded) -> list[LOp]:
    """the rows of an Encoded: up to the one END, the last word"""
    n = len(e.rows)
    if not (n >= 1 and e.rows[-1] & 31 == Opcode.END):
        raise EncodeError("the stream does not end with END")
    ops = decode_rows(e.rows[:-1])
    if any(o.op == Opcode.END for o in ops):
        raise EncodeError("an END among the rows")
    return ops


def decode(e: Encoded, l: Lowered) -> Lowered:
    """What a device sees, as a Lowered: the rows and edges decoded from `e`, the other tables from
    `l`. Each edge becomes its own current (its planes, components and type from the entry), so the
    decoded Lowered holds only what the encoding carries; its values must still equal `l`'s."""
    entries = [decode_edge(x) for x in e.edges]
    currents = tuple(LCurrent(start, n, ty) for (start, _, n, ty) in entries)
    edges = tuple(LEdge(k, form) for k, (_, form, _, _) in enumerate(entries))
    return validate(dataclasses.replace(l, version=e.version, currents=currents, edges=edges, ops=tuple(decode_stream(e))))

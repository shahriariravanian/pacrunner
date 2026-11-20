"""UIR, the Unified Intermediate Representation (v0.8): a program as a flat list of fixed-size rows with
every operand resolved, which the back ends read (the C generator, the GPU interpreters' 8-byte
encoding). `load_recurrence_export` (export.py) reads RustiCol's recurrence export into it row for
row. The design is section 7 of the pac-julia design doc ("The lowered IR"); this is pac-julia's
src/uir.jl and src/types.jl.

v0.8: sources are rows (SOURCE, anywhere in the stream: a program of several runs has each run's
sources in it) and may scatter their components (an embedding); the reduction is rows (REDUCE, after
each run: out[h] += weight × |scale × amplitude|²); CLEAR_AMPLITUDE, COPY and SCATTER (RustiCol's
certified reuse and fanout); KERNEL_DIRECT stores the kernel's pairs unscaled, with its mode.

Operands are edges: an edge is a (current, momentum form) pair, and the edge table gives each edge's
current, whose layout entry gives its planes.

Indices are 0-based everywhere (unlike pac-julia's 1-based structures), and an absent one is None
(pac-julia's 0), so the encoding and the C write them as they are. Plane offsets (LCurrent.start,
Lowered.planes) are offsets, as in pac-julia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

UIR_VERSION = "0.8.0"   # as pac-julia spells v"0.8"


class PhysicalType(IntEnum):
    """What a current holds; the component count follows from it (TYPE_COMPONENTS). UNTYPED: no row
    or source says (a current only kernels touch)."""
    UNTYPED = 0
    SCALAR = 1
    WEYL = 2
    DIRAC = 3
    VECTOR = 4
    TENSOR = 5


TYPE_COMPONENTS = {PhysicalType.SCALAR: 1, PhysicalType.WEYL: 2, PhysicalType.DIRAC: 4, PhysicalType.VECTOR: 4,
                   PhysicalType.TENSOR: 6}


class Opcode(IntEnum):
    """The opcodes, numbered as in the 8-byte word (bits 0-4): the vertex types, the propagator types,
    then the other rows. A vertex's mode is the SET flag, a kernel's direct store its own opcode."""
    V_WEYL_VECTOR_A = 0   # l Weyl, r vector → Weyl
    V_WEYL_VECTOR_B = 1   # l Weyl, r vector → Weyl
    V_TENSOR_VECTOR = 2   # l tensor, r vector → vector
    V_VECTOR_WEDGE = 3    # l vector, r vector → tensor
    V_THREE_VECTOR = 4    # l vector, r vector → vector; reads both momenta
    P_WEYL_A = 5          # the propagators: current Weyl, reads its momentum
    P_WEYL_B = 6
    P_VECTOR = 7          # current vector (Feynman gauge)
    CLOSURE = 8
    SCALE_ROW = 9         # scale (SCALE is a KernelInput kind)
    CLEAR = 10
    KERNEL = 11
    KERNEL_DIRECT = 12    # a direct binding: the pairs stored as they are (the factor not applied), with the mode
    SOURCE = 13           # a source current
    CLEAR_AMPLITUDE = 14  # an amplitude set to 0
    COPY = 15             # a current's components × a factor into another (RustiCol's certified reuse)
    SCATTER = 16          # a fanout scatter: as COPY, with a mode and RustiCol's factor shortcuts
    REDUCE = 17           # out[h] += weight × |amplitude|² (the amplitude × its replay scale first)
    END = 31              # the end of the stream: in the encoding only, its last word


def is_vertex(op: Opcode) -> bool:
    return op <= Opcode.V_THREE_VECTOR


def is_propagator(op: Opcode) -> bool:
    return Opcode.P_WEYL_A <= op <= Opcode.P_VECTOR


def is_kernel(op: Opcode) -> bool:
    return op == Opcode.KERNEL or op == Opcode.KERNEL_DIRECT


class InputKind(IntEnum):
    """A kernel input: a part of an operand's component (IN_L, IN_R, OUT), a momentum component
    (MOMENTUM_U, MOMENTUM_V), the real part of a scale, a part of the row's factor."""
    IN_L = 0
    IN_R = 1
    OUT = 2
    MOMENTUM_U = 3
    MOMENTUM_V = 4
    SCALE = 5
    FACTOR = 6


@dataclass(frozen=True)
class KernelInput:
    kind: InputKind
    n: Optional[int]   # a component, a momentum component, the slot of SCALE in the scale set; None for FACTOR
    im: bool           # the imaginary part


@dataclass(frozen=True)
class Kernel:
    """A kernel of the export: a saved symjit application (.sjb), named by its SHA-256. Kernels are
    equal when their SHA-256 is."""
    sha256: str
    application: bytes = field(repr=False, compare=False)

    @property
    def name(self) -> str:
        """its name in the C: k_ and the first 16 hex digits of the SHA-256"""
        return "k_" + self.sha256[:16]


@dataclass(frozen=True)
class Form:
    """q = Σ coefficient × p[leg] over the terms, the first term added to 0.0, in term order"""
    terms: tuple[tuple[int, float], ...]   # (leg, coefficient)


@dataclass(frozen=True)
class LCurrent:
    """a current's place in the arena: planes start .. start + components - 1, start even"""
    start: int
    components: int
    type: PhysicalType


@dataclass(frozen=True)
class LEdge:
    """a current with the momentum form it is read with (None: none)"""
    current: int
    form: Optional[int]


@dataclass(frozen=True)
class LOp:
    """A row. The fields are the UIR word's (D, T, L, R, N, X), with the same meaning per opcode, but
    full integers: the widths are encode's concern. Every field is an index (None: none) but N of
    SCALE_ROW and CLEAR (a component count) and of SOURCE (its template).

      vertex           d out edge, t coupling (scale), l, r edges, x factor
      kernel           d out edge, t binding instance, l, r edges (None: a null operand), x factor
      propagator       d the current's edge, x factor
      closure          d amplitude, t component-factor list, l, r edges, x factor
      scale            d the current's edge, n components, x factor
      clear            d the current's edge, n components
      source           d the current's edge (its form: the wavefunction's momentum), t mass (a scale:
                       its real part), n template (source_template), l embedding (None: none), x factor
                       (None with an embedding, whose entries carry their own)
      clear_amplitude  d amplitude
      copy             d, l edges (destination, source; d's components are copied), x factor
      scatter          d, l edges (as copy), x factor; set = mode "="
      reduce           d output (helicity), t amplitude, x the amplitude's scale (None: none), n weight

    Flags: set = mode "=" (vertex, kernel, scatter); first, last = a fused run (vertex rows only)."""
    op: Opcode
    set: bool = False
    first: bool = False
    last: bool = False
    d: Optional[int] = None
    t: Optional[int] = None
    l: Optional[int] = None
    r: Optional[int] = None
    n: Optional[int] = None
    x: Optional[int] = None


def source_template(family: PhysicalType, antiparticle: bool, helicity: int, chirality: int) -> int:
    """A SOURCE row's template (its N): family bits 0-2, antiparticle 3, helicity + 2 bits 4-6,
    chirality + 1 bits 7-8"""
    if not (-2 <= helicity <= 5 and -1 <= chirality <= 2):
        raise ValueError(f"source helicity {helicity}, chirality {chirality}")
    return int(family) | int(antiparticle) << 3 | (helicity + 2) << 4 | (chirality + 1) << 7


def source_fields(op: LOp) -> tuple[PhysicalType, bool, int, int]:
    """a SOURCE row's template as (family, antiparticle, helicity, chirality)"""
    n = op.n
    return (PhysicalType(n & 7), (n >> 3) & 1 == 1, ((n >> 4) & 7) - 2, ((n >> 7) & 3) - 1)


def source_op(edge: int, family: PhysicalType, antiparticle: bool, helicity: int, chirality: int, mass: int,
              factor: int, embedding: Optional[int] = None) -> LOp:
    """A source row: factor × the wavefunction of the template at its edge's form into its edge's
    current; with an embedding, component j of the current = factor_j × wavefunction component k_j
    (None: zero)"""
    return LOp(Opcode.SOURCE, d=edge, t=mass, n=source_template(family, antiparticle, helicity, chirality),
               l=embedding, x=factor if embedding is None else None)


# an embedding entry: (the current's component, the wavefunction's component or None for zero, factor)
EmbeddingEntry = tuple[int, Optional[int], int]


@dataclass(frozen=True)
class LStub:
    """A kernel stub: a kernel with its input layout compiled in. The layout is the binding's inputs
    with each scale replaced by its slot in the scale set (KernelInput(SCALE, slot, False)), so
    bindings that differ only in their scales share a stub."""
    kernel: Kernel
    inputs: tuple[KernelInput, ...]
    outputs: tuple[int, ...]   # the components of out, one per output pair


@dataclass(frozen=True)
class Lowered:
    version: str
    process: str
    colour: str
    legs: int
    crossing: Optional[tuple[tuple[int, float], ...]]   # program leg → (caller leg, sign); None: the identity
    forms: tuple[Form, ...]
    scales: tuple[complex, ...]
    factors: tuple[complex, ...]
    alpha_s: Optional[tuple[float, Optional[int]]]       # (value, power of |M|²)
    helicities: tuple[tuple[int, ...], ...]
    currents: tuple[LCurrent, ...]
    planes: int                                          # the arena's planes
    edges: tuple[LEdge, ...]                             # edge c is current c with its own momentum; extra pairs follow
    lists: tuple[tuple[int, ...], ...]                   # closure component factors
    embeddings: tuple[tuple[EmbeddingEntry, ...], ...]   # source embeddings
    stubs: tuple[LStub, ...]
    scalesets: tuple[tuple[int, ...], ...]
    instances: tuple[tuple[int, int], ...]               # (stub, scale set)
    ops: tuple[LOp, ...]
    amplitudes: int
    weights: tuple[float, ...]                           # the REDUCE rows' weights
    group_factor: tuple[int, int]                        # applied to every output once, after the rows
    fixed_layout: bool                                   # the layout is the producer's (RustiCol's)

    def output_count(self) -> int:
        """the helicity outputs"""
        return len(self.helicities)

    def edge_current(self, e: Optional[int]) -> Optional[int]:
        return None if e is None else self.edges[e].current

    def edge_form(self, e: Optional[int]) -> Optional[int]:
        return None if e is None else self.edges[e].form

    def edge_layout(self, e: int) -> LCurrent:
        return self.currents[self.edges[e].current]

    def edge_plane(self, e: int) -> int:
        return self.edge_layout(e).start

    def plane_range(self, e: int) -> range:
        """an edge's planes"""
        c = self.edge_layout(e)
        return range(c.start, c.start + c.components)

    def __repr__(self) -> str:
        return (f"Lowered(UIR v{self.version}: {self.process}, {self.colour}; {len(self.ops)} rows, "
                f"{len(self.currents)} currents on {self.planes} planes, {len(self.edges)} edges)")


def describe(l: Lowered, op: LOp) -> str:
    """A row as text, for the C comments and dumps (pac-julia's describe, the same text): main.json's
    names (c<i>, q<i>, s<i>, x<i>, a<i>, from 0), an edge as its current with the momentum it is read
    with (c12@q3); a binding instance, list, embedding and weight counted from 1, as pac-julia writes
    them."""
    def edge(e):
        if e is None:
            return "-"
        f = l.edge_form(e)
        return f"c{l.edge_current(e)}" + ("" if f is None else f"@q{f}")

    def x(v):
        return -1 if v is None else v   # pac-julia's x$(0 - 1) for no factor

    o = op.op
    mode = "=" if op.set else "+="
    if not is_vertex(o) or (op.first and op.last):
        run = ""
    elif op.first:
        run = " [run starts]"
    elif op.last:
        run = " [run ends]"
    else:
        run = " [in run]"
    if is_vertex(o) or is_kernel(o):
        what = f"s{op.t}" if is_vertex(o) else f"binding instance {op.t + 1}"
        return f"{o.name} {edge(op.d)} {mode} {edge(op.l)}, {edge(op.r)}; {what} x{x(op.x)}{run}"
    if is_propagator(o):
        return f"{o.name} {edge(op.d)}; x{x(op.x)}"
    if o == Opcode.CLOSURE:
        return f"CLOSURE a{op.d} += {edge(op.l)} . {edge(op.r)}; list {op.t + 1} x{x(op.x)}"
    if o == Opcode.SCALE_ROW:
        return f"SCALE {edge(op.d)}, {op.n} components; x{x(op.x)}"
    if o == Opcode.SOURCE:
        (family, anti, h, ch) = source_fields(op)
        how = f"x{x(op.x)}" if op.l is None else f"embedding {op.l + 1}"
        return (f"SOURCE {edge(op.d)} = {family.name}{' antiparticle' if anti else ''}, helicity {h}, chirality {ch}, "
                f"mass s{op.t}; {how}")
    if o == Opcode.CLEAR_AMPLITUDE:
        return f"CLEAR_AMPLITUDE a{op.d}"
    if o == Opcode.COPY:
        return f"COPY {edge(op.d)} = {edge(op.l)}; x{x(op.x)}"
    if o == Opcode.SCATTER:
        return f"SCATTER {edge(op.d)} {mode} {edge(op.l)}; x{x(op.x)}"
    if o == Opcode.REDUCE:
        sc = "" if op.x is None else f" (× x{op.x})"
        return f"REDUCE out[{op.d}] += w{op.n + 1} × |a{op.t}{sc}|²"
    return f"{o.name} {edge(op.d)}, {op.n} components"


def overlaps(a: range, b: range) -> bool:
    return max(a.start, b.start) < min(a.stop, b.stop)


def no_destination(op: LOp) -> bool:
    """the rows whose d is not an edge (an amplitude, an output)"""
    return op.op in (Opcode.CLOSURE, Opcode.CLEAR_AMPLITUDE, Opcode.REDUCE)


def operand_edges(op: LOp) -> tuple:
    """the edges a row reads as operands (a source's l is its embedding, not an edge)"""
    if op.op in (Opcode.SOURCE, Opcode.CLEAR_AMPLITUDE, Opcode.REDUCE):
        return ()
    return (op.l, op.r)

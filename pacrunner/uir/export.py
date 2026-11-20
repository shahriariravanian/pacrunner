"""RustiCol's recurrence export (`pyamplicol export-recurrence LC_DIR -o FILE`, pyAmpliCol 1.0.x; schema
2, execution semantics rusticol-batched-f64-v1, pyAmpliCol's docs/reference/recurrence-export.md)
read into Lowered programs, row for row: pac-julia's src/recurrence_export.jl.
`load_recurrence_export(path)` gives one Lowered per program entry, in the file's order (a
process-group member or alias: its representative's rows with its own crossing, group factor and
helicity labels); `recurrence_references(path)` the reference values the file holds
(`--reference-points`).

The mapping (export → UIR v0.8):
  currents  the producer's layout is kept (fixed_layout): a Lowered current is a range of the
            export's component offsets, (offset, components, type), made for the rows that address
            it; ranges of different currents may overlap, as the producer's storage reuse does. Its
            type is the one its row reads it as (vertex, propagator, plain source), else UNTYPED.
  edges     (current, form); a current's own edge has the first form a row reads it with.
  values    couplings, masses and kernel literals interned into `scales`; every other complex value
            (row factors, closure component factors, embedding factors, replay amplitude scales) into
            `factors`; bit for bit (-0.0 and 0.0 differ).
  runs      concatenated: each run's body (its sources from its own table), then its reduction as
            REDUCE rows (the replay scale, when applied, as REDUCE's factor).
  kernels   each binding a stub (its literals as scale slots), its relative projections the stub's
            inputs (parent 0/1 → IN_L/IN_R, destination → OUT, momentum 0/1 → MOMENTUM_U/V, factor →
            FACTOR); the kernels are the saved applications the file carries (Kernel), checked
            against their SHA-256.
Unsupported (an ExportError): spin-2 sources, kernels with an amplitude destination or input, outputs
that are not (re, im) pairs of one component.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from os import PathLike
from typing import Optional, Union

from .types import (TYPE_COMPONENTS, UIR_VERSION, Form, InputKind, Kernel, KernelInput, LCurrent, LEdge, LOp, Lowered,
                    LStub, Opcode, PhysicalType, no_destination, operand_edges, source_op)
from .validate import FAMILY_TYPES, PROPAGATOR_OPS, VERTEX_OPS, Table, complex_key, float_key, validate

EXPORT_SCHEMA = 2
EXPORT_SEMANTICS = "rusticol-batched-f64-v1"

# RustiCol's versioned intrinsic names → the operations' names (validate.py's VERTEX_OPS, PROPAGATOR_OPS)
EXPORT_VERTICES = {
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1": "weyl_vector_a",
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1": "weyl_vector_b",
    "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1": "tensor_vector",
    "rusticol.recurrence-intrinsic.vector-wedge-vector.v1": "vector_wedge",
    "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1": "three_vector",
}
EXPORT_PROPAGATORS = {
    "rusticol.recurrence-intrinsic.weyl-propagator-a.v1": "weyl_a",
    "rusticol.recurrence-intrinsic.weyl-propagator-b.v1": "weyl_b",
    "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1": "vector",
}

PathOrDict = Union[str, PathLike, dict]


class ExportError(ValueError):
    def __init__(self, msg: str):
        super().__init__(f"recurrence export: {msg}")


def _cvalue(z) -> complex:
    return complex(float(z["re"]), float(z["im"]))


def read_export(path: PathOrDict) -> dict:
    """the file, checked: its schema and semantics (a dict already read is checked as it is)"""
    if isinstance(path, dict):
        d, name = path, "the export"
    else:
        with open(path, "rb") as f:
            d = json.load(f)
        name = str(path)
    if d.get("schema_version") != EXPORT_SCHEMA:
        raise ExportError(f"{name}: schema_version {d.get('schema_version')}, this reader takes {EXPORT_SCHEMA}")
    if d.get("execution_semantics") != EXPORT_SEMANTICS:
        raise ExportError(f"{name}: execution_semantics {d.get('execution_semantics')}, not {EXPORT_SEMANTICS}")
    return d


def export_kernels(d: dict) -> dict[str, Kernel]:
    """the kernels of the file by SHA-256, each saved application decoded and checked against it"""
    out = {}
    for k in d["kernels"]:
        sha = str(k["sha256"])
        data = base64.b64decode(k["application_base64"])
        if hashlib.sha256(data).hexdigest() != sha:
            raise ExportError(f"kernel {sha}: its application has another SHA-256")
        out[sha] = Kernel(sha, data)
    return out


# ---- one program entry → Lowered -----------------------------------------------------------------------

@dataclass
class _Binding:
    """a kernel binding of the file: the stub's inputs (literals as slots of `scaleset`), its output
    components, the kernel, and whether it stores its pairs directly"""
    inputs: tuple[KernelInput, ...]
    scaleset: tuple[int, ...]
    outputs: tuple[int, ...]
    kernel: Kernel
    direct: bool


def _form_of(x) -> int:
    if x is None:
        raise ExportError("an operand without a momentum form")
    return int(x)


def _set_mode(m) -> bool:
    if m == "set":
        return True
    if m == "add":
        return False
    raise ExportError(f"mode {m}")


@dataclass
class _Builder:
    """What _import_program collects while it reads an entry: the Lowered's tables, and the rows with
    their d, l, r still edge requests (an index into `requests`), resolved to edges at the end
    (_resolved_ops) when each current's own form is known."""
    scales: Table = field(default_factory=lambda: Table(complex_key))
    factors: Table = field(default_factory=lambda: Table(complex_key))
    layouts: Table = field(default_factory=Table)            # the currents: (offset, components, type)
    owns: list = field(default_factory=list)                 # each current's own form (None: none yet)
    requests: list = field(default_factory=list)             # (current, form or None: any edge of it)
    lists: Table = field(default_factory=Table)
    embeddings: list = field(default_factory=list)
    stubs: Table = field(default_factory=lambda: Table(lambda s: (s.kernel, s.inputs, s.outputs)))
    scalesets: Table = field(default_factory=Table)
    instances: Table = field(default_factory=Table)
    weights: Table = field(default_factory=lambda: Table(float_key))
    bindings: dict = field(default_factory=dict)             # by the file's id
    rows: list = field(default_factory=list)

    def scale(self, z: complex) -> int:
        return self.scales.add(z)

    def factor(self, z: complex) -> int:
        return self.factors.add(z)

    def want(self, offset, components, ty: PhysicalType, form: Optional[int] = None) -> int:
        """a request for an edge of the current at (offset, components, type), with `form` when the
        row reads its momentum; the first form asked of a current becomes its own"""
        c = self.layouts.add((int(offset), int(components), ty))
        if c == len(self.owns):
            self.owns.append(None)
        if form is not None and self.owns[c] is None:
            self.owns[c] = form
        self.requests.append((c, form))
        return len(self.requests) - 1

    def instance(self, x: _Binding) -> int:
        """the binding instance (stub, scale set) of a binding, both interned"""
        s = self.stubs.add(LStub(x.kernel, x.inputs, x.outputs))
        return self.instances.add((s, self.scalesets.add(x.scaleset)))


def _binding_input(b: _Builder, scaleset: list, i, id_) -> KernelInput:
    """one input of a binding; a literal takes the next slot of `scaleset`"""
    kind = i["kind"]
    if kind == "parent":
        return KernelInput(InputKind.IN_L if int(i["parent"]) == 0 else InputKind.IN_R, int(i["component"]), bool(i["imaginary"]))
    if kind == "destination":
        if i["amplitude"]:
            raise ExportError(f"binding {id_}: an amplitude input")
        return KernelInput(InputKind.OUT, int(i["component"]), bool(i["imaginary"]))
    if kind == "momentum":
        return KernelInput(InputKind.MOMENTUM_U if int(i["operand"]) == 0 else InputKind.MOMENTUM_V, int(i["component"]), False)
    if kind == "literal":
        scaleset.append(b.scale(complex(float(i["value"]), 0.0)))
        return KernelInput(InputKind.SCALE, len(scaleset) - 1, False)
    if kind == "factor":
        return KernelInput(InputKind.FACTOR, None, bool(i["imaginary"]))
    raise ExportError(f"binding {id_}: input kind {kind}")


def _binding_outputs(o, id_) -> tuple[int, ...]:
    """the output components of a binding: its outputs are (re, im) pairs of one destination component each"""
    if len(o) % 2 != 0:
        raise ExportError(f"binding {id_}: {len(o)} outputs")
    outputs = []
    for j in range(0, len(o), 2):
        (re, im) = (o[j], o[j + 1])
        if not (all(x["kind"] == "destination" and not x["amplitude"] for x in (re, im))
                and re["component"] == im["component"] and not re["imaginary"] and im["imaginary"]):
            raise ExportError(f"binding {id_}: outputs {j}, {j + 1} are not one component's (re, im)")
        outputs.append(int(re["component"]))
    return tuple(outputs)


def _import_binding(b: _Builder, x, kernels: dict[str, Kernel]) -> None:
    id_, sha = x["id"], str(x["sha256"])
    if sha not in kernels:
        raise ExportError(f"binding {id_}: no kernel {sha} in the file")
    scaleset: list = []
    inputs = tuple(_binding_input(b, scaleset, i, id_) for i in x["inputs"])
    b.bindings[int(id_)] = _Binding(inputs, tuple(scaleset), _binding_outputs(x["outputs"], id_), kernels[sha], bool(x["direct"]))


# ---- the rows of the body, one function per instruction kind

def _import_clear(b: _Builder, i) -> None:
    n = int(i["component_count"])
    b.rows.append(LOp(Opcode.CLEAR, d=b.want(i["current"], n, PhysicalType.UNTYPED), n=n))


def _import_clear_amplitude(b: _Builder, i) -> None:
    b.rows.append(LOp(Opcode.CLEAR_AMPLITUDE, d=int(i["amplitude"])))


def _import_source(b: _Builder, s) -> None:
    """`s`: the run's source the instruction names"""
    family = s["family"]
    if family not in FAMILY_TYPES:
        raise ExportError(f"source family {family}")
    ty = FAMILY_TYPES[family]
    # component_count is the wavefunction's; an embedding writes the components it lists
    # (union sources: a Weyl spinor into a 4-component slot, the rest +0)
    n = int(s["component_count"])
    if n != TYPE_COMPONENTS[ty]:
        raise ExportError(f"a {family} source of {n} components")
    embedded = len(s["embedding"]) > 0
    width = max(int(e["component"]) for e in s["embedding"]) + 1 if embedded else n
    d = b.want(s["current"], width, PhysicalType.UNTYPED if embedded else ty, _form_of(s["momentum_form"]))
    emb = None
    if embedded:
        b.embeddings.append(tuple((int(e["component"]), None if e["source_component"] is None else int(e["source_component"]),
                                   b.factor(_cvalue(e["factor"]))) for e in s["embedding"]))
        emb = len(b.embeddings) - 1
    b.rows.append(source_op(d, ty, s["orientation"] == "antiparticle", int(s["helicity"]), int(s["chirality"]),
                            b.scale(_cvalue(s["mass"])), b.factor(_cvalue(s["factor"])), embedding=emb))


def _import_vertex(b: _Builder, i) -> None:
    name = EXPORT_VERTICES.get(i["vertex"])
    if name is None:
        raise ExportError(f"vertex {i['vertex']}")
    (opc, tl, tr, tout) = VERTEX_OPS[name]
    reads = opc == Opcode.V_THREE_VECTOR
    if i["right"] is None:
        raise ExportError(f"vertex {i['vertex']} without a right operand")
    l = b.want(i["left"], TYPE_COMPONENTS[tl], tl, _form_of(i["left_momentum"]) if reads else None)
    r = b.want(i["right"], TYPE_COMPONENTS[tr], tr, _form_of(i["right_momentum"]) if reads else None)
    d = b.want(i["current"], TYPE_COMPONENTS[tout], tout)
    b.rows.append(LOp(opc, set=_set_mode(i["mode"]), first=True, last=True, d=d, t=b.scale(_cvalue(i["coupling"])), l=l, r=r,
                      x=b.factor(_cvalue(i["factor"]))))


def _import_propagator(b: _Builder, i) -> None:
    name = EXPORT_PROPAGATORS.get(i["propagator"])
    if name is None:
        raise ExportError(f"propagator {i['propagator']}")
    (opc, ty) = PROPAGATOR_OPS[name]
    if int(i["component_count"]) != TYPE_COMPONENTS[ty]:
        raise ExportError(f"propagator {name} of {i['component_count']} components")
    d = b.want(i["current"], TYPE_COMPONENTS[ty], ty, _form_of(i["momentum_form"]))
    b.rows.append(LOp(opc, d=d, x=b.factor(_cvalue(i["factor"]))))


def _import_scale(b: _Builder, i) -> None:
    n = int(i["component_count"])
    b.rows.append(LOp(Opcode.SCALE_ROW, d=b.want(i["current"], n, PhysicalType.UNTYPED), n=n, x=b.factor(_cvalue(i["factor"]))))


def _import_copy(b: _Builder, i, scatter: bool) -> None:
    """a copy or a scatter (which has a mode)"""
    n = int(i["component_count"])
    d = b.want(i["current"], n, PhysicalType.UNTYPED)
    l = b.want(i["source"], n, PhysicalType.UNTYPED)
    b.rows.append(LOp(Opcode.SCATTER if scatter else Opcode.COPY, set=scatter and _set_mode(i["mode"]), d=d, l=l,
                      x=b.factor(_cvalue(i["factor"]))))


def _import_closure(b: _Builder, i) -> None:
    cf = tuple(b.factor(_cvalue(z)) for z in i["component_factors"])
    n = len(cf)
    b.rows.append(LOp(Opcode.CLOSURE, d=int(i["amplitude"]), t=b.lists.add(cf), l=b.want(i["left"], n, PhysicalType.UNTYPED),
                      r=b.want(i["right"], n, PhysicalType.UNTYPED), x=b.factor(_cvalue(i["factor"]))))


def _reach(x: _Binding, *kinds: InputKind) -> int:
    """the components of the given kinds a binding reads: the largest one + 1 (0: none)"""
    return max((i.n + 1 for i in x.inputs if i.kind in kinds), default=0)


def _kernel_operand(b: _Builder, x: _Binding, at, part: InputKind, momentum: InputKind, form) -> Optional[int]:
    """a kernel row's operand `at`, which the binding reads through the input kinds `part` (a
    component) and `momentum` (its form): its edge request, or None when the binding reads neither"""
    comps = _reach(x, part)
    reads_form = any(i.kind == momentum for i in x.inputs)
    if comps == 0 and not reads_form:
        return None
    if at is None:
        raise ExportError("a kernel input of a null operand")
    return b.want(at, max(1, comps), PhysicalType.UNTYPED, _form_of(form) if reads_form else None)


def _import_kernel(b: _Builder, i) -> None:
    if i["amplitude"]:
        raise ExportError("a kernel with an amplitude destination")
    x = b.bindings.get(int(i["binding"]))
    if x is None:
        raise ExportError(f"kernel row: no binding {i['binding']}")
    # each operand's components: the largest one the binding reads (at least 1)
    nd = max(1, _reach(x, InputKind.OUT), max((j + 1 for j in x.outputs), default=0))
    l = _kernel_operand(b, x, i["left"], InputKind.IN_L, InputKind.MOMENTUM_U, i["left_momentum"])
    r = _kernel_operand(b, x, i["right"], InputKind.IN_R, InputKind.MOMENTUM_V, i["right_momentum"])
    b.rows.append(LOp(Opcode.KERNEL_DIRECT if x.direct else Opcode.KERNEL, set=_set_mode(i["mode"]),
                      d=b.want(i["destination"], nd, PhysicalType.UNTYPED), t=b.instance(x), l=l, r=r,
                      x=b.factor(_cvalue(i["factor"]))))


_ROWS = {
    "clear": _import_clear,
    "clear_amplitude": _import_clear_amplitude,
    "vertex": _import_vertex,
    "propagator": _import_propagator,
    "scale": _import_scale,
    "copy": lambda b, i: _import_copy(b, i, False),
    "scatter": lambda b, i: _import_copy(b, i, True),
    "closure": _import_closure,
    "kernel": _import_kernel,
}


def _import_row(b: _Builder, i, sources) -> None:
    """one instruction of the body, with the run's sources"""
    kind = i["kind"]
    if kind == "source":
        _import_source(b, sources[int(i["source"])])
    elif kind in _ROWS:
        _ROWS[kind](b, i)
    else:
        raise ExportError(f"instruction {kind}")


def _import_reduction(b: _Builder, t) -> None:
    """one term of a run's reduction"""
    x = b.factor(_cvalue(t["amplitude_scale"])) if t["apply_amplitude_scale"] else None
    b.rows.append(LOp(Opcode.REDUCE, d=int(t["helicity_output"]), t=int(t["amplitude"]), x=x, n=b.weights.add(float(t["weight"]))))


def _resolved_ops(b: _Builder) -> tuple[tuple[LEdge, ...], tuple[LOp, ...]]:
    """The edges and the rows with their requests resolved: current c's own edge first (edge c), then
    the other (current, form) pairs in the order the rows ask for them."""
    edges = Table(values=[LEdge(c, b.owns[c]) for c in range(len(b.layouts))])

    def resolve(k):
        if k is None:
            return None
        (c, f) = b.requests[k]
        return c if f is None or f == b.owns[c] else edges.add(LEdge(c, f))

    ops = []
    for o in b.rows:
        d = o.d if no_destination(o) else resolve(o.d)   # d first, then l and r: the edges' order
        (l, r) = (o.l, o.r) if not operand_edges(o) else (resolve(o.l), resolve(o.r))
        ops.append(dataclasses.replace(o, d=d, l=l, r=r))
    return (tuple(edges.values), tuple(ops))


def _export_alpha_s(p):
    a = p["alpha_s"]
    if a is None:
        return None
    power = a["squared_amplitude_power"]
    return (float(a["value"]["re"]), None if power is None else int(power))


def _export_crossing(p, legs: int):
    """program leg → (caller leg, sign), or None: the identity"""
    x = p["crossing"]
    if x is None:
        return None
    out: list = [None] * legs
    for e in x:
        out[int(e["program_leg"])] = (int(e["caller_leg"]), float(e["sign"]))
    if any(v is None for v in out):
        raise ExportError(f"{p['process_key']}: the crossing misses a leg")
    return tuple(out)


def _export_helicities(p):
    return tuple(tuple(int(h) for h in x["helicities"]) for x in p["helicity_outputs"])


def _export_group(p):
    return (int(p["group_factor"]["numerator"]), int(p["group_factor"]["denominator"]))


def _import_program(p, kernels: dict[str, Kernel]) -> Lowered:
    """The rows of one full program entry, as a Lowered (its own process, colour, crossing, group
    factor and helicities): the bindings, then per run the body (with the run's sources) and the run's
    reduction."""
    b = _Builder()
    for x in p["kernel_bindings"]:
        _import_binding(b, x, kernels)
    for run in p["runs"]:
        for i in p["body"]:
            _import_row(b, i, run["sources"])
        for t in run["reduction"]:
            _import_reduction(b, t)
    (edges, ops) = _resolved_ops(b)
    legs = int(p["external_count"])
    forms = tuple(Form(tuple((int(t["leg"]), float(t["coefficient"])) for t in f)) for f in p["momentum_forms"])
    l = Lowered(
        version=UIR_VERSION, process=str(p["process_key"]), colour=str(p["colour_id"]), legs=legs,
        crossing=_export_crossing(p, legs), forms=forms, scales=tuple(b.scales.values), factors=tuple(b.factors.values),
        alpha_s=_export_alpha_s(p), helicities=_export_helicities(p),
        currents=tuple(LCurrent(offset, n, ty) for (offset, n, ty) in b.layouts.values),
        planes=int(p["current_component_count"]), edges=edges, lists=tuple(b.lists.values), embeddings=tuple(b.embeddings),
        stubs=tuple(b.stubs.values), scalesets=tuple(b.scalesets.values), instances=tuple(b.instances.values), ops=ops,
        amplitudes=int(p["amplitude_count"]), weights=tuple(b.weights.values), group_factor=_export_group(p), fixed_layout=True)
    return validate(l)


def _key(p) -> tuple[str, str]:
    return (str(p["process_key"]), str(p["colour_id"]))


def load_recurrence_export(path: PathOrDict) -> list[Lowered]:
    """The programs of a RustiCol recurrence export (`pyamplicol export-recurrence`, schema 2), one
    Lowered per entry of `programs` in the file's order; a member or alias entry is its
    representative's rows with its own process, colour, crossing, group factor and helicities.
    `path`: the file, or its JSON already read."""
    d = read_export(path)
    kernels = export_kernels(d)
    # a (process, colour flow) names one program: a second entry would silently replace the first
    seen = set()
    for p in d["programs"]:
        key = _key(p)
        if key in seen:
            raise ExportError(f"{key[0]} {key[1]}: more than one program")
        seen.add(key)
    full = {_key(p): _import_program(p, kernels) for p in d["programs"] if "program_reference" not in p}
    out = []
    for p in d["programs"]:
        key = _key(p)
        if "program_reference" not in p:
            out.append(full[key])
            continue
        r = p["program_reference"]
        rep = full.get(_key(r))
        if rep is None:
            raise ExportError(f"{key[0]} {key[1]}: no program {r['process_key']} {r['colour_id']}")
        hel = _export_helicities(p)
        if len(hel) != rep.output_count():
            raise ExportError(f"{key[0]} {key[1]}: {len(hel)} helicities, its program {rep.output_count()}")
        out.append(dataclasses.replace(rep, process=key[0], colour=key[1], crossing=_export_crossing(p, rep.legs),
                                       group_factor=_export_group(p), helicities=hel))
    return out


def recurrence_references(path: PathOrDict) -> list:
    """The reference values of a recurrence export written with `--reference-points`, one entry per
    program entry (as load_recurrence_export's order): None, or (momenta, values) with momenta[point]
    the caller's legs as (E, px, py, pz) tuples and values[point] the outputs in helicity order."""
    d = read_export(path)
    out = []
    for p in d["programs"]:
        r = p.get("reference_values")
        if r is None:
            out.append(None)
            continue
        if any(len(v) != len(p["helicity_outputs"]) for v in r["values"]):
            raise ExportError(f"{p['process_key']}: reference values not one per helicity output")
        momenta = [[tuple(float(c) for c in leg) for leg in point] for point in r["momenta"]]
        values = [[float(x) for x in v] for v in r["values"]]
        out.append((momenta, values))
    return out

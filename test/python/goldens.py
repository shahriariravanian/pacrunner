"""The goldens (tools/goldens/make_goldens.jl: pac-julia's outputs) and pacrunner's values in their JSON
form, for comparing one with the other.

Where the goldens are: PACRUNNER_GOLDENS (a folder of OUT/<export name>/ folders), else the corpus
~/symjit-data/pacrunner-goldens when it exists; test/data/goldens (the fixture exports' goldens,
committed) always."""

import json
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_EXPORTS = ROOT / "test" / "data" / "export"
FIXTURE_GOLDENS = ROOT / "test" / "data" / "goldens"
CORPUS_GOLDENS = Path(os.environ.get("PACRUNNER_GOLDENS", Path.home() / "symjit-data" / "pacrunner-goldens"))


def golden_sets():
    """(name, golden folder, export path) of every golden set found: the fixtures, then the corpus"""
    out = []
    for base in (FIXTURE_GOLDENS, CORPUS_GOLDENS):
        if not base.is_dir():
            continue
        for d in sorted(p for p in base.iterdir() if (p / "manifest.json").is_file()):
            m = json.loads((d / "manifest.json").read_text())
            export = Path(m["export_path"])
            if not export.is_file():                     # a fixture: its export in test/data/export
                export = FIXTURE_EXPORTS / export.name
            out.append((f"{base.name}/{d.name}", d, export, m))
    return out


# ---- values as the goldens write them

def bits(x: float) -> str:
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def cbits(z: complex) -> list:
    return [bits(z.real), bits(z.imag)]


def op_json(op):
    from pacrunner.uir import fields
    row = {"op": op.op.name, "set": op.set, "first": op.first, "last": op.last}
    for (s, _, _) in fields(op.op):
        row[s] = getattr(op, s)
    return row


def lowered_json(l):
    return {
        "version": l.version,
        "process": l.process,
        "colour": l.colour,
        "legs": l.legs,
        "crossing": None if l.crossing is None else [[leg, bits(sign)] for (leg, sign) in l.crossing],
        "forms": [[[leg, bits(c)] for (leg, c) in f.terms] for f in l.forms],
        "scales": [cbits(z) for z in l.scales],
        "factors": [cbits(z) for z in l.factors],
        "alpha_s": None if l.alpha_s is None else {"value": bits(l.alpha_s[0]), "power": l.alpha_s[1]},
        "helicities": [list(h) for h in l.helicities],
        "currents": [[c.start, c.components, c.type.name] for c in l.currents],
        "planes": l.planes,
        "edges": [[e.current, e.form] for e in l.edges],
        "lists": [list(x) for x in l.lists],
        "embeddings": [[list(entry) for entry in x] for x in l.embeddings],
        "stubs": [{"kernel": s.kernel.sha256, "name": s.kernel.name,
                   "inputs": [{"kind": i.kind.name, "n": i.n, "im": i.im} for i in s.inputs],
                   "outputs": list(s.outputs)} for s in l.stubs],
        "scalesets": [list(x) for x in l.scalesets],
        "instances": [list(x) for x in l.instances],
        "ops": [op_json(o) for o in l.ops],
        "amplitudes": l.amplitudes,
        "weights": [bits(w) for w in l.weights],
        "group_factor": list(l.group_factor),
        "fixed_layout": l.fixed_layout,
    }


def encoded_json(e):
    return {"version": e.version, "rows": ["0x%016x" % w for w in e.rows], "edges": list(e.edges)}


def first_difference(a, b, path="$"):
    """where two JSON values first differ (None: equal)"""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return f"{path}.{k}: only in {'golden' if k in a else 'ours'}"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None if len(a) == len(b) else f"{path}: {len(a)} entries in the golden, {len(b)} in ours"
    if type(a) is not type(b) or a != b:
        return f"{path}: golden {a!r}, ours {b!r}"
    return None

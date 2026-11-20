"""The optional passes over a Lowered (UIR v0.8): pac-julia's src/passes.jl. Each returns a new,
validated Lowered whose values are the input's bit for bit; `optimize` runs both.

  sort_runs  each segment of consecutive vertex rows stably sorted by destination, when safe
             (pacuda's sort_contribution_runs), so the rows on one destination become adjacent
  fuse       adjacent vertex rows on one destination joined into fused runs (FIRST ... LAST)

Order: sort_runs before fuse (it sorts single rows, not runs).

pac-julia's third pass, reuse_planes (a liveness layout), is not carried over: it leaves a fixed
layout (the producer's, which every program read from a recurrence export has) as it is.
"""

from __future__ import annotations

import dataclasses

from .types import Lowered, is_vertex, overlaps
from .validate import validate


def sort_runs(l: Lowered) -> Lowered:
    """A segment of consecutive vertex rows is sorted (stably, by its destinations' first planes) when
    no two different destinations share a plane and no row reads a plane any row of the segment
    writes. Then every destination keeps its own rows in their order, so its sums are unchanged, and
    no row sees another destination's partial value. The rows must be unfused (first and last)."""
    ops = list(l.ops)
    k = 0
    while k < len(ops):
        if not is_vertex(ops[k].op):
            k += 1
            continue
        e = k
        while e < len(ops) and is_vertex(ops[e].op):
            e += 1
        seg = range(k, e)
        k = e
        if not all(ops[i].first and ops[i].last for i in seg):
            raise ValueError("sort_runs comes before fuse: a segment holds a fused run")
        dests = list(dict.fromkeys(ops[i].d for i in seg))
        ranges = [l.plane_range(d) for d in dests]
        safe = all(not overlaps(ranges[i], ranges[j]) for i in range(len(ranges)) for j in range(i + 1, len(ranges)))
        safe = safe and all(not overlaps(l.plane_range(x), r) for i in seg for x in (ops[i].l, ops[i].r) for r in ranges)
        if not safe:
            continue
        ops[seg.start:seg.stop] = sorted(ops[seg.start:seg.stop], key=lambda o: l.currents[l.edge_current(o.d)].start)
    return validate(dataclasses.replace(l, ops=tuple(ops)))


def fuse(l: Lowered) -> Lowered:
    """A vertex row joins the run of the vertex row before it when both write the same destination, it
    is "+=" (a "=" row starts its own run), and it reads none of the destination's planes."""
    ops = list(l.ops)
    for k in range(1, len(ops)):
        (a, b) = (ops[k - 1], ops[k])
        if not (is_vertex(a.op) and is_vertex(b.op) and a.d == b.d and not b.set):
            continue
        dest = l.plane_range(b.d)
        if any(overlaps(l.plane_range(e), dest) for e in (b.l, b.r)):
            continue
        ops[k - 1] = dataclasses.replace(a, last=False)
        ops[k] = dataclasses.replace(b, first=False)
    return validate(dataclasses.replace(l, ops=tuple(ops)))


def optimize(l: Lowered) -> Lowered:
    if not l.fixed_layout:
        raise ValueError("optimize: a Lowered without a fixed layout (pac-julia's reuse_planes is not ported)")
    return fuse(sort_runs(l))

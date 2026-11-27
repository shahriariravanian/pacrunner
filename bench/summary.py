"""summary.py LOGDIR: the runs of bench.py. Wall: min and median of the uninstrumented runs (pass A); ×stock,
×cpu: the min walls' ratios. ME µs/pt: the time inside umami_matrix_element per event (pass B, the timer
run; thread-µs summed over MadSpace's threads, so a GPU variant's includes threads waiting for the shared
GPU); cuda-full: pass C, marked *, 64k-event batches, synchronized (µs/event over all channels). σ against
stock's; events: the same bits as stock's, as cpu's?"""
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

L = sys.argv[1]
ORDER = ["stock", "cpu", "simd-avx2", "simd-neon", "simd-split-avx2", "simd-split-neon", "simd-avx512",
         "simd-split-avx512", "cuda-f64", "cuda-mixed", "metal-mixed", "cuda-full-f64", "cuda-full-mixed"]


def same(a, b):
    common = [c for c in a.dtype.names if c in b.dtype.names]
    return a.shape == b.shape and all(np.array_equal(a[c], b[c]) for c in common)


def me(d, v):
    t = os.path.join(d, f"timer-{v}.timer")
    if os.path.exists(t):
        rows = [l.split() for l in open(t) if l.strip() and not l.startswith("ratio:")]
        if rows:
            return f"{1e6 * sum(float(r[4]) for r in rows) / sum(int(r[3]) for r in rows):.3f}"
    p = os.path.join(d, f"points-{v}.log")
    if os.path.exists(p):
        m = re.search(r"µs/event runtime ([0-9.]+)", open(p).read())
        if m:
            return m.group(1) + "*"
    return ""


machine = os.path.join(L, "machine.txt")
if os.path.exists(machine):
    print(open(machine).read().rstrip())
for name in sorted(os.listdir(L)):
    d = os.path.join(L, name)
    if not os.path.isdir(d):
        continue
    present = {os.path.basename(f)[5:].rsplit("-", 1)[0] for f in glob.glob(os.path.join(d, "wall-*-[0-9].wall"))}
    print(f"\n{name}")
    print(f"  {'variant':18} {'wall min':>9} {'median':>8} {'×stock':>7} {'×cpu':>6} {'ME µs/pt':>9} {'σ (pb)':>22} "
          f"{'σ/σ_stock-1':>11}  events")
    ref = {}
    for v in [v for v in ORDER if v in present]:
        walls = [float(open(f).read()) for f in sorted(glob.glob(os.path.join(d, f"wall-{v}-[0-9].wall")))]
        dirf = os.path.join(d, f"wall-{v}-1.dir")
        folder = open(dirf).read().strip() if os.path.exists(dirf) else ""
        if not walls or not folder:
            continue
        p = json.load(open(os.path.join(folder, "info.json")))["process"]
        sigma, err = float(p["mean"]), float(p["error"])
        ev = np.load(os.path.join(folder, "events.npy"))
        ref[v] = (min(walls), sigma, ev)
        sw, ss, sev = ref.get("stock", (None, None, None))
        cw, _, cev = ref.get("cpu", (None, None, None))
        tags = []
        if v != "stock" and sev is not None:
            tags.append("= stock" if same(ev, sev) else "≠ stock")
        if v not in ("stock", "cpu") and cev is not None:
            tags.append("= cpu" if same(ev, cev) else "≠ cpu")
        xs = f"{sw / min(walls):6.2f}x" if sw else ""
        xc = f"{cw / min(walls):5.2f}x" if cw and v != "stock" else ""
        rel = "" if v == "stock" or not ss else f"{sigma / ss - 1:+.1e}"
        print(f"  {v:18} {min(walls):8.2f}s {statistics.median(walls):7.2f}s {xs:>7} {xc:>6} {me(d, v):>9}"
              f" {sigma:14.6g} ± {err:<6.3g} {rel:>11}  {', '.join(tags)}")

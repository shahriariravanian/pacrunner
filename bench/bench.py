"""bench.py: PacRunner's benchmark on this machine: every variant of the suite that this machine runs, on the
benchmark processes (MadGraph7 outputs, 100k events, all threads, 64k batches: make_outputs.sh). Usually
run through bench_full.sh or bench_cpu.sh (run.sh), which set the paths and write the summary.

  python bench.py --suite full|cpu --bench DIR --logs DIR --pacrunner DIR --runpy PYTHON --python PYTHON
                  [--names a,b] [--variants v,w] [--repeats 3] [--skip-prepare] [--list]

Variants (the suite "cpu" is "full" without the GPU ones; a variant runs where the machine has it):

  stock                 the output's stock pyAmpliCol library (bin/generate_events)
  cpu                   PacRunner's C runner, f64
  simd-avx2, simd-neon  one symjit SIMD kernel per program (AVX2 on x86-64, NEON on aarch64), f64
  simd-split-avx2, simd-split-neon   the split kernels (f64 kernel A, f32 kernel B)
  simd-avx512, simd-split-avx512     the same with AVX-512 (x86-64 CPUs that have it)
  cuda-f64, cuda-mixed  CUDA libraries, MadSpace on the CPU (the GPU shared by its threads)
  metal-mixed           Metal libraries (macOS: mixed f32/df64), MadSpace on the CPU
  cuda-full-f64, cuda-full-mixed     everything on the GPU (pacrunner device cuda_full, MadSpace's GPU runtime):
                        on NAME_gpu, a copy of the output with run.device = ["cuda"], made here when missing;
                        leading colour only (no full-colour process)

Passes: A (wall) REPEATS runs of every variant, uninstrumented; B (timer) one run each with PAC_UMAMI_TIMER
(the time inside umami_matrix_element; not cuda-full, whose calls return once queued on MadSpace's stream);
C (cuda-full only) its libraries on 64k-event batches on a CUDA context, synchronized after each call
(test/bench_libraries.py, best of 5; MadSpace's CUDA build and CuPy). Logs in LOGS/<name>/: wall-<v>-<r>.wall,
.dir (the run's Events folder), .log; timer-<v>.timer; points-<v>.log; summary.py reads them.

--runpy runs MadGraph7 (the outputs' python); --python builds PacRunner's libraries; pacrunner (the
--pacrunner tree) and symjit must be importable by both, through PYTHONPATH."""
import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import time

# label: (pacrunner device, precision, needs, on the _gpu copy)
VARIANTS = {
    "stock": (None, None, None, False),
    "cpu": ("cpu", "f64", None, False),
    "simd-avx2": ("simd-avx2", "f64", "simd-avx2", False),
    "simd-neon": ("simd-neon", "f64", "simd-neon", False),
    "simd-split-avx2": ("simd-avx2", "split", "simd-avx2", False),
    "simd-split-neon": ("simd-neon", "split", "simd-neon", False),
    "simd-avx512": ("simd-avx512", "f64", "simd-avx512", False),
    "simd-split-avx512": ("simd-avx512", "split", "simd-avx512", False),
    "cuda-f64": ("cuda", "f64", "cuda", False),
    "cuda-mixed": ("cuda", "mixed", "cuda", False),
    "metal-mixed": ("metal", "mixed", "metal", False),
    "cuda-full-f64": ("cuda_full", "f64", "cuda-full", True),
    "cuda-full-mixed": ("cuda_full", "mixed", "cuda-full", True),
}
GPU = {"cuda-f64", "cuda-mixed", "metal-mixed", "cuda-full-f64", "cuda-full-mixed"}
SUITES = {"full": list(VARIANTS), "cpu": [v for v in VARIANTS if v not in GPU]}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--suite", choices=SUITES, default="full")
ap.add_argument("--bench", required=True, help="the folder of the outputs (NAME, NAME_gpu)")
ap.add_argument("--logs", required=True)
ap.add_argument("--runpy", required=True)
ap.add_argument("--python", required=True)
ap.add_argument("--pacrunner", required=True)
ap.add_argument("--names", default="pp_zjj,pp_jjj,gg_ttgg,uu_zgggg,gg_4g")
ap.add_argument("--variants", default="", help="a subset of the suite's variants")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--skip-prepare", action="store_true")
ap.add_argument("--list", action="store_true", help="print the variants this machine runs, and stop")
a = ap.parse_args()

sys.path.insert(0, a.pacrunner)
from pacrunner import _env   # noqa: E402

MG7 = os.environ.get("PACRUNNER_MADGRAPH7", "")
MADSPACE_CUDA = os.path.join(MG7, "madspace", "install-cuda")


def have(need):
    if need is None:
        return True
    if need.startswith("simd-"):
        return need in _env.simd_devices()
    if need == "metal":
        return sys.platform == "darwin"
    cuda = sys.platform != "darwin" and shutil.which("nvcc") is not None and shutil.which("nvidia-smi") is not None
    return cuda and (need == "cuda" or os.path.isdir(MADSPACE_CUDA))


wanted = a.variants.split(",") if a.variants else SUITES[a.suite]
unknown = [v for v in wanted if v not in SUITES[a.suite]]
if unknown:
    sys.exit(f"variants {unknown} are not in the suite {a.suite!r}: {', '.join(SUITES[a.suite])}")
variants = [v for v in wanted if have(VARIANTS[v][2])]
skipped = [v for v in wanted if v not in variants]
build = {v: _env.library_name(*VARIANTS[v][:2]) for v in variants if v != "stock"}   # run --device <build>
print(f"{time.strftime('%Y-%m-%d %H:%M')} | {os.uname().nodename} ({platform.machine()}) | suite {a.suite} | "
      f"variants {', '.join(variants)}" + (f" | not on this machine: {', '.join(skipped)}" if skipped else ""), flush=True)
if a.list:
    sys.exit()

DRV = [a.runpy, os.path.join(a.pacrunner, "test", "mg7_run.py")]
env0 = {k: v for k, v in os.environ.items() if k != "PAC_UMAMI_TIMER"}


def run(cmd, cwd, log, env):
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env)
    return time.time() - t, r.returncode


def latest_run(out):
    runs = glob.glob(os.path.join(out, "Events", "run_*"))
    return max(runs, key=lambda p: int(p.rsplit("_", 1)[1])) if runs else ""


def output(name, v):
    return os.path.join(a.bench, name + ("_gpu" if VARIANTS[v][3] else ""))


def lc(out):
    return re.search(r'^color_accuracy = "lc"', open(os.path.join(out, "Cards", "run_card.toml")).read(), re.M)


def gpu_copy(name):
    """NAME_gpu: NAME without its runs and PacRunner builds, run.device = ["cuda"]"""
    src, dst = os.path.join(a.bench, name), os.path.join(a.bench, name + "_gpu")
    if os.path.isdir(dst):
        return dst
    shutil.copytree(src, dst, symlinks=True, ignore=lambda d, names: [
        n for n in names if (d == src and n == "Events") or (os.path.basename(d) == "lib" and n.startswith("pacrunner"))])
    card = os.path.join(dst, "Cards", "run_card.toml")
    text = re.sub(r"^device = .*$", 'device = ["cuda"]', open(card).read(), flags=re.M)
    open(card, "w").write(text)
    print(f"  made {dst}", flush=True)
    return dst


def one(name, v, key, timer, L):
    out = output(name, v)
    env = dict(env0)
    if timer:
        env["PAC_UMAMI_TIMER"] = os.path.join(L, f"{key}.timer")
        if os.path.exists(env["PAC_UMAMI_TIMER"]):
            os.remove(env["PAC_UMAMI_TIMER"])
    if v == "stock":
        cmd = DRV + ["run", ".", "--device", "stock", "-f"] if timer else [a.runpy, "bin/generate_events", "-f"]
    else:
        cmd = DRV + ["run", ".", "--device", build[v], "-f"]
    wall, rc = run(cmd, out, os.path.join(L, f"{key}.log"), env)
    if rc != 0:
        print(f"  {key}: FAILED ({rc}; {os.path.join(L, key + '.log')})", flush=True)
        return
    open(os.path.join(L, f"{key}.wall"), "w").write(f"{wall:.2f}\n")
    open(os.path.join(L, f"{key}.dir"), "w").write(latest_run(out) + "\n")
    print(f"  {key}: {wall:.2f} s", flush=True)


def points(name, v, L):
    """pass C: the cuda-full libraries on synchronized 64k-event batches (test/bench_libraries.py)"""
    for f in sorted(glob.glob(os.path.join(output(name, v), "lib", f"pacrunner-{build[v]}", "*", "umami_pac.json"))):
        env = dict(env0, PYTHONPATH=os.pathsep.join(p for p in (MADSPACE_CUDA, env0.get("PYTHONPATH")) if p))
        t, rc = run([a.runpy, os.path.join(a.pacrunner, "test", "bench_libraries.py"), os.path.dirname(f),
                     "--events", "64000", "--repeats", "5", "--no-stock"], output(name, v),
                    os.path.join(L, f"points-{v}.log"), env)
        last = open(os.path.join(L, f"points-{v}.log")).read().strip().splitlines()[-1:] or [""]
        print(f"  points-{v}: " + (last[0] if rc == 0 else f"FAILED ({rc}; {os.path.join(L, f'points-{v}.log')})"), flush=True)


for name in a.names.split(","):
    L = os.path.join(a.logs, name)
    os.makedirs(L, exist_ok=True)
    if not os.path.isdir(os.path.join(a.bench, name)):
        print(f"== {name}: no output in {a.bench}", flush=True)
        continue
    here = [v for v in variants if not VARIANTS[v][3] or lc(os.path.join(a.bench, name))]
    print(f"== {name}" + ("" if here == variants else f" (full colour: no {', '.join(v for v in variants if v not in here)})"),
          flush=True)
    if any(VARIANTS[v][3] for v in here):
        gpu_copy(name)
    if not a.skip_prepare:
        for v in here:
            if v == "stock":
                continue
            device, precision = VARIANTS[v][:2]
            t, rc = run(DRV + ["prepare", output(name, v), "--device", device, "--precision", precision, "--python", a.python],
                        output(name, v), os.path.join(L, f"prepare-{v}.log"), env0)
            print(f"  prepare {v}: {t:.1f} s" + ("" if rc == 0 else f" FAILED ({rc})"), flush=True)
    for r in range(1, a.repeats + 1):
        for v in here:
            one(name, v, f"wall-{v}-{r}", False, L)
    for v in here:
        if VARIANTS[v][3]:
            points(name, v, L)
        else:
            one(name, v, f"timer-{v}", True, L)
print("ALL DONE", flush=True)

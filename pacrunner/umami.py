"""UMAMI libraries for MadSpace, one per program (pac-julia's src/umami.jl): umami_libraries(export,
metadata, dir) writes, for each (process, colour flow, leg permutation) a pyAmpliCol artifact's UMAMI
provider uses (API/umami/metadata.json: its channels, each channel's flavours, each flavour one
contribution), a shared library made of the program's compiled C (codegen.compile: runner.c, pacops.c)
and c/umami_pac.c, a thin UMAMI layer: the permutation, the helicity tables, the factor, the colour
index, the masses and the exported model parameters are constants of the library.

The flavour splitting is MadSpace's: dir/umami_pac.json lists the libraries and, per channel, the
library of each flavour (`flavor_libraries`) and the channel's libraries (`libraries`). An integrand
per channel loads flavour f's library as a matrix element of its own, with one differential cross
section per flavour:
  flavor_diff_xs_indices[f] = f,
  flavor_remap[f] = 0 (each library sees flavour 0),
flavor_per_subproc_remap and flavor_subproc_indices as for the stock provider (the event output's
flavour and subprocess ids). MadSpace splits each batch by flavour itself; a library never sees
another flavour's events.

device "cuda": the program runs on a CUDA GPU (the UIR interpreter, c/cuda/uirgpu.c, built by nvcc); the
library still identifies as a CPU provider (host pointers, MadSpace's CPU threads).
device "cuda_provider" (alias "cuda_full"): the library identifies as CUDA, for MadSpace's GPU runtime
(its device pointers and stream; asynchronous, umami_pac.c).
device "metal" (macOS): the program runs on an Apple GPU (the UIR interpreter in MSL, c/metal/, mixed
f32/df64: Metal has no f64), the library a CPU provider (c/metal/umami_metal.c: the inputs and each
event's end on the host in f64, one GPU run per call); one slot pool per process, libpacmetal.dylib.
device "gpu": this system's GPU with MadSpace on the CPU, metal on macOS and cuda elsewhere.
device "simd" (x86-64): the program as one symjit kernel (codegen.compile_simd: the library of pacops
operations spliced; symjit's AVX2 kernel, and its AVX-512 one where the building CPU has AVX-512, as object
files), 4 or 8 events per kernel call; the library a CPU provider with the CPU build's bits (umami_pac.c,
PAC_SIMD).

The programs come from RustiCol's recurrence export (uir.load_recurrence_export) of the same artifact,
with the same model parameters: a library refuses other values (umami_set_parameter).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from . import _env, codegen, kernels, uir
from ._literal import julia_repr
from .codegen.common import C_DIR, cstring
from .codegen.compile_simd import host_lanes
from .codegen.compile_vm import GPU_MEMORY_DEFAULT
from .uir import Lowered

DEVICES = _env.DEVICES                        # cuda: on the GPU, a CPU provider; cuda_provider: a CUDA one;
                                              # metal: an Apple GPU, a CPU provider, always mixed
DEVICE_ALIASES = _env.DEVICE_ALIASES          # cuda_full: all on the GPU (MadSpace's GPU runtime)
PRECISIONS = _env.PRECISIONS                  # a CUDA device: mixed is pacops.h's PAC_MIXED
GPU_DEVICES = ("cuda", "cuda_provider", "metal")
SO = "dylib" if sys.platform == "darwin" else "so"

Flags = Union[str, Sequence[str]]


class UmamiError(RuntimeError):
    """a provider umami_libraries cannot build"""


def device_name(device: str) -> str:
    """a device's own name (an alias's target; gpu: metal on macOS, cuda elsewhere)"""
    return _env.device_name(device)


@dataclass(frozen=True)
class UmamiLibrary:
    process: str                  # the provider's runtime process (the export's process_key)
    colour: str                   # its colour flow (colour_id)
    permutation: tuple            # runtime leg j = input leg permutation[j]
    colour_index: int
    factor: float
    helicity_export: tuple        # runtime helicity → exported helicity


def clit(x: float) -> str:
    if x != x or x in (float("inf"), float("-inf")):
        raise UmamiError(f"not finite: {x}")
    return julia_repr(float(x))


def permuted(l: Lowered, permutation: Sequence[int]) -> Lowered:
    """`l` reading the caller's legs (MadSpace's order): the library's leg permutation (runtime leg j =
    input leg permutation[j]) folded into the crossing, so program leg l is sign × the caller's leg
    permutation[crossing leg] (× 1.0 without a crossing: the same bits). No gather in the library: the
    form fill reads the caller's momenta as they are."""
    if len(permutation) != l.legs:
        raise UmamiError(f"{l.process} {l.colour}: a permutation of {len(permutation)} legs for {l.legs}")
    if sorted(permutation) != list(range(l.legs)):
        raise UmamiError(f"{l.process} {l.colour}: {list(permutation)} is not a permutation")
    if l.crossing is None:
        crossing = tuple((permutation[k], 1.0) for k in range(l.legs))
    else:
        crossing = tuple((permutation[leg], sign) for (leg, sign) in l.crossing)
    return dataclasses.replace(l, crossing=crossing)


def _jlist(v) -> str:
    """a vector as Julia prints it: [0, 1, 2]"""
    return "[" + ", ".join(str(x) for x in v) + "]"


def umami_program_h(lib: UmamiLibrary, l: Lowered, provider: dict, exported_count: int, parameters) -> tuple[str, str]:
    """umami_tables.h of `lib` (whose program is `l`): what umami_finish.h reads, on the CPU and the GPU
    (its tables PAC_TABLE: __device__ under nvcc), and umami_program.h: the host's constants"""
    what = f"{lib.process} {lib.colour}"
    if exported_count <= 0:
        raise UmamiError(f"{what}: the provider exports no helicities")
    if not lib.helicity_export:
        raise UmamiError(f"{what}: no runtime helicities")
    runtime = [-1] * exported_count
    for r, e in enumerate(lib.helicity_export):
        if not 0 <= e < exported_count:
            raise UmamiError(f"{what}: runtime helicity {r} is exported helicity {e}, not in 0:{exported_count - 1}")
        if runtime[e] != -1:
            raise UmamiError(f"{what}: exported helicity {e} is runtime helicity {runtime[e]} and {r}")
        runtime[e] = r
    (alpha_s0, power) = (0.0, None) if l.alpha_s is None else l.alpha_s
    scales = power is not None and alpha_s0 > 0   # values at another alpha_s: × (alpha_s / alpha_s0)^power
    masses = [float(m) for m in provider["masses"]]
    tables = "".join(s + "\n" for s in [
        f"/* {lib.process} {lib.colour}, permutation {_jlist(lib.permutation)} (in the program's crossing): "
        "written by PacRunner's umami_libraries, included by umami_finish.h */",
        f"#define PAC_PARTICLES {int(provider['particle_count'])}",
        f"#define PAC_HELICITY_COUNT {int(provider['helicity_count'])}",
        f"#define PAC_EXPORTED_HELICITY_COUNT {exported_count}",
        f"#define PAC_COLOR_COUNT {int(provider['color_count'])}",
        f"#define PAC_COLOR_INDEX {lib.colour_index}",
        f"#define PAC_FACTOR {clit(lib.factor)}",
        f"#define PAC_ALPHA_S_SCALES {int(scales)}",
        f"#define PAC_ALPHA_S0 {clit(alpha_s0)}",
        f"#define PAC_ALPHA_S_POWER {power if scales else 0}",
        "PAC_TABLE int pac_helicity_runtime[] = {" + ", ".join(str(x) for x in runtime) + "};",
        "PAC_TABLE int pac_helicity_export[] = {" + ", ".join(str(x) for x in lib.helicity_export) + "};",
    ])
    program = [f"/* {lib.process} {lib.colour}: written by PacRunner's umami_libraries (umami_tables.h: the rest) */",
               "static const double pac_masses[] = {" + ", ".join(clit(m) for m in masses) + "};",
               "static const PacParameter pac_parameters[] = {"]
    program += [f"    {{{cstring(name)}, {clit(z.real)}, {clit(z.imag)}}}," for (name, z) in parameters]
    program += ["    {NULL, 0.0, 0.0},", "};"]
    return (tables, "".join(s + "\n" for s in program))


# ---- the tools ------------------------------------------------------------------------------------------------

def _flags(f: Flags) -> list[str]:
    return shlex.split(f) if isinstance(f, str) else list(f)


def find_cc() -> Optional[str]:
    """CC (a command, possibly with arguments: "ccache gcc") when its program is found, else None"""
    c = os.environ.get("CC") or "cc"
    words = shlex.split(c)
    return c if words and shutil.which(words[0]) else None


def _replace_copy(src, dst) -> None:
    """copy `src` to `dst` as a new file (written beside it, then renamed): a process that has the old
    `dst` loaded (a shared library) keeps its own copy instead of reading a truncated one"""
    dst = Path(dst)
    with tempfile.NamedTemporaryFile(dir=dst.parent, prefix=f".{dst.name}.", delete=False) as f:
        tmp = f.name
    try:
        shutil.copyfile(src, tmp)
        os.chmod(tmp, 0o755)
        os.replace(tmp, dst)
    except BaseException:
        os.unlink(tmp)
        raise


def find_nvcc() -> Optional[str]:
    c = os.environ.get("NVCC") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    return c if shutil.which(c) or os.path.isfile(c) else None


def _run(args, cwd=None) -> None:
    r = subprocess.run([os.fspath(a) for a in args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise UmamiError(f"{shlex.join(os.fspath(a) for a in args)} failed ({r.returncode}):\n{r.stdout}{r.stderr}")


def nvcc_toolkit(nvcc: str) -> Optional[str]:
    """the CUDA toolkit's folder as nvcc itself reports it (its `TOP`): right also when `nvcc` is a
    wrapper script (Ubuntu's /usr/bin/nvcc execs a toolkit's nvcc); None if nvcc cannot say"""
    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "t.cu"
            src.write_text("")
            r = subprocess.run([nvcc, "--dryrun", "-c", str(src), "-o", str(Path(d) / "t.o")], capture_output=True, text=True)
            if r.returncode != 0:
                return None
            out = r.stdout + r.stderr   # nvcc --dryrun prints its steps on stderr
    except OSError:
        return None
    m = re.search(r"^#\$ TOP=(.*)$", out, re.M)
    return os.path.normpath(m.group(1).strip()) if m else None


def find_cudart(nvcc: str) -> str:
    """the CUDA runtime library's file of nvcc's toolkit (libcudart.so.N, its soname): the toolkit nvcc
    reports, else the folders next to nvcc"""
    bin_ = os.path.dirname(shutil.which(nvcc) or nvcc)
    tops = list(dict.fromkeys(t for t in (nvcc_toolkit(nvcc), os.path.join(bin_, "..")) if t))
    tried = []
    for top in tops:
        for lib in (os.path.join(top, "lib64"), os.path.join(top, "targets", "x86_64-linux", "lib")):
            tried.append(os.path.normpath(lib))
            if not os.path.isdir(lib):
                continue
            names = sorted(f for f in os.listdir(lib) if re.fullmatch(r"libcudart\.so\.\d+", f))
            if names:
                return os.path.realpath(os.path.join(lib, names[0]))   # the file it links to
    raise UmamiError(f"no libcudart.so.N in the CUDA toolkit of {nvcc} (looked in {', '.join(tried)}); "
                     "set NVCC to the toolkit's nvcc, e.g. NVCC=/usr/local/cuda/bin/nvcc")


def build_cuda_runtime(dir, nvcc: Optional[str] = None, cuda_arch: str = "native", pool: bool = True) -> str:
    """the process-wide GPU pool (pool.cu) in `dir`: libpacpool.so, with a copy of the CUDA runtime it
    and the program libraries share (libcudart.so.N, found through rpath $ORIGIN). pool = False (the
    CUDA provider, which runs on the caller's stream): the runtime copy alone."""
    nvcc = nvcc or find_nvcc() or "nvcc"
    dir = Path(dir)
    so = dir / f"libpacpool.{SO}"
    cudart = find_cudart(nvcc)
    m = re.match(r"(libcudart\.so\.\d+)", os.path.basename(cudart))
    copy = dir / (m.group(1) if m else os.path.basename(cudart))
    _replace_copy(cudart, copy)
    if not pool:
        return str(copy)
    shutil.copyfile(C_DIR / "cuda" / "pool.h", dir / "pool.h")
    _run([nvcc, "-O3", f"-arch={cuda_arch}", "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared", "-Xlinker",
          f"-soname=libpacpool.{SO}", "-Xlinker", "-rpath", "-Xlinker", "$ORIGIN", f"-I{dir}", "-o", so, C_DIR / "cuda" / "pool.cu"])
    return str(so)


METAL_POOL = ("metalpool.h", "metalpool.m")
METAL_FRAMEWORKS = ["-framework", "Metal", "-framework", "Foundation"]


def _require_macos(device: str = "metal") -> None:
    if sys.platform != "darwin":
        raise UmamiError(f"device = {device} builds on macOS only (Metal)")


def build_metal_runtime(dir, cc: Optional[str] = None) -> str:
    """the process-wide Metal slot pool (c/metal/metalpool.m) in `dir`: libpacmetal.dylib (macOS), which
    every program library links (rpath @loader_path)"""
    _require_macos()
    dir = Path(dir)
    for f in METAL_POOL:
        shutil.copyfile(C_DIR / "metal" / f, dir / f)
    so = dir / "libpacmetal.dylib"
    _run([*_flags(cc or find_cc() or "cc"), "-O2", "-fPIC", "-fobjc-arc", "-dynamiclib", "-install_name",
          "@rpath/libpacmetal.dylib", "-o", so, dir / "metalpool.m", *METAL_FRAMEWORKS])
    return str(so)


def build_umami_metal(dir, lib: UmamiLibrary, l: Lowered, provider: dict, exported_count: int, parameters, *,
                      cc: Optional[str] = None, cflags: Flags = ("-O2",), runtime_dir=None) -> str:
    """build_umami_library for device "metal" (macOS): compile_metal's folder (the permutation folded into
    the crossing), the UMAMI layer c/metal/umami_metal.c, linked against libpacmetal.dylib in
    `runtime_dir` (built there if missing); every symbol but the UMAMI functions hidden"""
    _require_macos()
    if int(provider["particle_count"]) != l.legs:
        raise UmamiError(f"{lib.process} {lib.colour}: the provider has {provider['particle_count']} particles, "
                         f"the program {l.legs} legs")
    dir = Path(dir)
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else dir
    l = permuted(l, lib.permutation)   # the library reads the caller's legs
    codegen.compile_metal(dir, l)
    (tables, program) = umami_program_h(lib, l, provider, exported_count, parameters)
    (dir / "umami_tables.h").write_text(tables, encoding="utf-8")
    (dir / "umami_program.h").write_text(program, encoding="utf-8")
    shutil.copyfile(C_DIR / "metal" / "umami_metal.c", dir / "umami_metal.c")
    for f in ("umami_finish.h", "umami.h"):
        shutil.copyfile(C_DIR / f, dir / f)
    if not (runtime_dir / "libpacmetal.dylib").is_file():
        build_metal_runtime(runtime_dir, cc)
    so = dir / "libumami_pac.dylib"
    rpath = "@loader_path/" + os.path.relpath(runtime_dir.resolve(), dir.resolve())
    _run([*_flags(cc or find_cc() or "cc"), *_flags(cflags), "-ffp-contract=off", "-fPIC", "-fvisibility=hidden", "-fobjc-arc",
          "-dynamiclib", "-o", so, dir / "umami_metal.c", dir / "uirmetal.m", dir / "runner_metal.c", f"-I{runtime_dir}",
          f"-L{runtime_dir}", "-lpacmetal", f"-Wl,-rpath,{rpath}", *METAL_FRAMEWORKS, "-lm"])
    return str(so)


def build_umami_simd(dir, lib: UmamiLibrary, l: Lowered, provider: dict, exported_count: int, parameters, *,
                     cc: Optional[str] = None, cflags: Flags = ("-O2",)) -> str:
    """build_umami_library for device "simd" (x86-64): compile_simd's folder (the permutation folded into the
    crossing; kernel4.o, the AVX2 kernel, and kernel8.o, the AVX-512 one, where this CPU has AVX-512) and
    umami_pac.c with PAC_SIMD, linked with only the UMAMI functions exported"""
    if not _env.simd_machine():
        raise UmamiError("device simd: x86-64 only (AVX2 and AVX-512 kernels)")
    if int(provider["particle_count"]) != l.legs:
        raise UmamiError(f"{lib.process} {lib.colour}: the provider has {provider['particle_count']} particles, "
                         f"the program {l.legs} legs")
    cc = _flags(cc or find_cc() or "cc")
    dir = Path(dir)
    l = permuted(l, lib.permutation)   # the library reads the caller's legs
    try:
        codegen.compile_simd(dir, l)
    except codegen.CompileError as e:
        raise UmamiError(f"{lib.process} {lib.colour}: {e}") from e
    (tables, program) = umami_program_h(lib, l, provider, exported_count, parameters)
    (dir / "umami_tables.h").write_text(tables, encoding="utf-8")
    (dir / "umami_program.h").write_text(program, encoding="utf-8")
    for f in ("umami_pac.c", "umami_finish.h", "umami.h"):
        shutil.copyfile(C_DIR / f, dir / f)
    exports = dir / "umami.map"
    exports.write_text("{ global: umami_*; local: *; };\n")
    hide = [f"-Wl,--version-script={exports}"] if sys.platform.startswith("linux") else []
    so = dir / f"libumami_pac.{SO}"
    _run([*cc, *_flags(cflags), "-std=c99", "-ffp-contract=off", "-fPIC", "-fvisibility=hidden", "-DPAC_SIMD", "-shared",
          *hide, "-o", so, dir / "umami_pac.c", *sorted(dir.glob("kernel[48].o")), "-lm"])
    return str(so)


def build_umami_library(dir, lib: UmamiLibrary, l: Lowered, provider: dict, exported_count: int, parameters, *,
                        cc: Optional[str] = None, cflags: Flags = ("-O2",), interpreter: bool = False, device: str = "cpu",
                        nvcc: Optional[str] = None, cuda_arch: str = "native", nvflags: Flags = ("-O3",),
                        gpu_memory: int = GPU_MEMORY_DEFAULT, gpu_slots: int = 0, precision: str = "f64",
                        fmad: Optional[bool] = None, runtime_dir=None) -> str:
    """The shared library of `lib` in `dir` (codegen.compile's folder plus the UMAMI layer). device
    "cuda": the interpreter's runner.c built by nvcc for `cuda_arch` (--fmad=false: the CPU's bits), on
    the process-wide GPU slot pool of libpacpool.so in `runtime_dir` (built there if missing; one CUDA
    runtime, libcudart.so.N, shared through rpath); every symbol but the UMAMI functions hidden. The
    pool the library asks for (the first library of a process decides): `gpu_slots` slots (0: one per
    CPU) of `gpu_memory` device bytes each (a call needing more runs as several launches). precision
    "mixed" (a CUDA device only): pacops.h's PAC_MIXED, not the CPU's bits, so nvcc may contract
    multiply-adds unless fmad = False. device "metal" (macOS): build_umami_metal (always mixed f32/df64).
    device "simd" (x86-64): build_umami_simd."""
    device = device_name(device)
    if device not in DEVICES:
        raise UmamiError(f"device {device}: one of {DEVICES} (or {', '.join(DEVICE_ALIASES)}, gpu)")
    if precision not in PRECISIONS:
        raise UmamiError(f"precision {precision}: one of {PRECISIONS}")
    if device == "metal":
        return build_umami_metal(dir, lib, l, provider, exported_count, parameters, cc=cc, cflags=cflags,
                                 runtime_dir=runtime_dir)
    if device == "simd":
        if precision != "f64":
            raise UmamiError(f"precision = {precision} needs device = cuda or cuda_provider")
        return build_umami_simd(dir, lib, l, provider, exported_count, parameters, cc=cc, cflags=cflags)
    fmad = precision == "mixed" if fmad is None else fmad
    cuda = device in ("cuda", "cuda_provider")
    provider_build = device == "cuda_provider"
    if precision != "f64" and not cuda:
        raise UmamiError(f"precision = {precision} needs device = cuda or cuda_provider")
    if fmad and precision == "f64":
        raise UmamiError("fmad = True is not the CPU's bits: precision = mixed only")
    cc = _flags(cc or find_cc() or "cc")   # a command, possibly with arguments
    nvcc = nvcc or find_nvcc()
    if cuda and nvcc is None:
        raise UmamiError("a CUDA device needs nvcc (NVCC, the PATH or /usr/local/cuda/bin)")
    if int(provider["particle_count"]) != l.legs:
        raise UmamiError(f"{lib.process} {lib.colour}: the provider has {provider['particle_count']} particles, "
                         f"the program {l.legs} legs")
    dir = Path(dir)
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else dir
    l = permuted(l, lib.permutation)   # the library reads the caller's legs
    codegen.compile(dir, l, interpreter=interpreter or cuda)
    (tables, program) = umami_program_h(lib, l, provider, exported_count, parameters)
    (dir / "umami_tables.h").write_text(tables, encoding="utf-8")
    (dir / "umami_program.h").write_text(program, encoding="utf-8")
    for f in ("umami_pac.c", "umami_finish.h", "umami.h"):
        shutil.copyfile(C_DIR / f, dir / f)
    so = dir / f"libumami_pac.{SO}"
    cflags = _flags(cflags)
    if cuda:
        if provider_build:
            if not any(f.startswith("libcudart.so") for f in os.listdir(runtime_dir)):
                build_cuda_runtime(runtime_dir, nvcc, cuda_arch, pool=False)
        elif not (runtime_dir / f"libpacpool.{SO}").is_file():
            build_cuda_runtime(runtime_dir, nvcc, cuda_arch)
        rpath = "$ORIGIN/" + os.path.relpath(runtime_dir.resolve(), dir.resolve())
        mixed = ["-DPAC_MIXED"] if precision == "mixed" else []
        # the CPU provider runs on the slot pool (libpacpool.so); the CUDA provider on the caller's
        # stream (gpu_memory: the workspace budget of a launch), with the same CUDA runtime copy
        gpu = ["-DUIRGPU_PROVIDER"] if provider_build else ["-DPACPOOL_EXTERNAL", "-DUIRGPU_UMAMI", f"-DUIRGPU_SLOTS={gpu_slots}"]
        _run([nvcc, *_flags(nvflags), f"-arch={cuda_arch}", f"--fmad={'true' if fmad else 'false'}", *mixed, "-Xcompiler",
              "-fPIC,-fvisibility=hidden", *gpu, f"-DUIRGPU_SLOT_BYTES={int(gpu_memory)}", "-x", "cu", "-c", "-o",
              dir / "runner_gpu.o", dir / "runner.c"])
        _run([*cc, *cflags, "-std=c99", "-ffp-contract=off", "-fPIC", "-fvisibility=hidden",
              "-DPAC_CUDA_PROVIDER" if provider_build else "-DPAC_CUDA", "-c", "-o", dir / "umami_pac.o", dir / "umami_pac.c"])
        pool = [] if provider_build else [f"-L{runtime_dir}", "-lpacpool"]
        _run([nvcc, "-shared", f"-arch={cuda_arch}", "-cudart", "shared", "-o", so, dir / "umami_pac.o", dir / "runner_gpu.o",
              *pool, "-Xlinker", "-rpath", "-Xlinker", rpath, "-lm"])
    else:
        sources = [dir / f for f in ("umami_pac.c", "runner.c", "pacops.c")]
        _run([*cc, *cflags, "-std=c99", "-ffp-contract=off", "-fPIC", "-fvisibility=hidden", "-shared", "-o", so, *sources, "-lm"])
    return str(so)


# ---- the provider ---------------------------------------------------------------------------------------------

def export_programs(d: dict) -> dict:
    """the export's programs by (process_key, colour_id), each with its helicity ids"""
    ls = uir.load_recurrence_export(d)
    return {(l.process, l.colour): (l, [str(h["id"]) for h in p["helicity_outputs"]]) for l, p in zip(ls, d["programs"])}


def flavour_library(r: dict, procs, programs: dict, export_path, place: str) -> UmamiLibrary:
    """the library of one flavour of a channel (`r`: its runtime entry in the metadata), checked
    against its program: the helicities are the provider's in order, one exported helicity per output"""
    proc = procs[int(r["process_index"])]
    lib = UmamiLibrary(str(proc["id"]), str(r["color_id"]), tuple(int(x) for x in r["permutation"]), int(r["color_index"]),
                       float(r["factor"]), tuple(int(x) for x in r["helicity_export"]))
    if (lib.process, lib.colour) not in programs:
        raise UmamiError(f"{place}: no program {lib.process} {lib.colour} in {export_path}")
    (l, ids) = programs[(lib.process, lib.colour)]
    if ids != [str(h) for h in proc["helicity_ids"]]:
        raise UmamiError(f"{lib.process} {lib.colour}: the export's helicities are not the provider's, in order")
    if len(lib.helicity_export) != l.output_count():
        raise UmamiError(f"{lib.process} {lib.colour}: {len(lib.helicity_export)} exported helicities for "
                         f"{l.output_count()} outputs")
    return lib


def provider_libraries(meta: dict, programs: dict, export_path) -> tuple[list, list]:
    """the provider's distinct libraries, and per channel each flavour's library (an index into them)"""
    procs = meta["runtime_processes"]
    libraries: list[UmamiLibrary] = []
    index: dict = {}
    channels = []
    for c, channel in enumerate(meta["channels"]):
        fl = []
        for f, flavour in enumerate(channel["processes"]):
            lib = flavour_library(flavour["runtime"], procs, programs, export_path, f"channel {c} flavour {f}")
            if lib not in index:
                index[lib] = len(libraries)
                libraries.append(lib)
            fl.append(index[lib])
        channels.append(fl)
    return (libraries, channels)


def library_entry(k: int, lib: UmamiLibrary, path: str) -> dict:
    """a library's entry of umami_pac.json (its path relative to the table)"""
    return {"index": k, "path": path, "process": lib.process, "colour": lib.colour, "permutation": list(lib.permutation),
            "colour_index": lib.colour_index, "factor": lib.factor}


def read_metadata(metadata_path, d: dict) -> dict:
    """the artifact's UMAMI metadata, checked against the export"""
    with open(metadata_path) as f:
        meta = json.load(f)
    if meta.get("kind") != "pyamplicol-umami-metadata":
        raise UmamiError(f"{metadata_path}: not a pyAmpliCol UMAMI metadata file")
    provider = meta["provider"]
    if provider["artifact_id"] != d["artifact_id"]:
        raise UmamiError(f"the export is of artifact {d['artifact_id']}, the metadata of {provider['artifact_id']}")
    if provider["color_accuracy"] != "lc":
        raise UmamiError(f"the provider is {provider['color_accuracy']}, not lc")
    return meta


def export_parameters(d: dict) -> list:
    return [(str(p["name"]), complex(float(p["value"]["re"]), float(p["value"]["im"]))) for p in d["parameters"]]


def umami_libraries(export, metadata, dir, *, optimize_programs: bool = True, interpreter: bool = False,
                    cc: Optional[str] = None, cflags: Flags = ("-O2",), device: str = "cpu", cuda_arch: str = "native",
                    nvcc: Optional[str] = None, gpu_slots: int = 0, gpu_memory: int = GPU_MEMORY_DEFAULT,
                    precision: str = "f64", fmad: Optional[bool] = None, nvflags: Flags = ("-O3",), quiet: bool = False,
                    jobs: Optional[int] = None) -> str:
    """One UMAMI library per (process, colour flow, permutation) of the artifact's UMAMI provider
    (`metadata`: the artifact's API/umami/metadata.json), its program from the recurrence export
    `export` of the same artifact; written into `dir`, with `dir/umami_pac.json` (the libraries and
    each channel's flavours, for MadSpace). Returns the path of umami_pac.json.

    `optimize_programs`: the programs after uir.optimize (the default; the same values) or as imported.
    `interpreter`: each program as the UIR interpreter's data instead of straight-line C; `cc` and
    `cflags` build the libraries. `device = "cuda"`: each program runs on a CUDA GPU (the interpreter,
    built by `nvcc` for `cuda_arch`, default "native": this machine's GPU), the libraries still CPU
    providers for MadSpace. The GPU memory and streams are a process-wide pool (`dir/libpacpool.so`,
    with a copy of the CUDA runtime): `gpu_slots` slots (default 0: one per CPU; 1: one call at a time)
    of `gpu_memory` device bytes each, allocated when first used (a call needing more runs as several
    launches). `precision` (a CUDA device): "f64", RustiCol's values bit for bit, or "mixed": the
    currents, their arithmetic and the vertex sums in f32, the momentum forms, propagator denominators,
    wavefunctions and amplitudes in f64, the kernels in f32 but the backward slices of their
    denominators (about 1e-7 relative on the matrix element); `nvflags` adds nvcc flags (-DPAC_ACC64:
    the vertex sums in f64; -DPAC_AMP32: the amplitudes in f32); `fmad = False`: no contracted
    multiply-adds. `device = "cuda_provider"` (alias "cuda_full"): the same GPU build, but each library
    identifies as a CUDA provider for MadSpace's GPU runtime: device pointers, the caller's stream,
    nothing synchronized; `gpu_memory` is then the workspace budget of one launch. `device = "metal"`
    (macOS): each program on an Apple GPU through Metal (c/metal/), always in mixed f32/df64 precision
    (`precision` is ignored), the libraries CPU providers; one slot pool per process
    (`dir/libpacmetal.dylib`; PAC_METAL_SLOTS, PAC_METAL_CHUNK, PAC_METAL_SLOT_BYTES). `device = "gpu"`:
    metal on macOS, cuda elsewhere. `device = "simd"` (x86-64): each program as one symjit kernel, symjit's
    AVX2 kernel (and its AVX-512 one where the building CPU has AVX-512) as object files, the CPU build's
    bits; the library picks the kernel of the CPU it runs on (PAC_SIMD_LANES=4 or 8 asks for one). The libraries are built in parallel (`jobs` at a time; default: the
    CPUs)."""
    device = device_name(device)
    if device not in DEVICES:
        raise UmamiError(f"device {device}: one of {DEVICES} (or {', '.join(DEVICE_ALIASES)}, gpu)")
    if precision not in PRECISIONS:
        raise UmamiError(f"precision {precision}: one of {PRECISIONS}")
    if device == "metal":
        _require_macos()
        precision = "mixed"   # Metal has no f64: always mixed f32/df64
    if precision != "f64" and device not in GPU_DEVICES:
        raise UmamiError(f"precision = {precision} needs device = cuda, cuda_provider or metal")
    fmad = (precision == "mixed" and device != "metal") if fmad is None else fmad
    if fmad and precision == "f64":
        raise UmamiError("fmad = True is not the CPU's bits: precision = mixed only")
    kernels.require_symjit()   # symjit >= 2.29 (the kernels' MIR), before any work
    if device == "simd":
        if not _env.simd_machine():
            raise UmamiError("device simd: x86-64 only (AVX2 and AVX-512 kernels)")
        if 4 not in host_lanes():
            raise UmamiError("device simd needs an x86-64 CPU with AVX2 and FMA to build on (symjit compiles the kernels)")
    export_path = os.path.abspath(export) if isinstance(export, (str, os.PathLike)) else None   # or an export already read
    d = uir.read_export(export)
    meta = read_metadata(metadata, d)
    provider = meta["provider"]
    exported_count = len(meta["helicities"])
    parameters = export_parameters(d)
    programs = export_programs(d)
    (libraries, channels) = provider_libraries(meta, programs, export_path or "the export")

    dir = Path(dir).resolve()
    dir.mkdir(parents=True, exist_ok=True)
    nvcc = nvcc or find_nvcc()
    if device in ("cuda", "cuda_provider"):
        if nvcc is None:
            raise UmamiError("a CUDA device needs nvcc (NVCC, the PATH or /usr/local/cuda/bin)")
        build_cuda_runtime(dir, nvcc, cuda_arch, pool=device == "cuda")
    if device == "metal":
        build_metal_runtime(dir, cc)

    def build(k: int) -> str:
        lib = libraries[k]
        sub = dir / "programs" / f"{k:04d}"
        sub.mkdir(parents=True, exist_ok=True)
        l = programs[(lib.process, lib.colour)][0]
        so = build_umami_library(sub, lib, uir.optimize(l) if optimize_programs else l, provider, exported_count, parameters,
                                 cc=cc, cflags=cflags, interpreter=interpreter, device=device, cuda_arch=cuda_arch, nvcc=nvcc,
                                 nvflags=nvflags, gpu_memory=gpu_memory, gpu_slots=gpu_slots, precision=precision, fmad=fmad,
                                 runtime_dir=dir)
        if not quiet:
            print(f"library {k} of {len(libraries)}: {lib.process} {lib.colour} {list(lib.permutation)}", flush=True)
        return so

    # the first failure is raised at once: the builds not started yet are cancelled
    with ThreadPoolExecutor(jobs or os.cpu_count() or 1) as pool:
        futures = [pool.submit(build, k) for k in range(len(libraries))]
        try:
            sos = [f.result() for f in futures]
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    table = {
        "kind": "pacrunner-umami-libraries", "schema_version": 1, "device": device, "precision": precision,
        "gpu_slots": gpu_slots if device == "cuda" else None, "gpu_memory": None if device == "cpu" else gpu_memory,
        "artifact_id": d["artifact_id"], "export": export_path, "metadata": os.path.abspath(metadata),
        "particle_count": provider["particle_count"],
        "libraries": [library_entry(k, lib, os.path.relpath(so, dir)) for k, (lib, so) in enumerate(zip(libraries, sos))],
        "channels": [{"flavor_libraries": fl, "libraries": sorted(set(fl))} for fl in channels],
    }
    path = dir / "umami_pac.json"
    path.write_text(json.dumps(table, indent=2))
    return str(path)


def main(argv=None) -> int:
    """python -m pacrunner.umami EXPORT METADATA DIR [options]: umami_libraries from the command line (a
    python with pacrunner and symjit >= 2.29, for scripts that run in another one, such as MadGraph7's);
    prints umami_pac.json's path (an error: its message, exit status 1). --help lists the options"""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m pacrunner.umami", description="PacRunner's UMAMI libraries of a provider")
    ap.add_argument("export")
    ap.add_argument("metadata")
    ap.add_argument("dir")
    ap.add_argument("--device", default="cpu", choices=DEVICES + tuple(DEVICE_ALIASES) + ("gpu",))
    ap.add_argument("--precision", default="f64", choices=PRECISIONS)
    ap.add_argument("--interpreter", action="store_true", help="the UIR interpreter's data instead of straight-line C")
    ap.add_argument("--imported", action="store_true", help="the programs as imported (not optimized)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--jobs", type=int, help="libraries built at a time (default: the CPUs)")
    ap.add_argument("--cc", help="the C compiler, possibly with arguments (default: CC, else cc)")
    ap.add_argument("--cflags", default="-O2", help="the C compiler's flags (default: -O2)")
    ap.add_argument("--nvcc", help="nvcc (default: NVCC, the PATH, else /usr/local/cuda/bin/nvcc)")
    ap.add_argument("--cuda-arch", default="native", help="nvcc's -arch (default: native, this machine's GPU)")
    ap.add_argument("--nvflags", default="-O3", help="nvcc's flags (default: -O3)")
    ap.add_argument("--gpu-slots", type=int, default=0, help="device cuda: the pool's slots (default 0: one per CPU)")
    ap.add_argument("--gpu-memory", type=int, default=GPU_MEMORY_DEFAULT, help="the device bytes of a slot (or a launch)")
    ap.add_argument("--fmad", action=argparse.BooleanOptionalAction, default=None,
                    help="nvcc's contracted multiply-adds (default: with precision mixed only)")
    a = ap.parse_args(argv)
    try:
        path = umami_libraries(a.export, a.metadata, a.dir, device=a.device, precision=a.precision, interpreter=a.interpreter,
                               optimize_programs=not a.imported, quiet=a.quiet, jobs=a.jobs, cc=a.cc, cflags=a.cflags,
                               nvcc=a.nvcc, cuda_arch=a.cuda_arch, nvflags=a.nvflags, gpu_slots=a.gpu_slots,
                               gpu_memory=a.gpu_memory, fmad=a.fmad)
    except (UmamiError, uir.ExportError, uir.InvalidLowered, uir.EncodeError, codegen.CompileError, kernels.KernelCError,
            kernels.MirError, OSError) as e:
        print(f"python -m pacrunner.umami: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

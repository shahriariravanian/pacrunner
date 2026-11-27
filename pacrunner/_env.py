"""Where pacrunner's tools are: its C sources (pacrunner/c/), a MadGraph7 checkout with its
MadSpace (CPU or CUDA build) and pyAmpliCol. Set with pacrunner.configure(...), else from the
environment (PACRUNNER_MADGRAPH7, PACRUNNER_PYAMPLICOL), else found."""

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


class PacRunnerError(RuntimeError):
    """A pacrunner setup or use error, with what to do about it"""


# the package, and the C sources the libraries are built from (package data: pacrunner/c/)
PACKAGE = Path(__file__).resolve().parent
C_DIR = PACKAGE / "c"
# the clone pacrunner is installed from, editable (its parent folder); not one for an installed copy
REPOSITORY = PACKAGE.parent


# the SIMD devices: one symjit kernel per program, for one instruction set (its f64 lanes)
SIMD_DEVICES = {"simd-avx2": 4, "simd-avx512": 8, "simd-neon": 2}
# cpu-jit: the CPU runner made by symjit: the SIMD devices' kernel of a program (simd-avx2's on x86-64, simd-neon's on
# aarch64) as symjit's scalar kernel, one event a call; f64, the cpu runner's bits; builds where a SIMD device does
CPU_JIT = "cpu-jit"
DEVICES = ("cpu", CPU_JIT, *SIMD_DEVICES, "cuda", "cuda_provider", "metal")
# cuda_full: everything on the GPU (MadSpace's GPU runtime); simd: this machine's SIMD device (simd_device)
DEVICE_ALIASES = {"cuda_full": "cuda_provider", "simd": None}
# f64; mixed: a GPU's f32 currents (Metal: f32/df64); split: a SIMD device's split kernels (f64 + f32)
PRECISIONS = ("f64", "mixed", "split")


def device_name(device: str) -> str:
    """a compile device's own name: an alias's target (cuda_full is cuda_provider), "gpu" this
    system's GPU with MadSpace on the CPU (metal on macOS, cuda elsewhere), "simd" this machine's SIMD
    device (simd_device)"""
    if device == "gpu":
        return "metal" if sys.platform == "darwin" else "cuda"
    if device == "simd":
        return simd_device() or "simd"
    return DEVICE_ALIASES.get(device) or device


def compile_target(device: str, precision: str = "f64") -> tuple[str, str]:
    """(device, precision) of a build: the device's own name (device_name), and for metal always
    "mixed" (Metal has no f64: mixed f32/df64). A SIMD device's name may carry its precision
    ("simd-avx2-split", "simd-split"); "mixed" on a SIMD device is "split" (its former name). An
    unknown device or precision, metal off macOS, or a SIMD device (or cpu-jit, f64 only) this machine
    cannot build (symjit's JIT compiles the kernels: simd_devices), is a PacRunnerError"""
    if device.endswith("-split") and device != "-split":
        device, precision = device[:-len("-split")], "split"
    name = device_name(device)
    if name not in DEVICES:
        raise PacRunnerError(f"device {device!r}: one of {', '.join(DEVICES)}, or gpu, "
                             f"{', '.join(DEVICE_ALIASES)}, or a SIMD device with -split")
    if precision not in PRECISIONS:
        raise PacRunnerError(f"precision {precision!r}: one of {', '.join(PRECISIONS)}")
    if name == "metal":
        if sys.platform != "darwin":
            raise PacRunnerError(f"device {device!r} is metal (an Apple GPU): macOS only")
        return name, "mixed"
    if name in SIMD_DEVICES:
        if name not in simd_devices():
            here = ", ".join(simd_devices()) or "none"
            raise PacRunnerError(f"device {name!r} does not build on this machine ({platform.machine()}; its SIMD "
                                 f"devices: {here})")
        if precision == "mixed":
            precision = "split"
        return name, precision
    if name == CPU_JIT:
        if not simd_devices():
            raise PacRunnerError(f"device {CPU_JIT!r} does not build on this machine ({platform.machine()}): symjit's JIT "
                                 "compiles its kernels, as a SIMD device's (x86-64 with AVX2 and FMA, or aarch64)")
        if precision != "f64":
            raise PacRunnerError(f"device {CPU_JIT!r} is f64 only (precision {precision!r}: a SIMD or GPU device)")
        return name, precision
    if precision == "split":
        raise PacRunnerError(f"precision 'split' needs a SIMD device ({', '.join(SIMD_DEVICES)}, simd)")
    if precision != "f64" and name == "cpu":
        raise PacRunnerError(f"precision {precision!r} needs a GPU device (cuda, cuda_provider, metal) or a SIMD device")
    return name, precision


def cpu_jit_isa() -> str | None:
    """the instruction set of the SIMD kernel whose scalar code device cpu-jit runs: avx2 on x86-64 (its fragments
    use 14 registers, which scalar code has too), neon on aarch64; None where no SIMD device builds"""
    here = simd_devices()
    return None if not here else ("neon" if here[0] == "simd-neon" else "avx2")


def library_name(device: str, precision: str = "f64") -> str:
    """the name of a build (compile_target's): its device, and its precision unless f64
    ("simd-avx512-split", "cuda-mixed", "cpu")"""
    device, precision = compile_target(device, precision)
    return device if precision == "f64" else f"{device}-{precision}"


def _cpu_flags() -> set:
    try:
        return set(Path("/proc/cpuinfo").read_text().split())
    except OSError:
        return set()


def simd_devices() -> tuple[str, ...]:
    """the SIMD devices that build on this machine (symjit's JIT compiles their kernels for this CPU):
    simd-avx2 (an x86-64 CPU with AVX2 and FMA) and simd-avx512 (AVX-512 F and DQ as well), simd-neon
    (aarch64: Linux or macOS)"""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        flags = _cpu_flags()
        if not {"avx2", "fma"} <= flags:
            return ()
        return ("simd-avx2", "simd-avx512") if {"avx512f", "avx512dq"} <= flags else ("simd-avx2",)
    if machine in ("aarch64", "arm64"):
        return ("simd-neon",)
    return ()


def simd_device() -> str | None:
    """this machine's SIMD device (the alias "simd"): simd-avx512 where the CPU has AVX-512, else
    simd-avx2, simd-neon on aarch64; None without one"""
    here = simd_devices()
    return here[-1] if here else None


def simd_machine() -> bool:
    """whether a SIMD device builds here"""
    return bool(simd_devices())


def c_sources() -> Path:
    """the C sources' folder (pacrunner/c/), checked"""
    if not (C_DIR / "umami_pac.c").is_file() or not (C_DIR / "pacops.c").is_file():
        raise PacRunnerError(f"{C_DIR}: no umami_pac.c, pacops.c: pacrunner is installed without its C sources "
                             "(package data); reinstall it")
    return C_DIR


def repository() -> Path | None:
    """the git clone pacrunner runs from (installed editable), else None (an installed copy)"""
    if (REPOSITORY / "pyproject.toml").is_file() and (REPOSITORY / ".git").exists():
        return REPOSITORY
    return None


@dataclass
class Config:
    madgraph7: str | None = None        # a MadGraph7 checkout (madspace/, pyamplicol/)
    cuda: bool | None = None            # MadSpace's CUDA build (None: when MadGraph7 has one)
    pyamplicol: str | None = None       # a pyAmpliCol installation: a python or an install folder


CONFIG = Config()


def configure(julia=None, madgraph7=None, cuda=None, pyamplicol=None):
    """Where pacrunner's tools are, for this process: `madgraph7` (a MadGraph7 checkout with MadSpace and
    pyAmpliCol installed; default PACRUNNER_MADGRAPH7, or an mg7 output's own), `cuda` (True: MadSpace's
    CUDA build, False: the CPU build; default: the CUDA build when MadGraph7 has one), `pyamplicol` (a
    pyAmpliCol installation for artifacts it produced: a python that imports pyamplicol, or an install
    folder such as MADGRAPH7/pyamplicol/install; default PACRUNNER_PYAMPLICOL, else found). MadSpace is
    loaded once per process: configure it before its first use. `julia` is ignored (with a warning):
    pacrunner no longer uses Julia. Returns the configuration."""
    if julia is not None:
        import warnings
        warnings.warn("pacrunner.configure(julia=...) is ignored: pacrunner no longer uses Julia", DeprecationWarning,
                      stacklevel=2)
    if madgraph7 is not None or cuda is not None:
        if "madspace" in sys.modules:
            raise PacRunnerError("MadSpace is already loaded in this process: configure(madgraph7=..., cuda=...) "
                                 "before the first runtime")
        if madgraph7 is not None:
            CONFIG.madgraph7 = os.fspath(Path(madgraph7).expanduser().resolve())
        if cuda is not None:
            CONFIG.cuda = bool(cuda)
    if pyamplicol is not None:
        CONFIG.pyamplicol = os.fspath(Path(pyamplicol).expanduser())
    return CONFIG


def madgraph7(required=True) -> Path | None:
    """the MadGraph7 checkout: configure(madgraph7=...) or PACRUNNER_MADGRAPH7"""
    path = CONFIG.madgraph7 or os.environ.get("PACRUNNER_MADGRAPH7") or None   # an empty variable: unset
    if path is None:
        if required:
            raise PacRunnerError("no MadGraph7: pacrunner.configure(madgraph7=...) or PACRUNNER_MADGRAPH7 (a MadGraph7 "
                                 "checkout with MadSpace installed: `install madspace` in its bin/madgraph)")
        return None
    path = Path(path).expanduser().resolve()
    if not (path / "bin" / "madgraph").is_file():
        raise PacRunnerError(f"{path}: not a MadGraph7 checkout (no bin/madgraph)")
    return path


def mg7_output_of(path) -> Path | None:
    """the mg7 output holding `path` (the first folder, `path` or above, with bin/generate_events)"""
    d = Path(path).expanduser().resolve()
    for p in (d, *d.parents):
        if (p / "bin" / "generate_events").is_file():
            return p
    return None


def madgraph7_of_output(output) -> Path:
    """the MadGraph7 checkout an mg7 output was made with (its bin/generate_events' sys.path line)"""
    import re
    text = (Path(output) / "bin" / "generate_events").read_text()
    m = re.search(r"sys\.path\.append\('([^']*)'\)", text)
    if m is None:
        raise PacRunnerError(f"{output}: bin/generate_events names no MadGraph7 folder")
    return Path(m.group(1)).resolve()


def madgraph7_for(path=None, required=True) -> Path | None:
    """the MadGraph7 checkout for something at `path`: the configured one, else the one of the mg7 output
    holding `path` (none: None, or the error when `required`)"""
    mg7 = madgraph7(required=False)
    if mg7 is not None:
        return mg7
    output = mg7_output_of(path) if path is not None else None
    if output is not None:
        return madgraph7_of_output(output)
    return madgraph7(required=required)   # none configured: None or the error


def madspace_dir(mg7: Path) -> Path:
    """the MadSpace build to import: install-cuda (configure(cuda=True), or by default when it exists:
    it has the CPU device too), else install"""
    cuda, cpu = mg7 / "madspace" / "install-cuda", mg7 / "madspace" / "install"
    want_cuda = CONFIG.cuda if CONFIG.cuda is not None else (cuda / "madspace").is_dir()
    path = cuda if want_cuda else cpu
    if not (path / "madspace").is_dir():
        what = "a CUDA build of MadSpace (cmake -DENABLE_CUDA=ON)" if want_cuda else "MadSpace (`install madspace`)"
        raise PacRunnerError(f"{mg7}: no {what} in {path}")
    return path


def import_madspace(mg7: Path | None = None):
    """MadSpace, imported once for this process: from the MadGraph7 checkout (`mg7`, or the configured
    one), else the one installed in this python (`import madspace`); Symbolica's lock folder defaulted
    (XDG_RUNTIME_DIR) for pyAmpliCol's libraries"""
    if "madspace" in sys.modules:
        return sys.modules["madspace"]
    mg7 = mg7 or madgraph7(required=False)
    os.environ.setdefault("XDG_RUNTIME_DIR", os.path.expanduser("~/.cache/xdg-runtime"))
    os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
    if mg7 is not None:
        sys.path.insert(0, os.fspath(madspace_dir(mg7)))
    try:
        import madspace
    except ImportError as e:
        if mg7 is not None:
            raise
        raise PacRunnerError("no MadSpace: install it in this python, or point pacrunner at a MadGraph7 checkout with "
                             "MadSpace installed (pacrunner.configure(madgraph7=...) or PACRUNNER_MADGRAPH7; "
                             "`install madspace` in its bin/madgraph)") from e
    return madspace

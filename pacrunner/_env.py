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


DEVICES = ("cpu", "simd", "cuda", "cuda_provider", "metal")
DEVICE_ALIASES = {"cuda_full": "cuda_provider"}   # cuda_full: everything on the GPU (MadSpace's GPU runtime)
PRECISIONS = ("f64", "mixed")


def device_name(device: str) -> str:
    """a compile device's own name: an alias's target (cuda_full is cuda_provider), and "gpu" this
    system's GPU with MadSpace on the CPU: metal on macOS, cuda elsewhere"""
    if device == "gpu":
        return "metal" if sys.platform == "darwin" else "cuda"
    return DEVICE_ALIASES.get(device, device)


def compile_target(device: str, precision: str = "f64") -> tuple[str, str]:
    """(device, precision) of a build: the device's own name (device_name), and for metal always
    "mixed" (Metal has no f64: mixed f32/df64); an unknown device or precision, metal off macOS, or simd
    off x86-64, is a PacRunnerError"""
    name = device_name(device)
    if name not in DEVICES:
        raise PacRunnerError(f"device {device!r}: one of {', '.join(DEVICES)}, or gpu, "
                             f"{', '.join(DEVICE_ALIASES)}")
    if precision not in PRECISIONS:
        raise PacRunnerError(f"precision {precision!r}: one of {', '.join(PRECISIONS)}")
    if name == "metal":
        if sys.platform != "darwin":
            raise PacRunnerError(f"device {device!r} is metal (an Apple GPU): macOS only")
        return name, "mixed"
    if name == "simd" and not simd_machine():
        raise PacRunnerError(f"device {device!r}: x86-64 only (AVX2 and AVX-512 kernels; this is {platform.machine()})")
    if precision != "f64" and name in ("cpu", "simd"):
        raise PacRunnerError(f"precision {precision!r} needs a GPU device (cuda, cuda_provider, metal)")
    return name, precision


def simd_machine() -> bool:
    """whether device "simd" builds here: an x86-64 machine (its kernels are AVX2 and AVX-512 objects)"""
    return platform.machine().lower() in ("x86_64", "amd64")


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

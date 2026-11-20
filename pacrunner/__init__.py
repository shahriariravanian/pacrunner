"""pacrunner: PacRunner (formerly pac-julia) in Python.

pyAmpliCol's leading-colour recurrence programs, compiled by PacRunner into UMAMI libraries (CPU, CUDA)
for MadGraph7 and MadSpace. Two layers:

  - the high-level API: Artifact (a pyAmpliCol artifact), Output and SubProcess (a MadGraph7 output and
    its pyAmpliCol subprocesses), Provider (an artifact's UMAMI provider: pyAmpliCol's stock library,
    Artifact/SubProcess.stock(), or PacRunner's build, .compile(); in an output, Provider.run:
    generate_events on it) and Runtime, the one callable for any UMAMI matrix element (one channel of
    a provider);
  - the low-level API, pacrunner.lowlevel: the recurrence export, UIR, the kernels' C, the generated C
    and the UMAMI libraries (the modules uir, kernels, codegen, umami).

Setup: this repository cloned and installed with `python -m pip install -e .`, symjit >= 2.29 (conda
install -c conda-forge 'symjit>=2.29': the kernels' MIR), a C compiler (nvcc for the GPU), a MadGraph7
checkout with MadSpace for running libraries:

    import pacrunner
    pacrunner.configure(madgraph7="~/MadGraph7")      # or PACRUNNER_MADGRAPH7
    provider = pacrunner.Output("OUTPUT").subprocess("A1_pp_jjj").compile(device="cpu")
    out = provider.runtime(channel=0)(momenta, flavours, random_helicity=r)
    out.matrix_element, out.color_index, out.helicity_index

Nomenclature: an Output holds processes (`generate`, `add process`), split by MadGraph into
subprocesses (SubProcesses/<name>), one pyAmpliCol artifact each; an artifact's provider (p0, ...) is
one UMAMI matrix-element library with its channels (each a group of flavours); a Provider is
pyAmpliCol's (stock) or PacRunner's build of it, the latter one library per (process, colour flow,
permutation) it uses. The 0.1 names
(MadOutput, MadArtifact, PacArtifact, PacLibrary, MadLibrary) remain as aliases for one release.
"""

from ._env import CONFIG, Config, PacRunnerError, configure, repository

__version__ = "0.2.0"

# imported on first use: `python -m pacrunner._madgraph` runs in MadGraph7's python, which may have
# no numpy (nor symjit: only building libraries needs it)
_LAZY = {"Output": "._mad", "SubProcess": "._mad", "Artifact": "._artifacts", "Provider": "._libraries",
         "Runtime": "._runtime", "RuntimeResult": "._runtime", "RunResult": "._mgrun", "load_libraries": "._libraries",
         "lowlevel": None, "uir": None, "kernels": None, "codegen": None, "umami": None,
         # the 0.1 names (one release)
         "MadOutput": "._mad", "MadArtifact": "._mad", "PacArtifact": "._artifacts", "PacLibrary": "._libraries",
         "MadLibrary": "._libraries"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name] or f".{name}", __name__)
        value = module if _LAZY[name] is None else getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pacrunner' has no attribute {name!r}")


__all__ = ["CONFIG", "Config", "PacRunnerError", "configure", "repository", "lowlevel", "uir", "kernels", "codegen", "umami",
           "Output", "SubProcess", "Artifact", "Provider", "Runtime", "RuntimeResult", "RunResult", "load_libraries",
           "__version__"]

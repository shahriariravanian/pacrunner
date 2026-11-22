# CLAUDE.md: the Metal runner (pacrunner/c/metal/), for work on a Mac

Guidance for PacRunner's Metal runner on macOS (Apple silicon). The file is in the repository and
shared with the Linux checkout; notes about one machine go under "Machine notes", marked as such.

## What this is

PacRunner's UIR interpreter in the Metal Shading Language, mixed f32/df64 (Metal has no double), as
UMAMI libraries that MadSpace loads as CPU providers (`device="metal"`, or `"gpu"` on macOS). Ported
from pac-julia (main 15991b1, its PR #26); `PLAN.md` beside this file is pac-julia's plan and results
(design, precision split, phases, the M4's numbers), still valid: read it first.

- Device code: `df64.h`, `pacops.metal`, `uirvm.metal`, concatenated per program with its kernels
  into `runner.metal` by `pacrunner/codegen/compile_metal.py`; the kernels by
  `pacrunner/kernels/kernel_msl.py`.
- Host: `uirmetal.m` (compile with `MTLMathModeSafe`, launch), `metalpool.m` (the process-wide slot
  pool, `libpacmetal.dylib`), `umami_metal.c` (the UMAMI layer); built by `pacrunner/umami.py`
  (`build_metal_runtime`, `build_umami_metal`).
- `msl_shim.h`: enough MSL for the `.metal` sources to compile as C++, so Linux tests them on the CPU.

## Rules

- `pacrunner/c/pacops.c` ↔ `pacops.metal` and `pacrunner/c/uirvm.c` ↔ `uirvm.metal` are kept in
  step: a fix in one belongs in the other (and in pac-julia's copies).
- `compile_metal`'s files and `kernel_msl`'s MSL equal pac-julia's byte for byte (the goldens,
  `tools/goldens/`); `kernel_c_mixed`'s C must not change when the shared mixed writer
  (`kernels/kernel_c.py`, language "c" or "msl") does.
- Accuracy is checked with tolerances (matrix element median ~2e-7, max ~1e-4; the GPU equal to the
  CPU shim bit for bit), never by loosening a tolerance to make a failure pass: find the cause.
- Record results (numbers, dates, the machine) in `PLAN.md`; keep this file for instructions.

## Tests

```sh
python -m unittest discover -s test/python -p test_metal.py     # from the clone (or -t test/python from / for an installed copy)
```

- `Goldens`: compile_metal's folders against pac-julia's (any system).
- `Shim`: the sources on the CPU through `msl_shim.h` (any system with a C++ compiler):
  `test/metal_shim_ops.cpp`, `test/metal_shim_run.cpp`.
- `Gpu` (macOS with a Metal GPU): each program built (`make`: `librunner_metal.dylib`) and run
  (`run_metal`) at its reference points, against RustiCol's values and the shim (bit for bit). The
  fixtures' exports, plus `PACRUNNER_TEST_METAL_EXPORTS` (":"-separated exports made with
  `pyamplicol export-recurrence LC_DIR -o FILE --reference-points MOMENTA.json`).
- `test_artifacts.py` on macOS compiles `device="gpu"` and compares it with the stock library;
  `test_mad.py`, `test_umami.py` check the device rules everywhere.

mg7 runs (the libraries are CPU providers: `run.device = ["cpu"]`, `color_accuracy = "lc"`):

```sh
PACRUNNER_PYTHON=<a python with pacrunner> python test/mg7_run.py prepare OUTPUT --device metal
python test/mg7_run.py run OUTPUT --device metal -f
python test/mg7_bench.py OUTPUT --modes stock,cpu,metal [--no-timer]     # σ, walls, the ME's time
```

## Things that bite

- Metal compiles each program's source at run time and caches it: the first run of a big program is
  slow, later runs fast. Time warm runs only.
- f32 denormals are flushed on the GPU (the shim does the same: FPCR.FZ on arm64, FTZ/DAZ on x86-64).
- `MTLMathModeSafe` does not stop fma contraction: `runner.metal`'s prelude has
  `#pragma METAL fp contract(off)`; without it the GPU differs from the shim by an ulp here and there.
- A Metal run may sample other points than the CPU run (MadSpace adapts on values ~1e-7 apart):
  compare σ within errors there, not to 1e-6.
- With many libraries the timer shims (`PAC_UMAMI_TIMER`) are built at the run's start: take wall
  times from `--no-timer` runs.
- Knobs: `PAC_METAL_CHUNK` (points per launch, default 4096), `PAC_METAL_SLOTS` (default the CPUs),
  `PAC_METAL_SLOT_BYTES` (default 256 MiB).

## Machine notes

**Apple M4 (10 cores, 16 GB), macOS 26.6.2, Command Line Tools (Apple clang 17)**, 2026-11-22:
- MadGraph7 in `~/mg7` (local branch `mg7-2079d93`: the Linux revision), MadSpace (CPU, SIMD) and
  pyAmpliCol built from source in it, from the Miniforge env `mg7` (Python 3.12).
- PacRunner installed not editable in `~/pacrunner-venv` (`--system-site-packages` on the `mg7` env;
  symjit from PyPI); the source in `~/pacrunner`.
- Builds: `CC=/usr/bin/clang CXX=/usr/bin/clang++`, and `/usr/bin` before the env on `PATH` (conda's
  `install_name_tool` leaves invalid code signatures: `import madspace` is killed).

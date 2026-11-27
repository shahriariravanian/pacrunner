# Benchmarks

PacRunner against the stock pyAmpliCol libraries in MadGraph7 `generate_events` runs, on five processes:
`p p > z j j`, `p p > j j j`, `g g > t t~ g g` (full colour), `u u~ > z g g g g` and `g g > g g g g`
(100k events, all threads, 64k batches; fixed scales, NNPDF23, seed 7; leading colour except `g g > t t~ g g`).

```sh
export MG7=/path/to/MadGraph7 RUNPY=/path/to/MadGraph7s/python    # WORK: default ~/pacrunner-bench
bench/make_outputs.sh               # the outputs in WORK/outputs, once (some minutes)
bench/bench_full.sh                 # every variant this machine runs
bench/bench_cpu.sh                  # the same without the GPU variants
```

Both benchmark this repository's HEAD (copied with `git archive` to `WORK/pacrunner-<commit>/`), or a checkout
or copied tree given as the first argument: `bench_full.sh PACRUNNER [TAG] [-- bench.py options]`, e.g.
`bench_cpu.sh . quick -- --names pp_zjj --repeats 1`. The table is `WORK/logs/<suite>-<commit>[-TAG]/summary.txt`,
the machine and revisions in `machine.txt` beside it. All paths and the pythons: `run.sh`; the options:
`python bench/bench.py --help`. Symbolica (pyAmpliCol) needs a writable `XDG_RUNTIME_DIR`.

| Variant | PacRunner device, precision | Suite |
| --- | --- | --- |
| `stock` | the output's stock pyAmpliCol library | cpu, full |
| `cpu` | `cpu`, f64 | cpu, full |
| `simd-avx2` / `simd-neon` | `simd-avx2` / `simd-neon`, f64 | cpu, full |
| `simd-split-avx2` / `simd-split-neon` | the same, `split` (f64 kernel A, f32 kernel B) | cpu, full |
| `simd-avx512`, `simd-split-avx512` | `simd-avx512`, f64 / `split` (x86-64 with AVX-512) | cpu, full |
| `cuda-f64`, `cuda-mixed` / `metal-mixed` | `cuda`, f64 / mixed; `metal` (macOS: mixed); MadSpace on the CPU | full |
| `cuda-full-f64`, `cuda-full-mixed` | `cuda_full`, f64 / mixed: everything on the GPU (MadSpace's GPU runtime) | full |

A variant runs where the machine has it (`bench.py --list` shows which). `cuda-full` runs on `NAME_gpu`, a
copy of the output with `run.device = ["cuda"]` that `bench.py` makes, for the leading-colour processes
only, and needs MadSpace's CUDA build (`MG7/madspace/install-cuda`) and CuPy.

The passes: A, the wall clock, 3 uninstrumented runs of each variant (`REPEATS`); B, the time inside the
matrix element, one run each with `PAC_UMAMI_TIMER` (`test/umami_timer.c`; summed over MadSpace's threads,
so a GPU variant's includes threads waiting for the shared GPU); C, for `cuda-full`, whose calls return
once queued on MadSpace's stream: its libraries on synchronized 64k-event batches
(`test/bench_libraries.py`, best of 5), marked `*` in the table. The table also gives each variant's
cross section against stock's and whether its events are stock's or `cpu`'s bit for bit (the f64 SIMD
devices give `cpu`'s).

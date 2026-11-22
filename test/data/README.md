# Test data

| Path | What it is | Used by |
| --- | --- | --- |
| `export/pp_3j.json`, `export/zggg_union.json` | RustiCol recurrence exports (`pyamplicol export-recurrence`, schema 2) with reference points: a cut of pp > jjj (12 programs, 4 of them crossed, 4 symjit kernels) and of the Z+3g union (1 program, 3 kernels) | `test/python/test_uir.py`, `test_kernels.py`, `test_codegen.py`, `test_umami.py` |
| `goldens/` | pac-julia's outputs for the two exports (`tools/goldens/make_goldens.jl --compile-max 0`, pac-julia main 15991b1): each program's `Lowered`, its rows after `optimize`, both encodings, the kernels' C (full and mixed precision) and MSL (`.msl`), for pp_3j compiled folders, and for both compile_metal folders (without `runner_metal.c`, 1.3 MB each: the larger corpus has them; zggg_union's `compile/`, 8 MB, left out likewise) | the same tests: the port must equal them byte for byte |
| `julia_repr.txt` | doubles as bit patterns and Julia's `repr` of each (2,209 values): the spelling of the C's literals | `test_kernels.py` (`julia_repr`) |
| `julia_repr_f32.txt` | floats as bit patterns and Julia's `repr` of each (24,356 values: random bit patterns, powers of ten, rounded decimals, df64 halves): the spelling of the MSL's literals | `test_kernels.py` (`julia_repr_f32`) |

The larger golden corpus (8 exports, 856 programs, 83 UMAMI libraries) is not in the repository:
`tools/goldens/README.md` says how it is made; the tests use it from `PACRUNNER_GOLDENS` (default
`~/symjit-data/pacrunner-goldens`) when it is there.

What a program's values and operations mean is in `CONVENTIONS.md` at the root.

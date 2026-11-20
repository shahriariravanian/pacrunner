# Goldens: pac-julia's outputs for the Python port

`make_goldens.jl` runs pac-julia (PacRunner.jl) on RustiCol recurrence exports and writes what it makes
of them: each program's `Lowered`, its optimized rows, the 8-byte encodings, the kernels' C, the
generated C folders and the UMAMI layer's generated files. The Python port must reproduce them: the
JSON value for value, the C byte for byte.

```bash
julia --project=$HOME/pac-julia tools/goldens/make_goldens.jl [--compile-max N] OUT EXPORT[,METADATA] ...
```

`EXPORT` is a schema-2 recurrence export (`pyamplicol export-recurrence`, with `--reference-points` for
`testing.c`); `METADATA` its artifact's `API/umami/metadata.json` (optional: the UMAMI goldens). The
project is a pac-julia checkout; `manifest.json` records its revision.

## Layout of `OUT/<export name>/`

| Path | What |
|---|---|
| `manifest.json` | `format` "pacrunner-goldens", `format_version` 1, `export_path`, `export_sha256`, `artifact_id`, `metadata`, `pac_julia` (revision, `+` when its tree had changes), `julia`, `written`, the counts, `compiled` (the entries compiled) |
| `programs.json` | per program entry, in the export's order: `index`, `process`, `colour`, `member` (a process-group member or alias), `kernel_rows`, `references` |
| `lowered/NNNN.json` | entry NNNN: `lowered` (as `load_recurrence_export` makes it), `optimized_ops` (the rows after `optimize`; the tables are unchanged by it, which the script checks), `encoded` and `encoded_optimized` (`version`, `rows` as 64-bit hex words with the END word last, `edges` as 32-bit integers; or `error`) |
| `kernels/<sha256>.c`, `<sha256>.mixed.c` | `kernel_c` and `kernel_c_mixed` of each kernel, named as `compile` names it (`k_` and the first 16 hex digits) |
| `compile/NNNN/<variant>/` | the files `compile` writes (`runner.c`, `runner.h`, `Makefile`, `testing.c` on the export's reference points), without the copies of `c/`; variants `imported-straight`, `imported-interpreter`, `optimized-straight`, `optimized-interpreter`; an `ERROR` file when `compile` refuses. For a sample of the entries (`--compile-max`, default 12, evenly spaced, plus the first two with kernel rows and the first two members) |
| `umami/libraries.json` | the libraries `umami_libraries` would build (`index`, `process`, `colour`, `permutation`, `colour_index`, `factor`, `helicity_export`) and per channel `flavor_libraries`, `libraries` |
| `umami/NNNN/{straight,interpreter}/`, `umami_tables.h`, `umami_program.h` | per library: its program optimized, its permutation folded into the crossing, compiled (straight-line: the CPU build; interpreter: the CUDA builds), and the UMAMI headers. Nothing is built |

## Conventions (the Python port's, not Julia's)

- **Indices are 0-based; an absent one is `null`** (Julia's 0).
- **Every double is its bit pattern**, `"0x"` and 16 hex digits (as symjit's MIR JSON); a complex value
  is a pair `[re, im]`.
- **Rows** (`ops`, `optimized_ops`): `op` (the opcode's name, e.g. `V_THREE_VECTOR`), `set`, `first`,
  `last`, and the fields `encode` writes for the opcode (`PacRunner.fields`): index fields shifted to
  0-based (`null` for none), counts as they are (the `n` of SCALE_ROW and CLEAR, a SOURCE's template).
  The other fields are 0 in Julia (checked) and left out.
- `lowered`: `version`, `process`, `colour`, `legs`, `crossing` (`[leg, sign]` per program leg, or
  null), `forms` (`[leg, coefficient]` terms), `scales`, `factors`, `alpha_s` (`{value, power}` or
  null), `helicities`, `currents` (`[start plane, components, type]`), `planes`, `edges`
  (`[current, form or null]`), `lists`, `embeddings` (`[component, wavefunction component or null,
  factor]`), `stubs` (`{kernel: sha256, name, inputs: [{kind, n, im}], outputs}`; a FACTOR input's `n`
  is null), `scalesets`, `instances` (`[stub, scale set]`), `ops`, `amplitudes`, `weights`,
  `group_factor` (`[numerator, denominator]`), `fixed_layout`.

## Checks of the goldens themselves (2026-11-20, pac-julia 8a666da)

Corpus: the eight pyAmpliCol 1.0.2 reference exports (`~/pyamplicol-1.0/exports2ref` on the x86-64
machine: gg_3g, gg_4g, pp_3j, ttgg, uu_ttgg, zgg, zgggg, zggg_union; 856 programs, 83 UMAMI
libraries), written to `~/symjit-data/pacrunner-goldens` (181 MB, 18 s).

- Every compiled folder (336: 84 entries × 4 variants), with pac-julia's `c/` copied in, passes
  `make test`: every output bit-identical to RustiCol's reference values.
- A second run gives the same files (but `manifest.json`'s `written`).

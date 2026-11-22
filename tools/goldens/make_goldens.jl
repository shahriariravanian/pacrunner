# The goldens of the Python port: what pac-julia (PacRunner.jl) makes of RustiCol's recurrence
# exports, written as files the Python tests compare with, value for value and byte for byte.
#
#   julia --project=$HOME/pac-julia make_goldens.jl [--compile-max N] OUT EXPORT[,METADATA] ...
#
# For each EXPORT (a schema-2 recurrence export, `pyamplicol export-recurrence`; with
# `--reference-points` for testing.c) the folder OUT/<export name>/ gets:
#
#   manifest.json        the export (path, SHA-256, artifact), pac-julia's revision, Julia's version,
#                        the counts of what was written
#   programs.json        one entry per program entry of the export, in the file's order
#   lowered/NNNN.json    program NNNN (0-based entry index): the Lowered load_recurrence_export
#                        makes ("lowered"), the rows after `optimize` ("optimized_ops"; the tables are
#                        unchanged by it: checked), and the 8-byte encodings of both ("encoded",
#                        "encoded_optimized")
#   kernels/<sha>.c      kernel_c of each kernel (named as compile names it: k_ and the first 16 hex
#                        digits of its SHA-256), <sha>.mixed.c kernel_c_mixed, <sha>.msl kernel_msl
#                        (the Metal runner's; <sha>.msl.ERROR when it refuses the kernel)
#   compile/NNNN/<variant>/
#                        the files compile writes (runner.c, runner.h, Makefile, and testing.c on the
#                        export's reference points), not the copies of c/; variants: imported and
#                        optimized, each straight-line and interpreter (an ERROR file when compile refuses)
#                        for a sample of the programs (--compile-max, default 12, plus the first two
#                        with kernel rows and the first two members of a process group)
#   compile_metal/NNNN/<variant>/
#                        the files compile_metal writes (runner.metal, program.bin, runner_metal.c,
#                        runner_metal.h, Makefile), not the copies of c/metal/; variants imported and
#                        optimized; the same sample (an ERROR file when compile_metal refuses)
#   umami/libraries.json, umami/NNNN/<variant>/
#                        with METADATA (the artifact's API/umami/metadata.json): every library
#                        umami_libraries would build (its program optimized, its leg permutation folded
#                        into the crossing): the compiled files of both variants (straight-line: the CPU
#                        build; interpreter: the CUDA builds), compile_metal's files (metal: the Metal
#                        build) and umami_tables.h, umami_program.h; nothing is built
#
# The JSON follows the Python port's conventions, not Julia's: every index is 0-based and an absent
# one is null (Julia's 0); every double is its bit pattern, "0x" and 16 hex digits (as symjit's MIR
# JSON), a complex value a pair of them. Row fields are those encode.jl's `fields` gives each opcode
# (the rest must be 0), index fields shifted, counts (a SCALE or CLEAR's n, a SOURCE's template) as
# they are. The format is described in README.md beside this script.

using PacRunner
using JSON
using SHA
using Dates

const P = PacRunner
const FORMAT_VERSION = 2   # 2: the Metal runner (kernels/*.msl, compile_metal/, umami/NNNN/metal/)

# ---- values ----------------------------------------------------------------------------------------

bits(x::Real) = "0x" * string(reinterpret(UInt64, Float64(x)); base = 16, pad = 16)
cbits(z::Complex) = [bits(real(z)), bits(imag(z))]
idx(v::Integer) = v == 0 ? nothing : Int(v) - 1     # an index that may be absent
req(v::Integer) = (v >= 1 || error("index $v: absent where one is required"); Int(v) - 1)

kernel_sha(k) = replace(basename(k.source), ".plane.symjit" => "")

# ---- a Lowered ---------------------------------------------------------------------------------------

function op_json(op)
    used = P.fields(op.op)
    names = Set(s for (s, _, _) in used)
    for s in (:d, :t, :l, :r, :n, :x)
        s in names || getfield(op, s) == 0 || error("$(op.op): field $s = $(getfield(op, s)) is not one of its fields")
    end
    row = Pair{Symbol,Any}[:op => string(op.op), :set => op.set, :first => op.first, :last => op.last]
    for (s, _, index) in used
        v = getfield(op, s)
        push!(row, s => (index ? idx(v) : v))
    end
    return (; row...)
end

input_json(i) = (kind = string(i.kind), n = i.kind == P.FACTOR ? (i.n == 0 ? nothing : error("FACTOR input n = $(i.n)")) : req(i.n), im = i.im)

stub_json(s) = (kernel = kernel_sha(s.kernel), name = s.kernel.name, inputs = input_json.(s.inputs), outputs = req.(s.outputs))

function lowered_json(l)
    (
        version = string(l.version),
        process = l.process,
        colour = l.colour,
        legs = l.legs,
        crossing = l.crossing === nothing ? nothing : [[req(leg), bits(sign)] for (leg, sign) in l.crossing],
        forms = [[[req(leg), bits(c)] for (leg, c) in f.terms] for f in l.forms],
        scales = cbits.(l.scales),
        factors = cbits.(l.factors),
        alpha_s = l.alpha_s === nothing ? nothing : (value = bits(l.alpha_s[1]), power = l.alpha_s[2]),
        helicities = l.helicities,
        currents = [[c.start, c.components, string(c.type)] for c in l.currents],
        planes = l.planes,
        edges = [[req(e.current), idx(e.form)] for e in l.edges],
        lists = [req.(list) for list in l.lists],
        embeddings = [[[req(j), idx(w), req(x)] for (j, w, x) in list] for list in l.embeddings],
        stubs = stub_json.(l.stubs),
        scalesets = [req.(set) for set in l.scalesets],
        instances = [[req(s), req(set)] for (s, set) in l.instances],
        ops = op_json.(l.ops),
        amplitudes = l.amplitudes,
        weights = bits.(l.weights),
        group_factor = collect(l.group_factor),
        fixed_layout = l.fixed_layout,
    )
end

encoded_json(e) = (version = string(e.version), rows = ["0x" * string(w; base = 16, pad = 16) for w in e.rows], edges = Int.(e.edges))

# the encoding, or the error encode gives
encoded_or_error(l) = try
    encoded_json(encode(l))
catch err
    (error = sprint(showerror, err),)
end

# optimize changes the rows only (sort_runs, fuse; reuse_planes keeps a fixed layout)
function optimized(l)
    o = optimize(l)
    for f in fieldnames(typeof(l))
        f == :ops && continue
        isequal(getfield(o, f), getfield(l, f)) || error("$(l.process) $(l.colour): optimize changed $f")
    end
    return o
end

# ---- files --------------------------------------------------------------------------------------------

writejson(path, x) = (mkpath(dirname(path)); write(path, JSON.json(x)))
const GENERATED = ("runner.c", "runner.h", "Makefile", "testing.c")

# compile's generated files for `l` into `dir` (not the copies of c/), or an ERROR file
function compiled!(dir, l; interpreter::Bool, examples = nothing)
    mkpath(dir)
    try
        mktempdir() do tmp
            compile(tmp, l; interpreter, examples)
            for f in GENERATED
                isfile(joinpath(tmp, f)) && cp(joinpath(tmp, f), joinpath(dir, f); force = true)
            end
        end
    catch err
        write(joinpath(dir, "ERROR"), sprint(showerror, err))
    end
end

const GENERATED_METAL = ("runner.metal", "program.bin", "runner_metal.c", "runner_metal.h", "Makefile")

# compile_metal's generated files for `l` into `dir` (not the copies of c/metal/), or an ERROR file
function compiled_metal!(dir, l)
    mkpath(dir)
    try
        mktempdir() do tmp
            compile_metal(tmp, l)
            foreach(f -> cp(joinpath(tmp, f), joinpath(dir, f); force = true), GENERATED_METAL)
        end
    catch err
        write(joinpath(dir, "ERROR"), sprint(showerror, err))
    end
end

function compile_variants!(dir, l, o; examples = nothing)
    for (name, prog) in (("imported", l), ("optimized", o)), interpreter in (false, true)
        compiled!(joinpath(dir, "$name-$(interpreter ? "interpreter" : "straight")"), prog; interpreter, examples)
    end
end

examples_of(l, ref) = ref === nothing ? nothing :
    (process = l.process, colour = l.colour, points = [(momenta = m, outputs = v) for (m, v) in zip(ref...)])

# ---- one export ----------------------------------------------------------------------------------------

function sample(n::Int, max::Int, extra)
    picked = max >= n ? collect(1:n) : unique(round.(Int, range(1, n; length = max)))
    return sort(unique(vcat(picked, extra)))
end

function goldens(out::String, path::String, metadata, compile_max::Int)
    name = replace(basename(path), r"\.json$" => "")
    dir = joinpath(out, name)
    isdir(dir) && rm(dir; recursive = true)
    d = P.read_export(path)
    ls = load_recurrence_export(path; kernels = :none)
    refs = recurrence_references(path)
    entries = d["programs"]
    pad(k) = lpad(k - 1, 4, '0')

    # the programs: each Lowered, its optimized rows, both encodings
    programs = []
    for (k, l) in enumerate(ls)
        o = optimized(l)
        writejson(joinpath(dir, "lowered", "$(pad(k)).json"),
                  (lowered = lowered_json(l), optimized_ops = op_json.(o.ops), encoded = encoded_or_error(l),
                   encoded_optimized = encoded_or_error(o)))
        push!(programs, (index = k - 1, process = l.process, colour = l.colour, member = haskey(entries[k], "program_reference"),
                         kernel_rows = count(op -> P.is_kernel(op.op), l.ops), references = refs[k] !== nothing))
    end
    writejson(joinpath(dir, "programs.json"), programs)

    # the kernels, as compile names them
    kernels = Dict{String,Any}()
    for l in ls, s in l.stubs
        kernels[kernel_sha(s.kernel)] = s.kernel
    end
    for (sha, k) in kernels
        mkpath(joinpath(dir, "kernels"))
        write(joinpath(dir, "kernels", "$sha.c"), P.kernel_c(k.name, k.source))
        write(joinpath(dir, "kernels", "$sha.mixed.c"), P.kernel_c_mixed(k.name, k.source))
        try
            write(joinpath(dir, "kernels", "$sha.msl"), P.kernel_msl(k.name, k.source))
        catch err
            write(joinpath(dir, "kernels", "$sha.msl.ERROR"), sprint(showerror, err))
        end
    end

    # compile: a sample, with the first programs that have kernel rows and the first members
    with_kernels = first([k for (k, p) in enumerate(programs) if p.kernel_rows > 0], 2)
    members = first([k for (k, p) in enumerate(programs) if p.member], 2)
    picked = sample(length(ls), compile_max, vcat(with_kernels, members))
    for k in picked
        compile_variants!(joinpath(dir, "compile", pad(k)), ls[k], optimized(ls[k]); examples = examples_of(ls[k], refs[k]))
        for (variant, prog) in (("imported", ls[k]), ("optimized", optimized(ls[k])))
            compiled_metal!(joinpath(dir, "compile_metal", pad(k), variant), prog)
        end
    end

    # the UMAMI libraries umami_libraries would build (nothing built)
    nlibs = 0
    if metadata !== nothing
        meta = JSON.parse(read(metadata, String))
        provider = meta["provider"]
        provider["artifact_id"] == d["artifact_id"] || error("$metadata is of artifact $(provider["artifact_id"]), $path of $(d["artifact_id"])")
        exported_count = length(meta["helicities"])
        parameters = [(String(p["name"]), P.cvalue(p["value"])) for p in d["parameters"]]
        progs = P.export_programs(d, path)
        (libraries, channels) = P.provider_libraries(meta, progs, path)
        for (k, lib) in enumerate(libraries)
            l = P.permuted(optimize(progs[(lib.process, lib.colour)][1]), lib.permutation)
            ldir = joinpath(dir, "umami", pad(k))
            for interpreter in (false, true)
                compiled!(joinpath(ldir, interpreter ? "interpreter" : "straight"), l; interpreter)
            end
            compiled_metal!(joinpath(ldir, "metal"), l)
            (tables, program) = P.umami_program_h(lib, l, provider, exported_count, parameters)
            write(joinpath(ldir, "umami_tables.h"), tables)
            write(joinpath(ldir, "umami_program.h"), program)
        end
        writejson(joinpath(dir, "umami", "libraries.json"),
                  (libraries = [(index = k - 1, process = lib.process, colour = lib.colour, permutation = lib.permutation,
                                 colour_index = lib.colour_index, factor = bits(lib.factor), helicity_export = lib.helicity_export)
                                for (k, lib) in enumerate(libraries)],
                   channels = [(flavor_libraries = fl .- 1, libraries = sort(unique(fl)) .- 1) for fl in channels]))
        nlibs = length(libraries)
    end

    writejson(joinpath(dir, "manifest.json"),
              (format = "pacrunner-goldens", format_version = FORMAT_VERSION, export_path = recorded(path),
               export_sha256 = bytes2hex(open(sha256, path)), artifact_id = d["artifact_id"],
               metadata = metadata === nothing ? nothing : recorded(metadata), pac_julia = pac_julia_revision(),
               julia = string(VERSION), written = string(now(UTC)) * "Z", programs = length(ls), kernels = length(kernels),
               compiled = [k - 1 for k in picked], umami_libraries = nlibs))
    println("$name: $(length(ls)) programs, $(length(kernels)) kernels, $(length(picked)) compiled, $nlibs UMAMI libraries")
end

# a path as the manifest records it: relative to this repository when inside it (the fixtures'
# exports: no machine's paths in the repository), else absolute
const REPOSITORY = dirname(dirname(@__DIR__))
recorded(path) = (p = abspath(path); startswith(p, REPOSITORY * "/") ? relpath(p, REPOSITORY) : p)

# PacRunner's git revision, "+" when its tree has changes
function pac_julia_revision()
    repo = dirname(dirname(pathof(PacRunner)))
    rev = readchomp(`git -C $repo rev-parse HEAD`)
    dirty = !isempty(readchomp(`git -C $repo status --porcelain --untracked-files=no`))
    return dirty ? rev * "+" : rev
end

function main(args)
    compile_max = 12
    if length(args) >= 2 && args[1] == "--compile-max"
        compile_max = parse(Int, args[2])
        args = args[3:end]
    end
    length(args) >= 2 || error("usage: make_goldens.jl [--compile-max N] OUT EXPORT[,METADATA] ...")
    out = args[1]
    for a in args[2:end]
        (path, metadata) = occursin(',', a) ? Tuple(split(a, ','; limit = 2)) : (a, nothing)
        goldens(out, String(path), metadata === nothing ? nothing : String(metadata), compile_max)
    end
end

main(ARGS)

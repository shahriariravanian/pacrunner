"""What the straight-line (compile.py) and the interpreter (compile_vm.py) generators share: the output
folder's writer, the arena's layout, C literals, the kernels a program uses, and the files beside
runner.c (runner.h, testing.c, the Makefile). Pac-julia's src/compile.jl; the files are its byte for
byte (their comments too, which name the Julia runner)."""

from __future__ import annotations

import functools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .. import _env
from .._literal import julia_repr
from ..kernels import KERNEL_PRELUDE, Mir, kernel_c, kernel_c_mixed, read_mir
from ..uir import Kernel, Lowered

# the C sources (pacops, the interpreter, the GPU pool): the package's c/ folder
C_DIR = _env.C_DIR


class CompileError(ValueError):
    """a program compile cannot write"""


class CodeWriter:
    """runner.c in the folder `path`, written line by line, indented by its braces (a tab per open
    brace, as pac-julia's CodeWriterCpu); pacops.c and pacops.h are copied beside it."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []
        self.indent = 0
        self.copy_home("pacops.c", "pacops.h")

    def copy_home(self, *names: str) -> None:
        """files of c/ (paths within it: "cuda/pool.h") into the output folder, flat"""
        for name in names:
            shutil.copyfile(C_DIR / name, self.path / Path(name).name)

    def emit(self, s: str) -> None:
        self.lines.append("\t" * self.indent + s + "\n")
        self.indent = max(self.indent + s.count("{") - s.count("}"), 0)

    def emit_lines(self, text: str) -> None:
        """each line of `text` (but a final newline's empty one)"""
        for line in (text[:-1] if text.endswith("\n") else text).split("\n"):
            self.emit(line)

    def newline(self) -> None:
        self.lines.append("\n")

    def close(self) -> None:
        (self.path / "runner.c").write_text("".join(self.lines), encoding="utf-8")


@dataclass(frozen=True)
class ArenaIndex:
    """The arena of runner.c, in doubles (0-based offsets): the forms (4 each), the amplitudes (2
    each), then the planes (2 each), a current at planes + 2 × its plane offset"""
    forms: int        # the first form's offset
    amplitudes: int   # the first amplitude's
    planes: int       # plane offset 0's
    size: int         # the arena's doubles (RUNNER_ARENA_SIZE)

    @staticmethod
    def of(l: Lowered) -> "ArenaIndex":
        forms = 0
        amplitudes = forms + 4 * len(l.forms)
        planes = amplitudes + 2 * l.amplitudes
        return ArenaIndex(forms, amplitudes, planes, planes + 2 * l.planes)

    def current_at(self, l: Lowered, c: int) -> int:
        return self.planes + 2 * l.currents[c].start

    def edge_at(self, l: Lowered, e: Optional[int]) -> int:
        if e is None:
            raise CompileError("an instruction reads a null operand")
        return self.current_at(l, l.edge_current(e))

    def form_at(self, f: Optional[int]) -> int:
        if f is None:
            raise CompileError("an instruction reads no momentum form")
        return self.forms + 4 * f

    def amplitude_at(self, a: int) -> int:
        return self.amplitudes + 2 * a


def lit(x: float) -> str:
    """a double as a C literal: the shortest digits that read back as the same double (Julia's
    spelling); NaN and Inf have no C literal"""
    y = float(x)
    if y != y or y in (float("inf"), float("-inf")):
        raise CompileError(f"{y} has no C literal")
    return julia_repr(y)


def cstring(s: str) -> str:
    """a string as a C string literal"""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


# ---- the kernels -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelInfo:
    """a kernel with its MIR: its name, inputs (count_states) and outputs (count_obs: 2 × the pairs)"""
    kernel: Kernel
    mir: Mir

    @property
    def name(self) -> str:
        return self.kernel.name

    @property
    def inputs(self) -> int:
        return self.mir.count_states

    @property
    def outputs(self) -> int:
        return self.mir.count_obs


@functools.lru_cache(maxsize=256)
def kernel_info(kernel: Kernel) -> KernelInfo:
    """a kernel's MIR, read once (symjit.mir_json)"""
    return KernelInfo(kernel, read_mir(kernel.application))


def used_kernels(prog: Lowered) -> list[KernelInfo]:
    """the kernels of `prog` in order of first use by a stub, each checked against its stubs: a stub
    lists as many inputs and output pairs as its kernel has (the outputs array would otherwise be read
    past its end)"""
    out: list[KernelInfo] = []
    for stub in prog.stubs:
        k = kernel_info(stub.kernel)
        if not (len(stub.inputs) == k.inputs and len(stub.outputs) == k.outputs // 2):
            raise CompileError(f"a binding of kernel {k.name} lists {len(stub.inputs)} inputs, {len(stub.outputs)} outputs; "
                               f"the kernel has {k.inputs}, {k.outputs // 2}")
        if all(x.name != k.name for x in out):
            out.append(k)
    return out


def emit_kernels(writer: CodeWriter, kernels: Sequence[KernelInfo], mixed: bool = False) -> None:
    """the C of the kernels (kernels/kernel_c.py), the helpers first: `static int <name>(double *mem)`
    each; mixed: each kernel also in mixed precision, the one compiled under PAC_MIXED"""
    if not kernels:
        return
    writer.emit_lines(KERNEL_PRELUDE)
    for k in kernels:
        if mixed:
            writer.emit("#ifdef PAC_MIXED")
            writer.emit_lines(kernel_c_mixed(k.name, k.mir))
            writer.emit("#else")
            writer.emit_lines(kernel_c(k.name, k.mir))
            writer.emit("#endif")
        else:
            writer.emit_lines(kernel_c(k.name, k.mir))
        writer.newline()


# ---- examples: the reference points testing.c runs -----------------------------------------------------

@dataclass(frozen=True)
class Examples:
    """a program's reference points: per point the caller's legs (E, px, py, pz each) and the outputs"""
    process: str
    colour: str
    points: tuple    # ((momenta, outputs), ...)

    @staticmethod
    def of(l: Lowered, references) -> Optional["Examples"]:
        """the examples of program `l` from its entry of uir.recurrence_references (None: none)"""
        if references is None:
            return None
        (momenta, values) = references
        return Examples(l.process, l.colour, tuple(zip(momenta, values)))


def check_examples(prog: Lowered, examples: Examples) -> None:
    """`examples` must be this program's entry, its points of the program's shape"""
    if (examples.process, examples.colour) != (prog.process, prog.colour):
        raise CompileError(f"the examples are of {examples.process} {examples.colour}, the program {prog.process} {prog.colour}")
    if not examples.points:
        raise CompileError("the examples have no points")
    for (momenta, outputs) in examples.points:
        if not (len(momenta) == prog.legs and all(len(v) == 4 for v in momenta) and len(outputs) == prog.output_count()):
            raise CompileError(f"an example point has {len(momenta)} momenta and {len(outputs)} outputs; "
                               f"the program has {prog.legs} legs and {prog.output_count()} outputs")


# ---- the files beside runner.c ---------------------------------------------------------------------------

def runner_h(prog: Lowered, b: ArenaIndex) -> str:
    return f"""\
/* {prog.process} {prog.colour}: the program compiled by PacRunner's compile (runner.c).
 * run(out, p, momenta) evaluates one phase-space point: momenta[4 * RUNNER_LEGS] are the caller's
 * legs (E, px, py, pz each, before the crossing), out[RUNNER_OUTPUTS] receives the helicity
 * outputs (group factor applied; the matrix element is their sum), p[RUNNER_ARENA_SIZE] is
 * scratch. Returns 0. */
#ifndef RUNNER_H
#define RUNNER_H

#define RUNNER_PROCESS {cstring(prog.process)}
#define RUNNER_COLOUR {cstring(prog.colour)}
#define RUNNER_LEGS {prog.legs}
#define RUNNER_OUTPUTS {prog.output_count()}
#define RUNNER_ARENA_SIZE {b.size}

#ifdef __cplusplus
extern "C" {{
#endif

int run(double *out, double *p, const double *momenta);

#ifdef __cplusplus
}}
#endif

#endif
"""


TESTING_MAIN = r"""
static double p[RUNNER_ARENA_SIZE];

int main(int argc, char **argv)
{
    double tolerance = argc > 1 ? atof(argv[1]) : 1e-14;
    double out[RUNNER_OUTPUTS], worst = 0.0;
    int beyond = 0, identical = 0;
    for (int i = 0; i < POINTS; i++) {
        if (run(out, p, momenta[i]) != 0) {
            printf("point %d: run failed\n", i);
            return 1;
        }
        double total = 0.0;
        for (int h = 0; h < RUNNER_OUTPUTS; h++) total += expected[i][h];
        double scale = fabs(total) > 0.0 ? fabs(total) : DBL_MIN;
        for (int h = 0; h < RUNNER_OUTPUTS; h++) {
            double d = fabs(out[h] - expected[i][h]) / scale;
            if (memcmp(&out[h], &expected[i][h], sizeof(double)) == 0) identical++;
            if (!(d <= tolerance)) {
                if (beyond < 10)
                    printf("point %d, output %d: %.17g, RustiCol %.17g (%.3g of the total)\n", i, h, out[h], expected[i][h], d);
                beyond++;
            }
            if (d > worst || d != d) worst = d;
        }
    }
    printf("%s %s: %d points, %d outputs; %d beyond %g of the total (worst %.3g), %d bit-identical to RustiCol%s\n",
           RUNNER_PROCESS, RUNNER_COLOUR, POINTS, POINTS * RUNNER_OUTPUTS, beyond, tolerance, worst, identical,
           beyond ? "  FAILED" : "");
    return beyond ? 1 : 0;
}
"""


def testing_c(prog: Lowered, examples: Examples) -> str:
    points = examples.points

    def rows(f):
        return ",\n".join("    {" + ", ".join(lit(x) for x in f(pt)) + "}" for pt in points)

    momenta = rows(lambda pt: [x for v in pt[0] for x in v])
    expected = rows(lambda pt: pt[1])
    return f"""\
/* The tests of {prog.process} {prog.colour}, generated by PacRunner's compile: run() (runner.c)
 * on the {len(points)} points of examples.json, each output compared with RustiCol's (the CPU's
 * outputs there) within TOLERANCE × |the point's total|.
 * usage: ./testing [TOLERANCE]     (default 1e-14; `make test`)
 * Exits 0 when every output is within the tolerance. The C is bit for bit the Julia runner's;
 * RustiCol's SIMD kernels may round differently in the last bits, hence the tolerance. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "runner.h"

#define POINTS {len(points)}

static const double momenta[POINTS][4 * RUNNER_LEGS] = {{
{momenta}
}};

static const double expected[POINTS][RUNNER_OUTPUTS] = {{
{expected}
}};
""" + TESTING_MAIN


VM_SOURCES = ("uirvm.c", "uirvm.h", "cuda/uirgpu.c", "cuda/uirgpu.h", "cuda/uirgpu_pool.c", "cuda/uirgpu_umami.c",
              "cuda/uirgpu_provider.c", "cuda/pool.cu", "cuda/pool.h")


def makefile(prog: Lowered, testing: bool, interpreter: bool = False) -> str:
    vm = "".join(" " + Path(f).name for f in VM_SOURCES) if interpreter else ""   # copied flat
    t = " testing" if testing else ""
    out = (f"# {prog.process} {prog.colour}, compiled by PacRunner's compile.\n"
           f"#   make         runner.o and pacops.o{', and testing' if testing else ''}\n")
    if testing:
        out += "#   make test    runs testing: the points of examples.json against RustiCol's outputs\n"
    if interpreter:
        out += "#   make gpu     runner_gpu.o (nvcc): run_gpu, every point on the GPU (uirvm.h)\n"
    out += ("#   make clean\n"
            "# CFLAGS may be changed; PACFLAGS may not: the values are bit for bit the Julia runner's only\n"
            "# without contracted multiply-adds (-ffp-contract=off).\n"
            "CC ?= cc\n"
            "CFLAGS ?= -O2\n"
            "PACFLAGS = -std=c99 -ffp-contract=off\n"
            f"all: runner.o pacops.o{t}\n"
            "\n"
            f"runner.o: runner.c runner.h pacops.h{vm}\n"
            "\t$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ runner.c\n"
            "\n"
            "pacops.o: pacops.c pacops.h\n"
            "\t$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ pacops.c\n")
    if interpreter:
        out += ("\n"
                "# the same runner.c on the GPU; NVFLAGS may be changed, NVPACFLAGS may not (bit for bit the CPU's\n"
                "# only without contracted multiply-adds)\n"
                "NVCC ?= nvcc\n"
                "NVFLAGS ?= -O3 -arch=native\n"
                "NVPACFLAGS = --fmad=false\n"
                "gpu: runner_gpu.o\n"
                "\n"
                f"runner_gpu.o: runner.c runner.h pacops.c pacops.h{vm}\n"
                "\t$(NVCC) $(NVFLAGS) $(NVPACFLAGS) -x cu -c -o $@ runner.c\n")
    if testing:
        out += ("\n"
                "testing: testing.c runner.h runner.o pacops.o\n"
                "\t$(CC) $(CFLAGS) $(PACFLAGS) -o $@ testing.c runner.o pacops.o -lm\n"
                "\n"
                "test: testing\n"
                "\t./testing\n")
    out += ("\n"
            "clean:\n"
            f"\trm -f runner.o pacops.o{t}{' runner_gpu.o' if interpreter else ''}\n"
            "\n"
            f".PHONY: all{' test' if testing else ''}{' gpu' if interpreter else ''} clean\n")
    return out


def write_beside(path, prog: Lowered, b: ArenaIndex, examples: Optional[Examples], interpreter: bool) -> None:
    """runner.h, testing.c (with examples) and the Makefile"""
    path = Path(path)
    (path / "runner.h").write_text(runner_h(prog, b), encoding="utf-8")
    if examples is not None:
        (path / "testing.c").write_text(testing_c(prog, examples), encoding="utf-8")
    (path / "Makefile").write_text(makefile(prog, examples is not None, interpreter=interpreter), encoding="utf-8")

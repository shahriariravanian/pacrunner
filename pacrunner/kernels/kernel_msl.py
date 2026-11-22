"""A symjit kernel in the Metal Shading Language, for the Metal runner (c/metal/), pac-julia's
src/kernel_msl.jl: kernel_msl(name, source) writes kernel_c_mixed's kernel (kernel_c.py: the same
steps, the same `need` rule) with df64 (c/metal/df64.h) in place of double, as
`static inline int name(thread df64 *mem)`: the inputs at mem[0 .. states), the outputs (re, im of
each output pair) at mem[states ..), as kernel_c's.

MSL has no goto, so a kernel must be straight-line once its funclets are inlined: the Complexifier's
lowered complex square root and division are funclets whose bodies are straight-line (selects, no
branches); each call is replaced by the funclet's body (inline_funclets), and any other control flow,
or a math-library call, is refused. Every kernel of our test data and of the mg7 processes is of this
kind (2026-11-15: 21 straight-line, 26 with one inlined complex root).

The bit operations act on both halves of a df64 (df64.h): masks and signs have their bits in both, so
the kernels' select, NaN-clearing and sign idioms keep their f64 meaning.

The MSL is pac-julia's byte for byte (its comment names kernel_msl.jl).
"""

from __future__ import annotations

import dataclasses

from .kernel_c import F32, F64, Source, _mir, check_kernel, mixed_body, refuse
from .mir import Branch, BranchIf, Call, Instruction, Label, Mir


def _control(i: Instruction) -> bool:
    return isinstance(i, (Label, Branch, BranchIf, Call))


def inline_funclets(name: str, code) -> list:
    """the code with every funclet call replaced by the funclet's body and the funclets removed (with
    the jump over each); refuses what stays non-straight-line"""
    code = list(code)
    called = {i.label for i in code if isinstance(i, Call)}
    labels = {i.label: k for k, i in enumerate(code) if isinstance(i, Label)}
    bodies = {}
    drop = [False] * len(code)
    for l in sorted(called):
        if l not in labels:
            refuse(f"{name}: a call of {l} (math functions are not supported)")
        s = labels[l]
        r = next((k for k in range(s + 1, len(code)) if isinstance(code[k], Branch) and code[k].label == ".ret"), None)
        if r is None:
            refuse(f"{name}: the funclet {l} does not return")
        body = code[s + 1:r]
        if any(_control(i) for i in body):
            refuse(f"{name}: the funclet {l} is not straight-line")
        bodies[l] = body
        for k in range(s, r + 1):
            drop[k] = True
        # the jump over it: `goto J` just before its label, `J:` just after its return
        if (s > 0 and r < len(code) - 1 and isinstance(code[s - 1], Branch) and isinstance(code[r + 1], Label)
                and code[s - 1].label == code[r + 1].label):
            drop[s - 1] = drop[r + 1] = True
    out = []
    for k, i in enumerate(code):
        if drop[k]:
            continue
        if isinstance(i, Call):
            if i.num_args != 0:
                refuse(f"{name}: a call of {i.label} with arguments")
            out.extend(bodies[i.label])
        else:
            out.append(i)
    if any(_control(i) for i in out):
        refuse(f"{name}: control flow other than straight-line funclets (MSL has no goto)")
    return out


def kernel_msl(name: str, source: Source) -> str:
    """the kernel `name` as an MSL function (df64.h before it); `source`: its Mir, or its saved
    application"""
    mir = _mir(source)
    check_kernel(name, mir)
    mir = dataclasses.replace(mir, code=tuple(inline_funclets(name, mir.code)))
    declarations, body, counts, _, _ = mixed_body(name, mir, "msl")
    out = [f"/* the symjit kernel {name} in mixed precision (kernel_msl.jl): {counts[F32]} f32 and "
           f"{counts[F64]} df64 operations, {counts['conversions']} conversions */",
           f"static inline int {name}(thread df64 *mem) {{"]
    out += declarations
    out += body
    out += ["return 0;", "}"]
    return "".join(line + "\n" for line in out)

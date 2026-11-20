"""A symjit kernel's MIR from symjit's MIR JSON (symjit ≥ 2.29, format "symjit-mir" v1; symjit's
rust/mir/json.rs): `read_mir(sjb bytes)` calls `symjit.mir_json(bytes, "json-complex")`, which reads
the saved application without compiling it and writes its MIR of real operations (a complex model's
complexified, LoadComplex/SaveComplex and ComplexBi kept). This replaces pac-julia's sjb.jl and
complexify.jl.

The instructions are symjit's `mir::Instruction`, variant for variant and field for field (serde's
externally tagged form: {"Bi": {"op": "Times", "dst": {"Gen": 3}, ...}}, "Nop"); registers and
locations are Reg and Loc, the constants doubles (the file has their bit patterns)."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, fields
from typing import Optional, Union


class MirError(ValueError):
    """MIR JSON this reader does not take"""


@dataclass(frozen=True)
class Reg:
    kind: str      # Ret, Temp, Left, Right, Gen, Static
    index: int = 0  # of Gen and Static

    def __repr__(self) -> str:
        return {"Ret": "%$", "Temp": "%t", "Left": "%l", "Right": "%r"}.get(self.kind) or \
            ("%" if self.kind == "Gen" else "σ") + str(self.index)


@dataclass(frozen=True)
class Loc:
    kind: str      # Mem, Stack, Param
    index: int

    def __repr__(self) -> str:
        return f"{self.kind}[{self.index}]"


# ---- the instructions (symjit's mir::Instruction) --------------------------------------------------------

class Instruction:
    pass


@dataclass(frozen=True)
class Nop(Instruction):
    pass


@dataclass(frozen=True)
class End(Instruction):
    pass


@dataclass(frozen=True)
class Uni(Instruction):
    op: str
    dst: Reg
    s1: Reg


@dataclass(frozen=True)
class Bi(Instruction):
    op: str
    dst: Reg
    s1: Reg
    s2: Reg


@dataclass(frozen=True)
class Mov(Instruction):
    dst: Reg
    s1: Reg


@dataclass(frozen=True)
class Load(Instruction):
    dst: Reg
    loc: Loc


@dataclass(frozen=True)
class Save(Instruction):
    src: Reg
    loc: Loc


@dataclass(frozen=True)
class LoadComplex(Instruction):
    xd: Reg
    yd: Reg
    loc: Loc


@dataclass(frozen=True)
class SaveComplex(Instruction):
    xs: Reg
    ys: Reg
    loc: Loc


@dataclass(frozen=True)
class LoadConst(Instruction):
    dst: Reg
    idx: int


@dataclass(frozen=True)
class LoadArgs(Instruction):
    locs: tuple
    complex: bool
    ultra: bool


@dataclass(frozen=True)
class SaveArgs(Instruction):
    num_args: int
    complex: bool
    ultra: bool


@dataclass(frozen=True)
class Call(Instruction):
    label: str
    num_args: int


@dataclass(frozen=True)
class Fused(Instruction):
    op: str
    dst: Reg
    a: Reg
    b: Reg
    c: Reg


@dataclass(frozen=True)
class IfElse(Instruction):
    dst: Reg
    true_val: Reg
    false_val: Reg
    cond: Loc


@dataclass(frozen=True)
class Label(Instruction):
    label: str


@dataclass(frozen=True)
class Branch(Instruction):
    label: str


@dataclass(frozen=True)
class BranchIf(Instruction):
    cond: Reg
    label: str
    is_else: bool


@dataclass(frozen=True)
class LoadMath(Instruction):
    op: str
    dst: Reg
    s1: Reg
    loc: Loc


@dataclass(frozen=True)
class LoadConstMath(Instruction):
    op: str
    dst: Reg
    s1: Reg
    idx: int


@dataclass(frozen=True)
class ComplexBi(Instruction):
    op: str
    xd: Reg
    yd: Reg
    x1: Reg
    y1: Reg
    x2: Reg
    y2: Reg


INSTRUCTIONS = {cls.__name__: cls for cls in (Nop, End, Uni, Bi, Mov, Load, Save, LoadComplex, SaveComplex, LoadConst, LoadArgs,
                                              SaveArgs, Call, Fused, IfElse, Label, Branch, BranchIf, LoadMath, LoadConstMath,
                                              ComplexBi)}


def _reg(x) -> Reg:
    if isinstance(x, str):
        if x not in ("Ret", "Temp", "Left", "Right"):
            raise MirError(f"register {x}")
        return Reg(x)
    ((kind, index),) = x.items()
    if kind not in ("Gen", "Static"):
        raise MirError(f"register {x}")
    return Reg(kind, int(index))


def _loc(x) -> Loc:
    ((kind, index),) = x.items()
    if kind not in ("Mem", "Stack", "Param"):
        raise MirError(f"location {x}")
    return Loc(kind, int(index))


def _value(cls, name: str, v):
    t = cls.__dataclass_fields__[name].type
    if t == "Reg":
        return _reg(v)
    if t == "Loc":
        return _loc(v)
    if name == "locs":
        return tuple(_loc(x) for x in v)
    return v


def instruction(x) -> Instruction:
    """one instruction from its JSON"""
    if isinstance(x, str):
        name, body = x, {}
    else:
        ((name, body),) = x.items()
    cls = INSTRUCTIONS.get(name)
    if cls is None:
        raise MirError(f"instruction {name}")
    names = [f.name for f in fields(cls)]
    if sorted(names) != sorted(body):
        raise MirError(f"{name}: fields {sorted(body)}, this reader takes {sorted(names)}")
    return cls(**{n: _value(cls, n, body[n]) for n in names})


def double(bits: str) -> float:
    """a constant from its bit pattern, "0x" and 16 hex digits"""
    if not (bits.startswith("0x") and len(bits) == 18):
        raise MirError(f"constant {bits}")
    return struct.unpack("<d", struct.pack("<Q", int(bits, 16)))[0]


# ---- the kernel ---------------------------------------------------------------------------------------------

FORMAT = "symjit-mir"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Mir:
    """A kernel's MIR of real operations and what kernel_c needs of its application: the counts, the
    configuration's options (symjit.toml's names: complex, direct_arena, ...) and the real locations."""
    code: tuple[Instruction, ...]
    consts: tuple[float, ...]
    labels: dict
    count_states: int
    count_params: int
    count_obs: int
    count_diffs: int
    options: dict
    reals: tuple[Loc, ...]
    complexified: bool
    symjit_version: str

    def is_complex(self) -> bool:
        """a complex model (its configuration's option: complexified or not)"""
        return bool(self.options.get("complex", False))

    def option(self, name: str) -> bool:
        return bool(self.options.get(name, False))


def parse_mir(text: Union[str, bytes, dict]) -> Mir:
    """a Mir from symjit's MIR JSON (format "symjit-mir" v1)"""
    d = text if isinstance(text, dict) else json.loads(text)
    if d.get("format") != FORMAT or d.get("format_version") != FORMAT_VERSION:
        raise MirError(f"format {d.get('format')} v{d.get('format_version')}, this reader takes {FORMAT} v{FORMAT_VERSION}")
    if d["config"]["options"].get("complex") and not d["complexified"]:
        raise MirError("a complex model's MIR before complexification (dump it with json-complex)")
    return Mir(code=tuple(instruction(x) for x in d["code"]), consts=tuple(double(c) for c in d["consts"]),
               labels=dict(d["labels"]), count_states=int(d["count_states"]), count_params=int(d["count_params"]),
               count_obs=int(d["count_obs"]), count_diffs=int(d["count_diffs"]), options=dict(d["config"]["options"]),
               reals=tuple(_loc(x) for x in d["reals"]), complexified=bool(d["complexified"]),
               symjit_version=str(d["symjit_version"]))


def read_mir(application: Union[bytes, "os.PathLike", str]) -> Mir:
    """the MIR of a saved symjit application (.sjb bytes, or a path), lowered to real operations"""
    try:
        import symjit
    except ImportError as e:
        raise MirError("symjit is not installed (pacrunner needs symjit >= 2.29)") from e
    if not hasattr(symjit, "mir_json"):
        raise MirError(f"symjit {getattr(symjit, '__version__', '?')} has no mir_json (pacrunner needs symjit >= 2.29)")
    return parse_mir(symjit.mir_json(application, "json-complex"))


def listing(mir: Mir) -> str:
    """the code, one instruction a line numbered from 0 (for messages and dumps)"""
    return "".join(f"{k:05d}\t{i}\n" for k, i in enumerate(mir.code))

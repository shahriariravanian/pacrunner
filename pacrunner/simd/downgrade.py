"""downgrade(code) -> bytes: an f64 SIMD kernel of symjit's JIT (AVX2 or AVX-512) turned into the same
kernel in f32, instruction for instruction (plans/F32-KERNEL.md: the mixed kernels are generated in
f64 + double-double and downgraded to f32 + df64 this way).

Why a rewrite works: every packed double instruction symjit's vector generators emit has a packed single
twin of the same length and operand form (vaddpd -> vaddps, vfmadd231pd -> vfmadd231ps, ...); a ymm holds
4 doubles or 8 floats and a zmm 8 or 16, both 32 or 64 bytes, so the stack slots, the states' and
outputs' offsets and the compress-mode slot arithmetic are the same bytes. Only the lanes double: an AVX2
call computes 8 points and an AVX-512 call 16. The integer code (prologue, argument unpacking, calls) is
left as it is.

The rewrite, per instruction (anything else symjit's kernels do not contain, and is refused):
  - VEX (AVX2): the packed double ops of map 0F (pp = 66: vmovupd, vmovapd, vaddpd, vsubpd, vmulpd,
    vdivpd, vsqrtpd, vminpd, vmaxpd, vandpd, vandnpd, vorpd, vxorpd, vcmppd) -> pp = none (ps); the
    scalar double ops (pp = F2) -> pp = F3 (ss); the FMA family (map 0F38, 0x96-0xBF) -> W = 0 (ps);
    vbroadcastsd (map 0F38 0x19) -> vbroadcastss (0x18); vroundpd (map 0F3A 0x09) -> vroundps (0x08).
  - EVEX (AVX-512): the same, with W = 1 -> 0 where the double form has it; vpmovm2q / vpmovq2m (map
    0F38, F3, 0x38 / 0x39, W = 1) -> vpmovm2d / vpmovd2m (W = 0). A broadcast load with a compressed
    8-bit displacement (disp8 x 8 for sd, x 4 for ss) gets its displacement doubled. Embedded broadcasts
    ({1toN}) are refused.
  - Mask (k) and integer instructions: unchanged.
  - The data area (constants and the generators' tables, read RIP-relative): each 8-byte double becomes
    the float of its value in both halves (vbroadcastss reads the low four bytes, a table read as a
    vector its lanes); a NaN pattern (the masks: all ones, 0x7fff...) keeps its high 32 bits in both
    halves. A double that has no float (out of range) is an error.

The data area: symjit puts the constant pool (all of the MIR's constants, in order) first after the code,
then its own tables. Given the pool, its start is the first instruction boundary where the pool's bytes
begin; without it, the lowest address an instruction reads RIP-relative (right when the first constants are
read: checked against the assembler's @data label on 70 kernels, plans/sjb2asm). The kernels call no
functions (simd/elf.py refuses them).

The caller passes the columns as floats: the kernel reads element `idx` of a state at byte 8 x idx,
so an f32 kernel is called with idx / 2 (the points are a multiple of 8 or 16, so idx is even).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Optional


class DowngradeError(ValueError):
    pass


@dataclass
class Ins:
    pc: int
    length: int
    kind: str                     # "legacy", "vex", "evex"
    map: int = 0                  # 1: 0F, 2: 0F38, 3: 0F3A
    opcode: int = 0
    pp: int = 0                   # 0 none, 1: 66, 2: F3, 3: F2
    w: int = 0
    modrm: Optional[int] = None
    rip: Optional[int] = None     # the address a RIP-relative operand reads
    disp8_at: Optional[int] = None
    broadcast: bool = False       # EVEX.b


def _modrm(code: bytes, i: int, end_of_ins_extra: int) -> tuple[int, Optional[int], Optional[int]]:
    """the length of ModRM + SIB + displacement at i; the RIP-relative displacement's offset (relative
    to i) if any; the 8-bit displacement's offset (relative to i) if any"""
    m = code[i]
    mod, rm = m >> 6, m & 7
    n = 1
    rip = disp8 = None
    if mod == 3:
        return n, None, None
    if rm == 4:
        sib = code[i + 1]
        n += 1
        if mod == 0 and (sib & 7) == 5:
            n += 4
    elif mod == 0 and rm == 5:
        rip = n
        n += 4
    if mod == 1:
        disp8 = n
        n += 1
    elif mod == 2:
        n += 4
    return n, rip, disp8


# legacy opcodes: (has ModRM, immediate bytes); 0xB8-0xBF and the jumps handled apart
_LEGACY = {
    0x01: (True, 0), 0x03: (True, 0), 0x09: (True, 0), 0x0B: (True, 0), 0x21: (True, 0), 0x23: (True, 0),
    0x29: (True, 0), 0x2B: (True, 0), 0x31: (True, 0), 0x33: (True, 0), 0x39: (True, 0), 0x3B: (True, 0),
    0x85: (True, 0), 0x89: (True, 0), 0x8B: (True, 0), 0x8D: (True, 0),
    0x81: (True, 4), 0x83: (True, 1), 0xC1: (True, 1), 0xD1: (True, 0), 0xC7: (True, 4), 0xF7: (True, 0),
    0xFF: (True, 0),
    0x90: (False, 0), 0xC3: (False, 0), 0xCC: (False, 0),
}
_LEGACY_0F = {0xB6: (True, 0), 0xB7: (True, 0), 0x1F: (True, 0), 0xAF: (True, 0)}


def _vex_imm(map_: int, opcode: int) -> int:
    if map_ == 3:
        return 1
    if map_ == 1 and opcode in (0xC2, 0xC6, 0x70, 0x71, 0x72, 0x73):
        return 1
    return 0


def decode_one(code: bytes, pc: int) -> Ins:
    i = pc
    rex_w = False
    while code[i] in (0x66, 0xF2, 0xF3):
        i += 1
    if 0x40 <= code[i] <= 0x4F:
        rex_w = bool(code[i] & 8)
        i += 1
    b = code[i]
    if b in (0xC4, 0xC5, 0x62) and i == pc:
        if b == 0xC5:
            b1 = code[i + 1]
            ins = Ins(pc, 0, "vex", map=1, pp=b1 & 3, w=0)
            op_at = i + 2
        elif b == 0xC4:
            b1, b2 = code[i + 1], code[i + 2]
            ins = Ins(pc, 0, "vex", map=b1 & 0x1F, pp=b2 & 3, w=b2 >> 7)
            op_at = i + 3
        else:
            p0, p1, p2 = code[i + 1], code[i + 2], code[i + 3]
            if not p1 & 4:
                raise DowngradeError(f"0x{pc:x}: not an EVEX prefix")
            ins = Ins(pc, 0, "evex", map=p0 & 7, pp=p1 & 3, w=p1 >> 7, broadcast=bool(p2 & 0x10))
            op_at = i + 4
        ins.opcode = code[op_at]
        if ins.map == 1 and ins.opcode == 0x77:            # vzeroupper / vzeroall: no operands
            ins.length = op_at + 1 - pc
            return ins
        ins.modrm = code[op_at + 1]
        n, rip, d8 = _modrm(code, op_at + 1, 0)
        end = op_at + 1 + n + _vex_imm(ins.map, ins.opcode)
        ins.length = end - pc
        if rip is not None:
            disp = struct.unpack_from("<i", code, op_at + 1 + rip)[0]
            ins.rip = end + disp
        if d8 is not None:
            ins.disp8_at = op_at + 1 + d8
        return ins
    ins = Ins(pc, 0, "legacy", opcode=b)
    if 0xB8 <= b <= 0xBF:
        ins.length = i + 1 + (8 if rex_w else 4) - pc
        return ins
    if b in (0xE8, 0xE9):
        ins.length = i + 5 - pc
        return ins
    if b == 0xEB or 0x70 <= b <= 0x7F:
        ins.length = i + 2 - pc
        return ins
    if 0x50 <= b <= 0x5F:
        ins.length = i + 1 - pc
        return ins
    if b == 0x0F:
        b2 = code[i + 1]
        if 0x80 <= b2 <= 0x8F:
            ins.length = i + 6 - pc
            return ins
        if b2 not in _LEGACY_0F:
            raise DowngradeError(f"0x{pc:x}: unknown instruction {code[pc:pc + 8].hex()}")
        has, imm = _LEGACY_0F[b2]
        ins.opcode = 0x0F00 | b2
        n, rip, _ = _modrm(code, i + 2, 0)
        end = i + 2 + n + imm
    else:
        if b not in _LEGACY:
            raise DowngradeError(f"0x{pc:x}: unknown instruction {code[pc:pc + 8].hex()}")
        has, imm = _LEGACY[b]
        if not has:
            ins.length = i + 1 + imm - pc
            return ins
        if b == 0xF7:
            imm = 4 if (code[i + 1] >> 3) & 7 == 0 else 0
        n, rip, _ = _modrm(code, i + 1, 0)
        end = i + 1 + n + imm
    ins.length = end - pc
    if rip is not None:
        raise DowngradeError(f"0x{pc:x}: an integer instruction reads RIP-relative data")
    return ins


def decode(code: bytes, end: int) -> list[Ins]:
    out, pc = [], 0
    while pc < end:
        ins = decode_one(code, pc)
        out.append(ins)
        pc += ins.length
    if pc != end:
        raise DowngradeError(f"the code does not end at the data area (0x{pc:x}, data at 0x{end:x})")
    return out


def data_start(code: bytes, consts: Optional[list] = None) -> int:
    """Where the data area starts. With the kernel's constant pool (`consts`: the MIR's constants in
    order, as "0x..." bit strings; the JIT puts them first in the data area, every one of them, used or
    not): the first instruction boundary, decoding from the start, where the pool's bytes begin (not
    past the lowest address the code reads).
    Without: the lowest address an instruction reads RIP-relative (wrong if the first constants are
    never read)."""
    if consts:
        pool = struct.pack(f"<{len(consts)}Q", *(int(c, 16) for c in consts))
        pc, lowest = 0, len(code)
        while pc <= min(lowest, len(code) - len(pool)):
            if code[pc:pc + len(pool)] == pool:      # an instruction boundary where the pool begins
                return pc
            ins = decode_one(code, pc)
            if ins.rip is not None:
                lowest = min(lowest, ins.rip)
            pc += ins.length
        raise DowngradeError("the constant pool is not in the code's data area")
    pc, lowest = 0, len(code)
    while pc < min(lowest, len(code)):
        ins = decode_one(code, pc)
        if ins.rip is not None:
            if ins.rip <= ins.pc:
                raise DowngradeError(f"0x{pc:x}: a RIP-relative read backwards")
            lowest = min(lowest, ins.rip)
        pc += ins.length
    return lowest


# the packed double ops of map 0F with a packed single twin (pp 66 -> none)
_PACKED_0F = {0x10, 0x11, 0x28, 0x29, 0x51, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5C, 0x5D, 0x5E, 0x5F, 0xC2}
# the scalar double ops of map 0F (pp F2 -> F3)
_SCALAR_0F = {0x10, 0x11, 0x51, 0x58, 0x59, 0x5C, 0x5D, 0x5E, 0x5F, 0xC2}
_K_OPS = {0x41, 0x42, 0x44, 0x45, 0x46, 0x47, 0x4A, 0x4B, 0x90, 0x91, 0x92, 0x93, 0x98, 0x99}


def _set_vex(code: bytearray, ins: Ins, pp: Optional[int] = None, w: Optional[int] = None,
             opcode: Optional[int] = None) -> None:
    pc = ins.pc
    if ins.kind == "vex":
        if code[pc] == 0xC5:
            if w:
                raise DowngradeError(f"0x{pc:x}: a two-byte VEX cannot hold W")
            if pp is not None:
                code[pc + 1] = (code[pc + 1] & ~3) | pp
            op_at = pc + 2
        else:
            b2 = code[pc + 2]
            if pp is not None:
                b2 = (b2 & ~3) | pp
            if w is not None:
                b2 = (b2 & 0x7F) | (w << 7)
            code[pc + 2] = b2
            op_at = pc + 3
    else:
        p1 = code[pc + 2]
        if pp is not None:
            p1 = (p1 & ~3) | pp
        if w is not None:
            p1 = (p1 & 0x7F) | (w << 7)
        code[pc + 2] = p1
        op_at = pc + 4
    if opcode is not None:
        code[op_at] = opcode


def _rewrite(code: bytearray, ins: Ins) -> str:
    """rewrites one instruction in place; returns what it did ("ps", "ss", "kept", ...)"""
    if ins.kind == "legacy" or ins.modrm is None:
        return "kept"
    evex = ins.kind == "evex"
    if evex and ins.broadcast and (ins.modrm >> 6) != 3:
        raise DowngradeError(f"0x{ins.pc:x}: an embedded broadcast")
    m, op, pp = ins.map, ins.opcode, ins.pp
    if m == 1 and pp == 0 and op in _K_OPS and not evex:
        return "kept"                                      # mask register ops (knotw, kmovw, ...)
    if m == 1 and pp == 1 and op in _PACKED_0F:
        _set_vex(code, ins, pp=0, w=0 if evex else None)
        return "ps"
    if m == 1 and pp == 3 and op in _SCALAR_0F:
        _set_vex(code, ins, pp=2, w=0 if evex else None)
        return "ss"
    if m == 2 and pp == 1 and 0x96 <= op <= 0xBF:          # FMA: W selects the double form
        if ins.w != 1:
            raise DowngradeError(f"0x{ins.pc:x}: a single-precision FMA in an f64 kernel")
        _set_vex(code, ins, w=0)
        return "ps"
    if m == 2 and pp == 1 and op == 0x19:                  # vbroadcastsd -> vbroadcastss
        _set_vex(code, ins, w=0, opcode=0x18)
        if evex and ins.disp8_at is not None:              # disp8 x 8 -> x 4
            d = struct.unpack_from("<b", code, ins.disp8_at)[0] * 2
            if not -128 <= d <= 127:
                raise DowngradeError(f"0x{ins.pc:x}: a broadcast displacement out of disp8 range")
            struct.pack_into("<b", code, ins.disp8_at, d)
        return "ps"
    if m == 3 and pp == 1 and op == 0x09:                  # vroundpd -> vroundps
        _set_vex(code, ins, opcode=0x08)
        return "ps"
    if evex and m == 2 and pp == 2 and op in (0x38, 0x39) and ins.w == 1:   # vpmovm2q / vpmovq2m
        _set_vex(code, ins, w=0)
        return "ps"
    raise DowngradeError(f"0x{ins.pc:x}: no f32 rewrite for {ins.kind} map {m} pp {pp} opcode 0x{op:02x} "
                         f"(W {ins.w})")


def _float_bits(q: int) -> int:
    """an 8-byte double as two copies of its float (a NaN pattern: of its high 32 bits)"""
    exp = (q >> 52) & 0x7FF
    if exp == 0x7FF and q & ((1 << 52) - 1):
        h = q >> 32
    else:
        x = struct.unpack("<d", struct.pack("<Q", q))[0]
        if math.isfinite(x) and abs(x) > 3.4028234663852886e38:
            raise DowngradeError(f"the constant {x!r} has no float")
        h = struct.unpack("<I", struct.pack("<f", x))[0]
    return h | (h << 32)


def rewrite_code(code: bytes, end: Optional[int] = None, stats: Optional[dict] = None) -> bytearray:
    """the instructions of code[:end] (default: all of it) rewritten to f32; the bytes after `end` as
    they are"""
    end = len(code) if end is None else end
    out = bytearray(code)
    for ins in decode(code, end):
        if ins.rip is not None and ins.rip < end:
            raise DowngradeError(f"0x{ins.pc:x}: a RIP-relative read of the code")
        what = _rewrite(out, ins)
        if stats is not None:
            stats[what] = stats.get(what, 0) + 1
    return out


def downgrade(code: bytes, stats: Optional[dict] = None, consts: Optional[list] = None) -> bytes:
    """the f32 twin of an f64 SIMD kernel's machine code (`dump(.., "simd")`); `consts`: the kernel's
    constant pool (its MIR's, data_start); `stats` collects the counts of what was rewritten"""
    start = data_start(code, consts)
    if start % 8 or len(code) % 8:
        raise DowngradeError(f"the data area (0x{start:x}..0x{len(code):x}) is not in 8-byte units")
    out = rewrite_code(code, start, stats)
    for at in range(start, len(code), 8):
        struct.pack_into("<Q", out, at, _float_bits(struct.unpack_from("<Q", code, at)[0]))
    if stats is not None:
        stats["data quads"] = (len(code) - start) // 8
    return bytes(out)

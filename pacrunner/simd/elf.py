"""The SIMD kernels as object files, without symjit-utils: symjit's own machine code (the JIT's SIMD
kernel, `dump(path, "simd")` or its f32 twin `dump(path, "simd-f32")`) wrapped in a minimal ELF64
relocatable object for x86-64 (elf_object), or for aarch64 in an assembler file that the C compiler
assembles (asm_source: ELF on Linux, Mach-O on macOS), and its C header.

This works because a kernel that calls no math function needs no relocation: the JIT's x86-64 code
loads its constants RIP-relative from its own buffer, and its jumps and compress-mode calls are
relative, so the code runs wherever the linker puts it (the bytes are sjb2obj --simd's, but for an
epilogue detail of sjb2obj's copied generators). A kernel calling a math function embeds the
function's address, which an object would have to relocate: refused (no_relocations).

The object: .text (the code, aligned to 4 KB as the JIT's buffer is), .note.GNU-stack (no executable
stack), .symtab with one global function, the kernel, at offset 0, .strtab and .shstrtab.

aarch64: the JIT's code reaches its constants with adrp (the 4 KB page of the instruction) + add, also
position-independent as long as the code starts on a 4 KB boundary, as in the JIT's buffer: the
assembler file aligns its section to 4 KB and includes the code's bytes (.incbin).
"""
from __future__ import annotations

import re
import struct

SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB = 1, 2, 3
SHF_ALLOC, SHF_EXECINSTR = 0x2, 0x4
STB_GLOBAL, STT_FUNC = 1, 2
EM_X86_64, ET_REL = 62, 1
STACK_OVERHEAD = 1024   # bytes beyond the frame's slots: saved registers, alignment, return addresses


class ObjectError(ValueError):
    pass


def no_relocations(code) -> None:
    """refuses a MIR (the instruction list) whose machine code would need relocations: calls of
    functions (math functions, defined functions, applets: num_args > 0)"""
    for ins in code:
        if isinstance(ins, dict) and "Call" in ins and ins["Call"]["num_args"] != 0:
            raise ObjectError(f"the kernel calls {ins['Call']['label']}: its code would need a relocation")


def stack_bytes(app: dict, lanes: int) -> int:
    """the bytes of the calling thread's stack one call uses: the indirect mode's frame of the states and
    outputs, the stack slots up to the largest the code names, each a vector of `lanes` doubles, and
    STACK_OVERHEAD"""
    text = __import__("json").dumps(app["mir"]["code"])
    slots = max([-1, *(int(x) for x in re.findall(r'"Stack": (\d+)', text))]) + 1
    slots = max(slots, int(app["program"]["builder"]["count_stack"]))
    p = app["program"]
    return (p["count_states"] + p["count_obs"] + slots) * 8 * lanes + STACK_OVERHEAD


def elf_object(symbol: str, code: bytes) -> bytes:
    """an x86-64 ELF64 relocatable object: `code` in .text, the global function `symbol` at its start"""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ObjectError(f"{symbol!r} is not a C identifier")
    shstr = b"\0.text\0.note.GNU-stack\0.symtab\0.strtab\0.shstrtab\0"
    name = {n: shstr.index(n.encode() + b"\0") for n in (".text", ".note.GNU-stack", ".symtab", ".strtab", ".shstrtab")}
    strtab = b"\0" + symbol.encode() + b"\0"
    # symbols: the null one, the kernel (global function in section 1)
    sym = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)
    sym += struct.pack("<IBBHQQ", 1, (STB_GLOBAL << 4) | STT_FUNC, 0, 1, 0, len(code))

    out = bytearray(64)                      # the ELF header, written last

    def place(data: bytes, align: int) -> int:
        out.extend(bytes(-len(out) % align))
        at = len(out)
        out.extend(data)
        return at

    text_at = place(code, 4096)
    symtab_at = place(sym, 8)
    strtab_at = place(strtab, 1)
    shstr_at = place(shstr, 1)
    out.extend(bytes(-len(out) % 8))
    shoff = len(out)

    def sh(nm, ty, flags, off, size, link=0, info=0, align=1, entsize=0):
        return struct.pack("<IIQQQQIIQQ", nm, ty, flags, 0, off, size, link, info, align, entsize)

    out += sh(0, 0, 0, 0, 0)                                                            # 0: null
    out += sh(name[".text"], SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_at, len(code), align=4096)   # 1
    out += sh(name[".note.GNU-stack"], SHT_PROGBITS, 0, shoff, 0)                       # 2
    out += sh(name[".symtab"], SHT_SYMTAB, 0, symtab_at, len(sym), link=4, info=1, align=8, entsize=24)  # 3
    out += sh(name[".strtab"], SHT_STRTAB, 0, strtab_at, len(strtab))                  # 4
    out += sh(name[".shstrtab"], SHT_STRTAB, 0, shstr_at, len(shstr))                  # 5
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)   # 64-bit, little-endian, version 1, System V
    out[:64] = ident + struct.pack("<HHIQQQIHHHHHH", ET_REL, EM_X86_64, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 6, 5)
    return bytes(out)


def asm_source(symbol: str, binary: str) -> str:
    """an assembler file (cc -c x.S) defining the hidden function `symbol` as the bytes of the file
    `binary` (an absolute path), its section aligned to 4 KB: ELF (Linux) or Mach-O (macOS)"""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ObjectError(f"{symbol!r} is not a C identifier")
    if '"' in binary or "\n" in binary:
        raise ObjectError(f"{binary!r}: a path the assembler cannot quote")
    return f"""\
/* {symbol}: symjit's machine code ({binary}), as PacRunner's simd/elf.py includes it */
#if defined(__APPLE__)
    .section __TEXT,__text,regular,pure_instructions
    .p2align 12
    .private_extern _{symbol}
    .globl _{symbol}
_{symbol}:
    .incbin "{binary}"
#else
    .section .text.{symbol},"ax",%progbits
    .p2align 12
    .hidden {symbol}
    .globl {symbol}
    .type {symbol}, %function
{symbol}:
    .incbin "{binary}"
    .size {symbol}, . - {symbol}
    .section .note.GNU-stack,"",%progbits
#endif
"""


def write_object(folder, stem: str, symbol: str, code: bytes, isa: str) -> str:
    """the kernel `code` as `folder/stem.o` (x86-64: elf_object) or `folder/stem.S` (+ `stem.bin`; aarch64:
    asm_source, assembled with the library); returns the file's name"""
    from pathlib import Path
    folder = Path(folder).resolve()
    if isa == "neon":
        (folder / f"{stem}.bin").write_bytes(code)
        (folder / f"{stem}.S").write_text(asm_source(symbol, str(folder / f"{stem}.bin")))
        return f"{stem}.S"
    (folder / f"{stem}.o").write_bytes(elf_object(symbol, code))
    return f"{stem}.o"


def header(symbol: str, lanes: int, count_states: int, count_obs: int, stack: int, what: str) -> str:
    """the C header of a kernel object"""
    up = symbol.upper()
    kind = "symjit's scalar kernel" if lanes == 1 else "symjit's SIMD kernel"
    shape = "A scalar kernel (1 lane)" if lanes == 1 else f"A SIMD kernel of {lanes} lanes"
    return f"""\
/* {symbol}: {what}, as an object by PacRunner ({kind}, simd/elf.py).

   {shape}, indirect mode: `states` points to {up}_COUNT_STATES + {up}_COUNT_OBS
   arrays (symjit_slice: the states, then the outputs); {symbol}(NULL, states, i, NULL) reads elements
   i .. i + {up}_LANES - 1 of each state array and writes those of each output array (every array holds
   a multiple of {up}_LANES elements). It returns 0. A call uses up to {up}_STACK_BYTES bytes of the
   calling thread's stack. It calls no library functions. */
#ifndef {up}_H
#define {up}_H

#include <stddef.h>

#define {up}_COUNT_STATES {count_states}
#define {up}_COUNT_OBS {count_obs}
#define {up}_LANES {lanes}
#define {up}_STACK_BYTES {stack}

#ifndef SYMJIT_SLICE_DEFINED
#define SYMJIT_SLICE_DEFINED
typedef struct {{
    double *ptr;
    size_t len;
}} symjit_slice;
#endif

#ifdef __cplusplus
extern "C" {{
#endif
int {symbol}(double *mem, const symjit_slice *states, size_t idx, const double *params);
#ifdef __cplusplus
}}
#endif

#endif
"""

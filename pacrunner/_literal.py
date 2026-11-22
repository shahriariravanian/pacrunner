"""Doubles spelled as Julia's `repr` spells them, so that the C pacrunner writes is pac-julia's byte for
byte: the shortest digits that read back exactly (Python's repr has the same digits), in fixed
notation when the decimal point falls within −4 < point ≤ 6 of the first digit and in Julia's
scientific notation otherwise (1.0e-5, 1.0e6, 1.234567e6, 5.0e-324; Python: 1e-05, 1000000.0).
Checked against Julia 1.13 on 40,388 values (random bit patterns, rounded decimals, powers of ten).
Floats (julia_repr_f32) likewise: the shortest digits of the float32, the same layout, with "f" for
"e" and "f0" after a fixed notation (0.5f0, 1.0f-5, 1.0f6); checked on 24,355 values
(test/data/julia_repr_f32.txt)."""

import numpy as np


def julia_repr(x: float) -> str:
    """a finite double as Julia's repr writes it"""
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"julia_repr: {x} is not finite")
    return _layout(repr(x), "e", "")


def julia_repr_f32(x) -> str:
    """a finite float32 (a numpy float32, or a value rounded to one) as Julia's repr writes it"""
    x = np.float32(x)
    if not np.isfinite(x):
        raise ValueError(f"julia_repr_f32: {x} is not finite")
    # numpy's shortest digits of the float32 ("0.5", "1e-05", "-0.0")
    return _layout(np.format_float_positional(x, unique=True, trim="0") if 1e-4 <= abs(x) < 1e16 or x == 0
                   else np.format_float_scientific(x, unique=True, trim="0"), "f", "f0")


def _layout(text: str, e: str, fixed_suffix: str) -> str:
    """Julia's layout of the shortest digits in `text` (Python's or numpy's spelling of them): fixed
    notation for −4 < point ≤ 6 (then `fixed_suffix`), else d.ddd + e + the exponent"""
    sign = "-" if text.startswith("-") else ""
    mantissa, _, exponent = text.lstrip("-").partition("e")
    whole, _, fraction = mantissa.partition(".")
    padded = whole + fraction
    digits = padded.lstrip("0")
    if not digits:
        return sign + "0.0" + fixed_suffix
    point = len(whole) + (int(exponent) if exponent else 0) - (len(padded) - len(digits))
    digits = digits.rstrip("0")
    if -4 < point <= 6:
        if point <= 0:
            s = "0." + "0" * -point + digits
        elif point >= len(digits):
            s = digits + "0" * (point - len(digits)) + ".0"
        else:
            s = digits[:point] + "." + digits[point:]
        s += fixed_suffix
    else:
        s = digits[0] + "." + (digits[1:] or "0") + e + str(point - 1)
    return sign + s

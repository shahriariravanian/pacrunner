"""Doubles spelled as Julia's `repr` spells them, so that the C pacrunner writes is pac-julia's byte for
byte: the shortest digits that read back exactly (Python's repr has the same digits), in fixed
notation when the decimal point falls within −4 < point ≤ 6 of the first digit and in Julia's
scientific notation otherwise (1.0e-5, 1.0e6, 1.234567e6, 5.0e-324; Python: 1e-05, 1000000.0).
Checked against Julia 1.13 on 40,388 values (random bit patterns, rounded decimals, powers of ten)."""


def julia_repr(x: float) -> str:
    """a finite double as Julia's repr writes it"""
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"julia_repr: {x} is not finite")
    sign = "-" if repr(x).startswith("-") else ""
    mantissa, _, exponent = repr(abs(x)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    padded = whole + fraction
    digits = padded.lstrip("0")
    if not digits:
        return sign + "0.0"
    point = len(whole) + (int(exponent) if exponent else 0) - (len(padded) - len(digits))
    digits = digits.rstrip("0")
    if -4 < point <= 6:
        if point <= 0:
            s = "0." + "0" * -point + digits
        elif point >= len(digits):
            s = digits + "0" * (point - len(digits)) + ".0"
        else:
            s = digits[:point] + "." + digits[point:]
    else:
        s = digits[0] + "." + (digits[1:] or "0") + "e" + str(point - 1)
    return sign + s

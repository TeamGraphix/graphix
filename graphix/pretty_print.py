"""Pretty-printing utilities."""

from __future__ import annotations

import enum
import math
import string
from enum import Enum
from fractions import Fraction


class OutputFormat(Enum):
    """Enumeration of the output format for pretty-printing."""

    ASCII = enum.auto()
    LaTeX = enum.auto()
    Unicode = enum.auto()


def angle_to_str(angle: float, output: OutputFormat, max_denominator: int = 1000) -> str:
    r"""
    Return a string representation of an angle given in units of π.

    - If the angle is a "simple" fraction of π (within the given max_denominator and a small tolerance),
      it returns a fractional string, e.g. "π/2", "2π", or "-3π/4".
    - Otherwise, it returns the angle in radians (angle * π) formatted to two decimal places.

    Parameters
    ----------
    angle : float
        The angle in multiples of π (e.g., 0.5 means π/2).
    output : OutputFormat
        Desired formatting style: Unicode (π symbol), LaTeX (\pi), or ASCII ("pi").
    max_denominator : int, optional
        Maximum denominator for detecting a simple fraction (default: 1000).

    Returns
    -------
    str
        The formatted angle.
    """
    frac = Fraction(angle).limit_denominator(max_denominator)

    if not math.isclose(angle, float(frac)):
        rad = angle * math.pi

        return f"{rad:.2f}"

    num, den = frac.numerator, frac.denominator
    sign = "-" if num < 0 else ""
    num = abs(num)

    if output == OutputFormat.LaTeX:
        pi = r"\pi"

        def mkfrac(num: str, den: str) -> str:
            return rf"\frac{{{num}}}{{{den}}}"
    else:
        pi = "π" if output == OutputFormat.Unicode else "pi"

        def mkfrac(num: str, den: str) -> str:
            return f"{num}/{den}"

    if den == 1:
        if num == 0:
            return "0"
        if num == 1:
            return f"{sign}{pi}"
        return f"{sign}{num}{pi}"

    den_str = f"{den}"
    num_str = pi if num == 1 else f"{num}{pi}"
    return f"{sign}{mkfrac(num_str, den_str)}"


def domain_to_str(domain: set[int]) -> str:
    """Return the string representation of a domain."""
    return f"{{{','.join(str(node) for node in domain)}}}"


SUBSCRIPTS = str.maketrans(string.digits, "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPTS = str.maketrans(string.digits, "⁰¹²³⁴⁵⁶⁷⁸⁹")

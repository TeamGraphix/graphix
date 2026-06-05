"""Pretty-printing utilities."""

from __future__ import annotations

import cmath
import enum
import math
import string
from enum import Enum
from fractions import Fraction
from math import pi
from typing import TYPE_CHECKING, Literal, SupportsComplex, SupportsFloat

# `assert_never` introduced in Python 3.11
from typing_extensions import assert_never

from graphix import command
from graphix.fundamentals import (
    AbstractMeasurement,
    Axis,
    Plane,
    Sign,
    angle_to_rad,
    rad_to_angle,
)
from graphix.measurements import BlochMeasurement, PauliMeasurement
from graphix.parameter import AffineExpression

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphix.command import Node
    from graphix.flow.core import PauliFlow, XZCorrections
    from graphix.fundamentals import Angle
    from graphix.pattern import Pattern
    from graphix.sim.density_matrix import DensityMatrix
    from graphix.sim.statevec import Statevec

_ENCODING = Literal["LSB", "MSB"]


class OutputFormat(Enum):
    """Enumeration of the output format for pretty-printing."""

    ASCII = enum.auto()
    LaTeX = enum.auto()
    Unicode = enum.auto()


def _validate_output_format(output: OutputFormat) -> None:
    assert output in (OutputFormat.ASCII, OutputFormat.LaTeX, OutputFormat.Unicode)


def angle_to_str(
    angle: Angle,
    output: OutputFormat,
    max_denominator: int = 1000,
    multiplication_sign: bool = False,
    abs_tol: float = 0.0,
) -> str:
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
    multiplication_sign : bool
        Optional (default: ``False``).
        If ``True``, the multiplication sign  is made explicit between the
        numerator and π:
        ``2×π`` in Unicode, ``2 \times \pi`` in LaTeX, and ``2*pi`` in ASCII.
        If ``False``, the multiplication sign is implicit:
        ``2π`` in Unicode, ``2\pi`` in LaTeX, ``2pi`` in ASCII.
    abs_tol : float, optional
        Absolute tolerance passed to :func:`math.isclose` (default: ``0.0``).

    Returns
    -------
    str
        The formatted angle.
    """
    _validate_output_format(output)
    frac = Fraction(angle).limit_denominator(max_denominator)

    if not math.isclose(angle, float(frac), abs_tol=abs_tol):
        return f"{angle_to_rad(angle):.2f}"

    num, den = frac.numerator, frac.denominator
    sign = "-" if num < 0 else ""
    num = abs(num)

    if output == OutputFormat.LaTeX:
        pi = r"\pi"

        def mkfrac(num: str, den: str) -> str:
            return rf"\frac{{{num}}}{{{den}}}"

        mul = r" \times "
    else:
        pi = "π" if output == OutputFormat.Unicode else "pi"

        def mkfrac(num: str, den: str) -> str:
            return f"{num}/{den}"

        mul = "×" if output == OutputFormat.Unicode else "*"

    if not multiplication_sign:
        mul = ""

    if den == 1:
        match num:
            case 0:
                return "0"
            case 1:
                return f"{sign}{pi}"
            case _:
                return f"{sign}{num}{mul}{pi}"

    den_str = f"{den}"
    num_str = pi if num == 1 else f"{num}{mul}{pi}"
    return f"{sign}{mkfrac(num_str, den_str)}"


_MAX_RADICAND = 10


def _format_helpers(
    output: OutputFormat,
) -> tuple[Callable[[str, str], str], Callable[[int], str]]:
    if output == OutputFormat.LaTeX:

        def mkfrac(num: str, den: str) -> str:
            return rf"\frac{{{num}}}{{{den}}}"

        def sqrt(n: int) -> str:
            return rf"\sqrt{{{n}}}"

    elif output == OutputFormat.Unicode:

        def mkfrac(num: str, den: str) -> str:
            return f"{num}/{den}"

        def sqrt(n: int) -> str:
            return f"√{n}"

    else:

        def mkfrac(num: str, den: str) -> str:
            return f"{num}/{den}"

        def sqrt(n: int) -> str:
            return f"sqrt({n})"

    return mkfrac, sqrt


def _real_scalar_to_str(
    x: float,
    output: OutputFormat,
    *,
    max_denominator: int = 1000,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-8,
    max_radicand: int = _MAX_RADICAND,
) -> str | None:
    mkfrac, sqrt = _format_helpers(output)

    frac = Fraction(x).limit_denominator(max_denominator)
    if math.isclose(x, float(frac), rel_tol=rel_tol, abs_tol=abs_tol):
        num, den = frac.numerator, frac.denominator
        sign = "-" if num < 0 else ""
        num = abs(num)
        if den == 1:
            return f"{sign}{num}"
        return f"{sign}{mkfrac(str(num), str(den))}"

    for n in range(1, max_radicand + 1):
        root = math.sqrt(n)
        for d in range(1, max_denominator + 1):
            val = root / d
            if math.isclose(x, val, rel_tol=rel_tol, abs_tol=abs_tol):
                num_str = sqrt(n) if n != 1 else "1"
                return num_str if d == 1 else mkfrac(num_str, str(d))
            if math.isclose(x, -val, rel_tol=rel_tol, abs_tol=abs_tol):
                num_str = sqrt(n) if n != 1 else "1"
                formatted = num_str if d == 1 else mkfrac(num_str, str(d))
                return f"-{formatted}"

    return None


def _scalar_or_decimal(
    x: float,
    output: OutputFormat,
    *,
    max_denominator: int,
    rel_tol: float,
    abs_tol: float,
) -> str:
    result = _real_scalar_to_str(x, output, max_denominator=max_denominator, rel_tol=rel_tol, abs_tol=abs_tol)
    if result:
        return result
    return f"{x:g}"


def _imag_scalar_to_str(
    x: float,
    output: OutputFormat,
    *,
    max_denominator: int,
    rel_tol: float,
    abs_tol: float,
) -> str:
    i = r"\mathrm{i}" if output == OutputFormat.LaTeX else "i"
    x = abs(x)

    if math.isclose(x, 1.0, rel_tol=rel_tol, abs_tol=abs_tol):
        return i

    _, sqrt = _format_helpers(output)

    frac = Fraction(x).limit_denominator(max_denominator)
    if math.isclose(x, float(frac), rel_tol=rel_tol, abs_tol=abs_tol):
        num, den = abs(frac.numerator), frac.denominator
        if den == 1:
            return f"{num}{i}"
        if num == 1:
            return f"{i}/{den}" if output != OutputFormat.LaTeX else rf"\frac{{{i}}}{{{den}}}"
        if output == OutputFormat.LaTeX:
            return rf"\frac{{{num}{i}}}{{{den}}}"
        return f"{num}{i}/{den}"

    for n in range(1, _MAX_RADICAND + 1):
        root = math.sqrt(n)
        for d in range(1, max_denominator + 1):
            val = root / d
            if math.isclose(x, val, rel_tol=rel_tol, abs_tol=abs_tol):
                num_str = sqrt(n) if n != 1 else "1"
                if d == 1:
                    return i if num_str == "1" else f"{num_str}{i}"
                if num_str == "1":
                    return f"{i}/{d}" if output != OutputFormat.LaTeX else rf"\frac{{{i}}}{{{d}}}"
                if output == OutputFormat.LaTeX:
                    return rf"\frac{{{num_str}{i}}}{{{d}}}"
                return f"{num_str}{i}/{d}"

    return f"{x:g}{i}"


def _exp_i_to_str(angle_str: str, output: OutputFormat) -> str:
    if output == OutputFormat.LaTeX:
        return rf"\mathrm{{e}}^{{\mathrm{{i}}{angle_str}}}"
    if output == OutputFormat.Unicode:
        return f"e^(i{angle_str})"
    return f"e^(i*{angle_str})"


def _cartesian_to_str(
    z: complex,
    output: OutputFormat,
    *,
    max_denominator: int,
    rel_tol: float,
    abs_tol: float,
) -> str:
    real_zero = math.isclose(z.real, 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
    imag_zero = math.isclose(z.imag, 0.0, rel_tol=rel_tol, abs_tol=abs_tol)

    if imag_zero:
        return _scalar_or_decimal(
            z.real,
            output,
            max_denominator=max_denominator,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    imag_part = _imag_scalar_to_str(
        z.imag,
        output,
        max_denominator=max_denominator,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    if real_zero:
        return f"-{imag_part}" if z.imag < 0 else imag_part

    real_str = _scalar_or_decimal(
        z.real,
        output,
        max_denominator=max_denominator,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    if z.imag >= 0:
        return f"{real_str} + {imag_part}"
    return f"{real_str} - {imag_part}"


def complex_to_str(
    z: complex | SupportsComplex,
    output: OutputFormat,
    *,
    max_denominator: int = 1000,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-8,
) -> str:
    r"""
    Return a string representation of a complex number.

    Common values are rendered symbolically:

    - rational reals such as ``0.25`` as ``1/4``,
    - radical rationals such as ``0.70710678`` as ``√2/2``,
    - unit-modulus values such as ``0.5 + 0.8660254j`` as ``e^(iπ/3)``.

    Parameters
    ----------
    z : complex
        The complex number to format.
    output : OutputFormat
        Desired formatting style: Unicode, LaTeX, or ASCII.
    max_denominator : int, optional
        Maximum denominator for detecting simple fractions and radical forms (default: 1000).
    rel_tol : float, optional
        Relative tolerance passed to :func:`math.isclose` (default: ``1e-9``).
    abs_tol : float, optional
        Absolute tolerance passed to :func:`math.isclose` (default: ``1e-8``).

    Returns
    -------
    str
        The formatted complex number.
    """
    _validate_output_format(output)
    z = complex(z)

    if math.isclose(z.real, 0.0, rel_tol=rel_tol, abs_tol=abs_tol) and math.isclose(
        z.imag, 0.0, rel_tol=rel_tol, abs_tol=abs_tol
    ):
        return "0"

    if math.isclose(abs(z), 1.0, rel_tol=rel_tol, abs_tol=abs_tol):
        if math.isclose(z.real, 1.0, rel_tol=rel_tol, abs_tol=abs_tol) and math.isclose(
            z.imag, 0.0, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return "1"
        if math.isclose(z.real, -1.0, rel_tol=rel_tol, abs_tol=abs_tol) and math.isclose(
            z.imag, 0.0, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return "-1"
        if math.isclose(z.real, 0.0, rel_tol=rel_tol, abs_tol=abs_tol) and math.isclose(
            z.imag, 1.0, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return r"\mathrm{i}" if output == OutputFormat.LaTeX else "i"
        if math.isclose(z.real, 0.0, rel_tol=rel_tol, abs_tol=abs_tol) and math.isclose(
            z.imag, -1.0, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return r"-\mathrm{i}" if output == OutputFormat.LaTeX else "-i"

        angle = cmath.phase(z) / pi
        frac = Fraction(angle).limit_denominator(max_denominator)
        if math.isclose(angle, float(frac), rel_tol=rel_tol, abs_tol=abs_tol):
            angle_str = angle_to_str(angle, output, max_denominator=max_denominator, abs_tol=abs_tol)
            return _exp_i_to_str(angle_str, output)

    return _cartesian_to_str(z, output, max_denominator=max_denominator, rel_tol=rel_tol, abs_tol=abs_tol)


def _ket_to_str(bits: str, output: OutputFormat) -> str:
    if output == OutputFormat.LaTeX:
        return rf"\ket{{{bits}}}"
    if output == OutputFormat.Unicode:
        return f"|{bits}⟩"
    return f"|{bits}>"


def _format_statevec_term(
    amp: complex,
    ket: str,
    output: OutputFormat,
    *,
    max_denominator: int,
    rel_tol: float,
    abs_tol: float,
) -> tuple[str, str]:
    coeff = complex_to_str(amp, output, max_denominator=max_denominator, rel_tol=rel_tol, abs_tol=abs_tol)
    ket_str = _ket_to_str(ket, output)
    if coeff == "1":
        return "+", ket_str
    if coeff == "-1":
        return "-", ket_str
    if coeff.startswith("-"):
        return "-", f"{coeff[1:]}{ket_str}"
    return "+", f"{coeff}{ket_str}"


def _join_statevec_terms(terms: list[tuple[str, str]]) -> str:
    if not terms:
        return "0"
    sign, body = terms[0]
    result = f"-{body}" if sign == "-" else body
    for sign, body in terms[1:]:
        result += f" - {body}" if sign == "-" else f" + {body}"
    return result


def statevec_to_str(
    statevec: Statevec,
    output: OutputFormat,
    encoding: _ENCODING = "MSB",
    *,
    rtol: float = 0.0,
    atol: float = 1e-8,
    max_denominator: int = 1000,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-8,
) -> str:
    r"""
    Return a string representation of a statevector in ket notation.

    Uses :meth:`graphix.sim.statevec.Statevec.to_dict` to obtain non-zero amplitudes,
    and formats each amplitude with :func:`complex_to_str`.

    Parameters
    ----------
    statevec : Statevec
        The statevector to format.
    output : OutputFormat
        Desired formatting style: Unicode, LaTeX, or ASCII.
    encoding : Literal["LSB", "MSB"], default="MSB"
        Encoding for the basis kets. See :meth:`graphix.sim.statevec.Statevec.to_dict`.
    rtol : float, default=0.0
        Relative tolerance for filtering zero amplitudes, passed to :meth:`Statevec.to_dict`.
    atol : float, default=1e-8
        Absolute tolerance for filtering zero amplitudes, passed to :meth:`Statevec.to_dict`.
    max_denominator : int, optional
        Maximum denominator for detecting simple amplitudes (default: 1000).
    rel_tol : float, optional
        Relative tolerance passed to :func:`complex_to_str` (default: ``1e-9``).
    abs_tol : float, optional
        Absolute tolerance passed to :func:`complex_to_str` (default: ``1e-8``).

    Returns
    -------
    str
        The formatted statevector as a sum of ket terms.
    """
    _validate_output_format(output)
    amplitudes = statevec.to_dict(encoding=encoding, rtol=rtol, atol=atol)
    terms = [
        _format_statevec_term(
            complex(amp),
            ket,
            output,
            max_denominator=max_denominator,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        for ket, amp in amplitudes.items()
    ]
    result = _join_statevec_terms(terms)
    if output == OutputFormat.LaTeX:
        return f"\\({result}\\)"
    return result


def density_matrix_to_str(
    density_matrix: DensityMatrix,
    output: OutputFormat,
    *,
    rtol: float = 0.0,
    atol: float = 1e-8,
    max_denominator: int = 1000,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-8,
) -> str:
    r"""
    Return a string representation of a density matrix.

    Formats each matrix element with :func:`complex_to_str`.

    Parameters
    ----------
    density_matrix : DensityMatrix
        The density matrix to format.
    output : OutputFormat
        Desired formatting style: Unicode, LaTeX, or ASCII.
    rtol : float, default=0.0
        Relative tolerance for displaying negligible elements as ``0``.
    atol : float, default=1e-8
        Absolute tolerance for displaying negligible elements as ``0``.
    max_denominator : int, optional
        Maximum denominator for detecting simple matrix elements (default: 1000).
    rel_tol : float, optional
        Relative tolerance passed to :func:`complex_to_str` (default: ``1e-9``).
    abs_tol : float, optional
        Absolute tolerance passed to :func:`complex_to_str` (default: ``1e-8``).

    Returns
    -------
    str
        The formatted density matrix.
    """
    _validate_output_format(output)
    rho = density_matrix.rho
    nrows, ncols = rho.shape

    def format_cell(value: complex) -> str:
        if math.isclose(abs(value), 0.0, rel_tol=rtol, abs_tol=atol):
            return "0"
        return complex_to_str(
            value,
            output,
            max_denominator=max_denominator,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    cells = [[format_cell(complex(rho[i, j])) for j in range(ncols)] for i in range(nrows)]
    col_widths = [max(len(cells[i][j]) for i in range(nrows)) for j in range(ncols)]

    if output == OutputFormat.LaTeX:
        rows = [" & ".join(cell.rjust(col_widths[j]) for j, cell in enumerate(row)) for row in cells]
        body = r" \\ ".join(rows)
        return rf"\(\begin{{pmatrix}} {body} \end{{pmatrix}}\)"

    lines: list[str] = []
    for i, row in enumerate(cells):
        inner = ", ".join(cell.rjust(col_widths[j]) for j, cell in enumerate(row))
        if nrows == 1:
            lines.append(f"[[{inner}]]")
        elif i == 0:
            lines.append(f"[[{inner}],")
        elif i == nrows - 1:
            lines.append(f" [{inner}]]")
        else:
            lines.append(f" [{inner}],")
    return "\n".join(lines)


def domain_to_str(domain: set[Node]) -> str:
    """Return the string representation of a domain."""
    return f"{{{','.join(str(node) for node in domain)}}}"


SUBSCRIPTS = str.maketrans(string.digits, "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPTS = str.maketrans(string.digits, "⁰¹²³⁴⁵⁶⁷⁸⁹")


def affine_expression_to_str(expr: AffineExpression, output: OutputFormat) -> str:
    """Return the string representation of an affine expression."""
    result = str(expr.x)
    if expr.a != 1:
        a = angle_to_str(rad_to_angle(expr.a), output)
        match output:
            case OutputFormat.LaTeX:
                mul = r" \times "
            case OutputFormat.Unicode:
                mul = "×"
            case OutputFormat.ASCII:
                mul = "*"
        result = f"{a}{mul}{result}"
    if expr.b != 0:
        result = f"{result}+{angle_to_str(rad_to_angle(expr.b), output)}"
    return result


def command_to_str(cmd: command.CommandType, output: OutputFormat) -> str:
    """Return the string representation of a command according to the given format.

    Parameters
    ----------
    cmd: CommandType
        The command to pretty print.
    output: OutputFormat
        The expected format.
    """
    out = [cmd.kind.name]

    match cmd.kind:
        case command.CommandKind.E:
            u, v = cmd.nodes
            match output:
                case OutputFormat.LaTeX:
                    out.append(f"_{{{u},{v}}}")
                case OutputFormat.Unicode:
                    u_subscripts = str(u).translate(SUBSCRIPTS)
                    v_subscripts = str(v).translate(SUBSCRIPTS)
                    out.append(f"{u_subscripts}₋{v_subscripts}")
                case _:
                    out.append(f"({u},{v})")
        case command.CommandKind.T:
            pass
        case _:
            # All other commands have a field `node` to print, together
            # with some other arguments and/or domains.
            arguments = []
            match cmd.kind:
                case command.CommandKind.M:
                    match cmd.measurement:
                        case BlochMeasurement(angle, plane):
                            if plane != Plane.XY:
                                arguments.append(plane.name)
                            # We use `SupportsFloat` since `isinstance(cmd.angle, float)`
                            # is `False` if `cmd.angle` is an integer.
                            match angle:
                                case SupportsFloat():
                                    s = angle_to_str(float(angle), output)
                                case AffineExpression():
                                    s = affine_expression_to_str(angle.scale_non_null(pi), output)
                                case _:
                                    # If the angle is a symbolic expression, we can only delegate the printing
                                    # TODO: We should have a mean to specify the format
                                    s = str(angle_to_rad(angle))
                            arguments.append(s)
                        case PauliMeasurement(Axis.X, Sign.PLUS):
                            pass
                        case _:
                            arguments.append(str(cmd.measurement))
                case command.CommandKind.C:
                    arguments.append(str(cmd.clifford))
            match cmd.kind:
                case command.CommandKind.X | command.CommandKind.Z | command.CommandKind.S:
                    command_domain: set[int] | None = cmd.domain
                case _:
                    command_domain = None
            match output:
                case OutputFormat.LaTeX:
                    out.append(f"_{{{cmd.node}}}")
                    if arguments:
                        out.append(f"^{{{','.join(arguments)}}}")
                case OutputFormat.Unicode:
                    node_subscripts = str(cmd.node).translate(SUBSCRIPTS)
                    out.append(f"{node_subscripts}")
                    if arguments:
                        out.append(f"({','.join(arguments)})")
                case _:
                    arguments = [str(cmd.node), *arguments]
                    if command_domain:
                        arguments.append(domain_to_str(command_domain))
                        command_domain = None
                    out.append(f"({','.join(arguments)})")
            if cmd.kind == command.CommandKind.M and (cmd.s_domain or cmd.t_domain):
                out = ["[", *out, "]"]
                if cmd.t_domain:
                    match output:
                        case OutputFormat.LaTeX:
                            t_domain_str = f"{{}}_{{{','.join(str(node) for node in cmd.t_domain)}}}"
                        case OutputFormat.Unicode:
                            t_domain_subscripts = [str(node).translate(SUBSCRIPTS) for node in cmd.t_domain]
                            t_domain_str = "₊".join(t_domain_subscripts)
                        case _:
                            t_domain_str = f"{{{','.join(str(node) for node in cmd.t_domain)}}}"
                    out = [t_domain_str, *out]
                command_domain = cmd.s_domain
            if command_domain:
                match output:
                    case OutputFormat.LaTeX:
                        domain_str = f"^{{{','.join(str(node) for node in command_domain)}}}"
                    case OutputFormat.Unicode:
                        domain_superscripts = [str(node).translate(SUPERSCRIPTS) for node in command_domain]
                        domain_str = "⁺".join(domain_superscripts)
                    case _:
                        domain_str = f"{{{','.join(str(node) for node in command_domain)}}}"
                out.append(domain_str)
    return f"{''.join(out)}"


def pattern_to_str(
    pattern: Pattern,
    output: OutputFormat,
    left_to_right: bool = False,
    limit: int | None = 40,
    target: Container[command.CommandKind] | None = None,
) -> str:
    """Return the string representation of a pattern according to the given format.

    Parameters
    ----------
    pattern: Pattern
        The pattern to pretty print.
    output: OutputFormat
        The expected format.
    left_to_right: bool, optional
        If ``True``, the first command will appear at the beginning of
        the resulting string. If ``False`` (the default), the first command will
        appear at the end of the string.
    limit: int | None, optional
        If set to an int (default: 40), only first ``limit`` commands are printed,
        and an ellipsis is added at the end to indicate that some commands have been elided.
        If ``limit=None``, there is no limit on the number of printed commands.
    target: Container[command.CommandKind], optional
        If set, only commands of kinds specified in ``target`` are printed.
    """
    separator = r"\," if output == OutputFormat.LaTeX else " "
    command_list = list(pattern)
    if target is not None:
        command_list = [command for command in command_list if command.kind in target]
    if not left_to_right:
        command_list.reverse()
    truncated = limit is not None and len(command_list) > limit
    # Note: The redundant test `limit is not None` is required for mypy
    # to narrow the type of `limit` in the then-branch.
    short_command_list = command_list[: limit - 1] if limit is not None and truncated else command_list
    result = separator.join(command_to_str(command, output) for command in short_command_list)
    if output == OutputFormat.LaTeX:
        result = f"\\({result}\\)"
    if limit is not None and truncated:
        return f"{result}...({len(command_list) - limit + 1} more commands)"
    return result


def set_to_str(objects: Iterable[object], output: OutputFormat) -> str:
    """Convert a set to a formatted string representation.

    Parameters
    ----------
    objects : Iterable[object]
        The set to format.
    output : OutputFormat
        The desired output format (ASCII, LaTeX or Unicode).
    """
    contents = ", ".join(str(item) for item in objects)
    if output == OutputFormat.LaTeX:
        return f"\\{{{contents}\\}}"
    return f"{{{contents}}}"


def correction_function_to_str(
    correction_function: Mapping[int, AbstractSet[int]],
    cf_name: str,
    output: OutputFormat,
    multiline: bool = False,
) -> str:
    """Convert a correction function mapping to a formatted string representation.

    Parameters
    ----------
    correction_function : Mapping[int, AbstractSet[int]]
        A mapping from node indices to sets of node indices representing the
        correction function. See :class:`graphix.flow.core.PauliFlow` for additional information.
    cf_name : str
        The name of the correction function (e.g., ``c`` for causal flow, ``g`` for gflow, etc.)
    output : OutputFormat
        The desired output format (ASCII, LaTeX or Unicode).
    multiline : bool, optional
        If ``True``, format each correction set on a separate line (or LaTeX line break).
        If ``False``, format each correction set on a single line separated by commas.
        Default is ``False``.

    Returns
    -------
    str
    """
    separator = (
        (r",\\" if output == OutputFormat.LaTeX else "\n")
        if multiline
        else (r", \;" if output == OutputFormat.LaTeX else ", ")
    )

    return separator.join(
        f"{cf_name}({node}) = {set_to_str(cset, output)}" for node, cset in correction_function.items()
    )


def partial_order_to_str(partial_order_layers: Sequence[AbstractSet[int]], output: OutputFormat) -> str:
    """Convert a partial order layering to a formatted string representation.

    Parameters
    ----------
    partial_order_layers : Sequence[AbstractSet[int]]
        Partial order between nodes in a layer form. See :class:`graphix.flow.core.PauliFlow` for additional information.
    output : OutputFormat
        The desired output format (ASCII, LaTeX or Unicode).

    Returns
    -------
    str
    """
    match output:
        case OutputFormat.ASCII:
            separator = " < "
        case OutputFormat.Unicode:
            separator = " ≺ "
        case OutputFormat.LaTeX:
            separator = r" \prec "
        case _:
            assert_never(output)

    return separator.join(f"{set_to_str(layer, output)}" for layer in partial_order_layers[::-1])


def component_separator_for(output: OutputFormat, multiline: bool = False) -> str:
    """Return a component separator to string-format a `PauliFlow` or a `XZCorrections` object.

    Parameters
    ----------
    output : OutputFormat
        The desired output format (ASCII, LaTeX or Unicode).
    multiline : bool, optional
        If ``True``, format each component on a separate line (or LaTeX line break).
        If ``False``, format each component set on a single line separated by semicolons.
        Default is ``False``.

    Returns
    -------
    str
    """
    return (
        (r";\\" if output == OutputFormat.LaTeX else "\n")
        if multiline
        else (r"; \;" if output == OutputFormat.LaTeX else "; ")
    )


def flow_to_str(flow: PauliFlow[AbstractMeasurement], output: OutputFormat, multiline: bool = False) -> str:
    """Convert a flow object to a formatted string representation.

    Parameters
    ----------
    flow : PauliFlow[AbstractMeasurement]
        The flow object to be formatted.
    output : OutputFormat
        The desired output format (ASCII, LaTeX or Unicode).
    multiline : bool, optional
        If ``True``, format each correction set on a separate line (or LaTeX line break).
        If ``False``, format each correction set on a single line separated by commas.
        Default is ``False``.

    Returns
    -------
    str
        A string representation of the flow object formatted according to the specified output format and layout.
    """
    separator = component_separator_for(output, multiline)

    return separator.join(
        (
            correction_function_to_str(flow.correction_function, flow._CF_PREFIX, output, multiline),
            partial_order_to_str(flow.partial_order_layers, output),
        )
    )


def xzcorr_to_str(
    xzcorr: XZCorrections[AbstractMeasurement],
    output: OutputFormat,
    multiline: bool = False,
) -> str:
    """Convert an XZCorrections object to a formatted string representation.

    Parameters
    ----------
    flow : XZCorrections[AbstractMeasurement]
        The XZCorrections object to be formatted.
    output : OutputFormat
        The desired output format (ASCII, LaTeX or Unicode).
    multiline : bool, optional
        If ``True``, format each correction set on a separate line (or LaTeX line break).
        If ``False``, format each correction set on a single line separated by commas.
        Default is ``False``.

    Returns
    -------
    str
        A string representation of the XZCorrections object formatted according to the specified output format and layout.
    """
    separator = component_separator_for(output, multiline)

    return separator.join(
        (
            correction_function_to_str(xzcorr.x_corrections, "x", output, multiline),
            correction_function_to_str(xzcorr.z_corrections, "z", output, multiline),
            partial_order_to_str(xzcorr.partial_order_layers, output),
        )
    )

"""Pretty-printing utilities."""

from __future__ import annotations

import enum
import math
import string
from enum import Enum
from fractions import Fraction
from math import pi
from typing import TYPE_CHECKING, SupportsFloat

import numpy as np

# `assert_never` introduced in Python 3.11
from typing_extensions import assert_never

from graphix import command
from graphix.fundamentals import AbstractMeasurement, Axis, Plane, Sign, angle_to_rad, rad_to_angle
from graphix.measurements import BlochMeasurement, PauliMeasurement
from graphix.parameter import AffineExpression

if TYPE_CHECKING:
    from collections.abc import Container, Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    import numpy.typing as npt

    from graphix.command import Node
    from graphix.flow.core import PauliFlow, XZCorrections
    from graphix.fundamentals import Angle
    from graphix.pattern import Pattern


class OutputFormat(Enum):
    """Enumeration of the output format for pretty-printing."""

    ASCII = enum.auto()
    LaTeX = enum.auto()
    Unicode = enum.auto()


def angle_to_str(
    angle: Angle, output: OutputFormat, max_denominator: int = 1000, multiplication_sign: bool = False
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

    Returns
    -------
    str
        The formatted angle.
    """
    frac = Fraction(angle).limit_denominator(max_denominator)

    if not math.isclose(angle, float(frac)):
        rad = angle_to_rad(angle)

        return f"{rad}"

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
    correction_function: Mapping[int, AbstractSet[int]], cf_name: str, output: OutputFormat, multiline: bool = False
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


def xzcorr_to_str(xzcorr: XZCorrections[AbstractMeasurement], output: OutputFormat, multiline: bool = False) -> str:
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


# ---------------------------------------------------------------------------
# Pretty-printing for quantum states (statevectors and density matrices)
# ---------------------------------------------------------------------------


_EPS = 1e-10


def _format_imag_unit(output: OutputFormat) -> str:
    """Return the imaginary unit formatted for the given output format."""
    if output == OutputFormat.LaTeX:
        return r"\mathrm{i}"
    return "i"


def _format_ket(basis: str, output: OutputFormat) -> str:
    """Format a basis bitstring as a ket.

    Parameters
    ----------
    basis : str
        Bitstring representation of a basis state (e.g. ``"01"``).
    output : OutputFormat
        Desired formatting style.

    Returns
    -------
    str
        The formatted ket.
    """
    if output == OutputFormat.LaTeX:
        return rf"\lvert {basis}\rangle"
    if output == OutputFormat.Unicode:
        return f"|{basis}⟩"
    return f"|{basis}>"


def _format_bra(basis: str, output: OutputFormat) -> str:
    """Format a basis bitstring as a bra.

    Parameters
    ----------
    basis : str
        Bitstring representation of a basis state (e.g. ``"01"``).
    output : OutputFormat
        Desired formatting style.

    Returns
    -------
    str
        The formatted bra.
    """
    if output == OutputFormat.LaTeX:
        return rf"\langle {basis}\rvert"
    if output == OutputFormat.Unicode:
        return f"⟨{basis}|"
    return f"<{basis}|"


def _format_frac(num: int, den: int, output: OutputFormat) -> str:
    """Format a fraction *num* / *den*.

    Parameters
    ----------
    num : int
        Numerator.
    den : int
        Denominator.
    output : OutputFormat
        Desired formatting style.

    Returns
    -------
    str
    """
    if den == 1:
        return str(num)

    if output == OutputFormat.LaTeX:
        return rf"\frac{{{num}}}{{{den}}}"
    return f"{num}/{den}"


def _format_scalar(val: float, output: OutputFormat, max_denominator: int = 1000) -> str:
    """Format a real scalar, detecting integers and simple fractions.

    Parameters
    ----------
    val : float
        Value to format.
    output : OutputFormat
        Desired formatting style.
    max_denominator : int, optional
        Maximum denominator for fraction detection (default: 1000).

    Returns
    -------
    str
    """
    if abs(val) < _EPS:
        return "0"

    # Integer
    if math.isclose(val, round(val), abs_tol=_EPS):
        return str(round(val))

    # Simple fraction
    frac = Fraction(val).limit_denominator(max_denominator)
    if math.isclose(val, float(frac), abs_tol=_EPS):
        return _format_frac(frac.numerator, frac.denominator, output)

    # Square-root of a simple fraction: |val| ≈ sqrt(n/d)
    val_sq = val * val
    frac_sq = Fraction(val_sq).limit_denominator(max_denominator)
    if math.isclose(val_sq, float(frac_sq), abs_tol=_EPS) and frac_sq != frac:
        result = _format_sqrt_frac(frac_sq, output)
        return f"-{result}" if val < 0 else result

    # Fallback: decimal
    return f"{val:.6g}"


def _format_sqrt_frac(frac: Fraction, output: OutputFormat) -> str:
    """Format the square root of a positive fraction.

    Parameters
    ----------
    frac : Fraction
        A positive fraction. The formatted string represents ``sqrt(frac)``.
    output : OutputFormat
        Desired formatting style.

    Returns
    -------
    str
    """
    num = frac.numerator
    den = frac.denominator

    sqrt_sym: str
    if output == OutputFormat.LaTeX:
        sqrt_sym = r"\sqrt"
    elif output == OutputFormat.Unicode:
        sqrt_sym = "√"
    else:
        sqrt_sym = "sqrt"

    if den == 1:
        if output == OutputFormat.LaTeX:
            return rf"{sqrt_sym}{{{num}}}"
        return f"{sqrt_sym}{num}" if output == OutputFormat.Unicode else f"{sqrt_sym}({num})"

    # Try to simplify by pulling perfect squares out of the denominator
    den_sqrt = math.isqrt(den)
    if den_sqrt * den_sqrt == den:
        # den is a perfect square: √(num/den) = √num / den_sqrt
        def _sqrt_num(n: int) -> str:
            if output == OutputFormat.LaTeX:
                return rf"{sqrt_sym}{{{n}}}"
            return f"{sqrt_sym}{n}" if output == OutputFormat.Unicode else f"{sqrt_sym}({n})"

        if num == 1:
            # √(1/den) = 1/den_sqrt
            if output == OutputFormat.LaTeX:
                return rf"\frac{{1}}{{{den_sqrt}}}"
            return f"1/{den_sqrt}"
        sqrt_top = _sqrt_num(num)
        if output == OutputFormat.LaTeX:
            return rf"\frac{{{sqrt_top}}}{{{den_sqrt}}}"
        return f"{sqrt_top}/{den_sqrt}"

    if num == 1:
        # 1/√(den)
        if output == OutputFormat.LaTeX:
            return rf"\frac{{1}}{{{sqrt_sym}{{{den}}}}}"
        inner = f"{sqrt_sym}{den}" if output == OutputFormat.Unicode else f"{sqrt_sym}({den})"
        return f"1/{inner}"

    # General: √(num/den)
    if output == OutputFormat.LaTeX:
        return rf"{sqrt_sym}{{\frac{{{num}}}{{{den}}}}}"
    return f"{sqrt_sym}({num}/{den})"


def complex_to_str(
    z: complex | np.complex128 | np.object_ | float,
    output: OutputFormat,
    max_denominator: int = 1000,
) -> str:
    r"""Pretty-print a complex number.

    Detects common values and renders them in a human-readable form:

    - *zero* → ``"0"``
    - *simple fractions* → ``"1/2"``, ``"3/4"``, etc.
    - *square roots of fractions* → ``"1/√2"``, ``"√3/2"``, etc.
    - *pure exponentials* (when ``|z| ≈ 1``) → ``"e^{iπ/3}"``, ``"e^{-iπ/2}"``, etc.
    - *fallback* → decimal notation.

    Parameters
    ----------
    z : complex or np.complex128 or np.object_ or float
        The complex number to format.
    output : OutputFormat
        Desired formatting style (ASCII, LaTeX or Unicode).
    max_denominator : int, optional
        Maximum denominator when detecting simple fractions (default: 1000).

    Returns
    -------
    str
    """
    if not isinstance(z, (complex, float, int, np.complexfloating, np.floating)):
        return str(z)

    if abs(z) < _EPS:
        return "0"

    imag_unit = _format_imag_unit(output)
    re, im = z.real, z.imag
    pure_real = abs(im) < _EPS
    pure_imag = abs(re) < _EPS

    # Pure-real numbers: format as a scalar
    if pure_real:
        return _format_scalar(re, output, max_denominator)

    # Pure-imaginary numbers: format as scalar · i
    if pure_imag:
        imag_str = _format_scalar(abs(im), output, max_denominator)
        if imag_str == "1":
            return imag_unit if im > 0 else f"-{imag_unit}"
        return f"{imag_str}{imag_unit}" if im > 0 else f"-{imag_str}{imag_unit}"

    # Exponential: |z| ≈ 1 and both components non-zero
    if abs(abs(z) - 1.0) < _EPS:
        angle = math.atan2(im, re)
        angle_in_pi = angle / math.pi
        frac = Fraction(angle_in_pi).limit_denominator(max_denominator)
        if math.isclose(angle_in_pi, float(frac), abs_tol=_EPS):
            return _format_exponential(float(frac), output, max_denominator)

    # General complex: a ± b·i
    real_str = _format_scalar(re, output, max_denominator)
    imag_str = _format_scalar(abs(im), output, max_denominator)
    imag_str = imag_unit if imag_str == "1" else f"{imag_str}{imag_unit}"

    if im >= 0:
        return f"{real_str}+{imag_str}"
    return f"{real_str}-{imag_str}"


def _format_exponential(angle_in_pi: float, output: OutputFormat, max_denominator: int) -> str:
    """Format a pure phase as ``e^{iθ}`` where *θ* is a fraction of π.

    Parameters
    ----------
    angle_in_pi : float
        The phase angle in units of π.
    output : OutputFormat
        Desired formatting style.
    max_denominator : int
        Maximum denominator for fraction detection.

    Returns
    -------
    str
    """
    imag_unit = _format_imag_unit(output)

    if abs(angle_in_pi) < _EPS:
        return "1"

    frac = Fraction(angle_in_pi).limit_denominator(max_denominator)
    num = frac.numerator
    den = frac.denominator

    if num < 0:
        sign_prefix = "-"
        num = -num
    else:
        sign_prefix = ""

    # Build the body in i·num·π/den format
    if output == OutputFormat.LaTeX:
        if num == 1 and den == 1:
            body = f"{imag_unit}\\pi"
        elif num == 1:
            body = f"{imag_unit}\\pi/{den}"
        elif den == 1:
            body = f"{imag_unit}{num}\\pi"
        else:
            body = f"{imag_unit}{num}\\pi/{den}"
    elif output == OutputFormat.Unicode:
        if num == 1 and den == 1:
            body = f"{imag_unit}π"
        elif num == 1:
            body = f"{imag_unit}π/{den}"
        elif den == 1:
            body = f"{imag_unit}{num}π"
        else:
            body = f"{imag_unit}{num}π/{den}"
    elif num == 1 and den == 1:
        body = f"{imag_unit}*pi"
    elif num == 1:
        body = f"{imag_unit}*pi/{den}"
    elif den == 1:
        body = f"{imag_unit}*{num}*pi"
    else:
        body = f"{imag_unit}*{num}*pi/{den}"

    body = f"{sign_prefix}{body}"

    if output == OutputFormat.LaTeX:
        return rf"\mathrm{{e}}^{{{body}}}"
    if output == OutputFormat.Unicode:
        return f"e^({body})"
    return f"exp({body})"


def _basis_label(index: int, nqubit: int) -> str:
    """Return the bitstring label for a basis index.

    Parameters
    ----------
    index : int
        Basis index (integer).
    nqubit : int
        Number of qubits.

    Returns
    -------
    str
        Bitstring of length *nqubit*.
    """
    return f"{index:0{nqubit}b}"


def statevec_to_str(
    sv_dict: Mapping[str, np.object_ | np.complex128],
    output: OutputFormat,
    max_denominator: int = 1000,
) -> str:
    """Pretty-print a statevector from its dictionary representation.

    Parameters
    ----------
    sv_dict : Mapping[str, complex]
        Statevector dictionary as returned by :meth:`graphix.sim.statevec.Statevec.to_dict`.
    output : OutputFormat
        Desired formatting style (ASCII, LaTeX or Unicode).
    max_denominator : int, optional
        Maximum denominator for fraction detection (default: 1000).

    Returns
    -------
    str
    """
    if not sv_dict:
        return "0"

    parts: list[str] = []
    for basis, amplitude in sv_dict.items():
        amp_str = complex_to_str(complex(amplitude), output, max_denominator)
        ket = _format_ket(basis, output)

        if amp_str == "1":
            term = ket
        elif amp_str == "-1":
            term = f"-{ket}"
        else:
            term = f"{amp_str}{ket}"

        if not parts:
            parts.append(term)
        elif term.startswith("-"):
            parts.append(f" - {term[1:]}")
        else:
            parts.append(f" + {term}")

    return "".join(parts)


def densitymatrix_to_str(
    rho: npt.NDArray[np.object_ | np.complex128],
    nqubit: int,
    output: OutputFormat,
    *,
    max_denominator: int = 1000,
    cutoff: float = 1e-10,
) -> str:
    r"""Pretty-print a density matrix using Dirac notation.

    Extracts non-zero elements and formats them as a sum of weighted projectors:

    .. math::

        \\rho = \\sum_{i,j} \\rho_{ij} \\lvert i \\rangle\\langle j \\rvert

    Parameters
    ----------
    rho : Matrix
        The density matrix as a ``2**nqubit × 2**nqubit`` array.
    nqubit : int
        Number of qubits.
    output : OutputFormat
        Desired formatting style (ASCII, LaTeX or Unicode).
    max_denominator : int, optional
        Maximum denominator for fraction detection (default: 1000).
    cutoff : float, optional
        Tolerance below which matrix elements are treated as zero (default: ``1e-10``).

    Returns
    -------
    str
    """
    n = rho.shape[0]

    terms: list[tuple[complex, str, str]] = []
    for i in range(n):
        for j in range(n):
            val: complex = complex(rho[i, j])
            if abs(val) < cutoff:
                continue
            val_str = complex_to_str(val, output, max_denominator)
            ket = _format_ket(_basis_label(i, nqubit), output)
            bra = _format_bra(_basis_label(j, nqubit), output)
            # |i⟩⟨j|
            dirac = f"{ket}{bra}"
            terms.append((val, val_str, dirac))

    if not terms:
        return "0"

    result_parts: list[str] = []
    for i, (_val, val_str, dirac) in enumerate(terms):
        if val_str == "1":
            term = dirac
        elif val_str == "-1":
            term = f"-{dirac}"
        else:
            term = f"{val_str}{dirac}"

        if i == 0:
            result_parts.append(term)
        elif term.startswith("-"):
            result_parts.append(f" - {term[1:]}")
        else:
            result_parts.append(f" + {term}")

    return "".join(result_parts)

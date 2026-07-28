"""Pretty-printing utilities."""

from __future__ import annotations

import enum
import math
import string
from enum import Enum
from fractions import Fraction
from math import pi
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat

# `assert_never` introduced in Python 3.11
from typing_extensions import assert_never

from graphix import command
from graphix.fundamentals import AbstractMeasurement, Axis, Plane, Sign, angle_to_rad, rad_to_angle
from graphix.measurements import BlochMeasurement, PauliMeasurement
from graphix.parameter import AffineExpression

if TYPE_CHECKING:
    from collections.abc import Container, Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphix.command import Node
    from graphix.flow.core import PauliFlow, XZCorrections
    from graphix.fundamentals import Angle
    from graphix.pattern import Pattern
    from graphix.sim.density_matrix import DensityMatrix
    from graphix.sim.statevec import _ENCODING, Statevector


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


# --- Complex amplitude and quantum-state pretty-printing ---------------------
#
# The recognition of "nice" real numbers (fractions, square roots) relies on a
# square-then-rationalize trick: a real ``x`` is matched against ``sqrt(p / q)``
# by approximating ``x ** 2`` with a rational ``p / q``. This single mechanism
# uniformly handles plain fractions (``1/4``), surds (``√2/2``, ``√3/2``) and,
# combined with :func:`angle_to_str`, the phase of exponentials (``e^{iπ/3}``).

_DEFAULT_MAX_DENOMINATOR = 1000
_DEFAULT_ATOL = 1e-9
_DEFAULT_RTOL = 0.0
_DEFAULT_PRECISION = 4


def _squarefree_decomposition(n: int) -> tuple[int, int]:
    """Decompose a non-negative integer as ``outer ** 2 * inner`` with ``inner`` squarefree.

    Parameters
    ----------
    n : int
        Non-negative integer to decompose.

    Returns
    -------
    tuple[int, int]
        ``(outer, inner)`` such that ``outer ** 2 * inner == n`` and ``inner`` is
        squarefree. ``n == 0`` returns ``(0, 1)``.
    """
    if n == 0:
        return 0, 1
    outer = 1
    inner = n
    d = 2
    while d * d <= inner:
        while inner % (d * d) == 0:
            inner //= d * d
            outer *= d
        d += 1
    return outer, inner


def _recognize_sqrt(x: float, max_denominator: int, atol: float, rtol: float) -> tuple[int, int, int] | None:
    """Recognize a real number as ``signed_num * sqrt(inner) / den``.

    The recognition approximates ``x ** 2`` by a rational ``p / q``; on success,
    ``x = ±sqrt(p / q)`` is rewritten with a rationalized, fully-reduced
    denominator. Pure rationals are covered as the special case ``inner == 1``.

    Parameters
    ----------
    x : float
        Real number to recognize.
    max_denominator : int
        Maximum denominator allowed when approximating ``x ** 2`` by a rational.
    atol : float
        Absolute tolerance for that rational approximation.
    rtol : float
        Relative tolerance for that rational approximation.

    Returns
    -------
    tuple[int, int, int] or None
        ``(signed_num, inner, den)`` with ``den > 0`` and ``inner`` a positive
        squarefree integer, encoding ``x = signed_num * sqrt(inner) / den``.
        Returns ``None`` when ``x`` is not recognized as such a value.
    """
    # Direct rational case first: ``x = ±p / q`` with ``q ≤ max_denominator``.
    # Without this, the squared-form path below would effectively require
    # ``q ** 2 ≤ max_denominator`` to recognise simple rationals like ``1/2``,
    # which surprises callers who pass ``max_denominator=2`` expecting denominators
    # up to 2 in the output. By trying the direct fraction first, ``max_denominator``
    # consistently bounds the denominator of the rendered value across all callers.
    direct = Fraction(x).limit_denominator(max_denominator)
    if math.isclose(x, float(direct), rel_tol=rtol, abs_tol=atol):
        magnitude = abs(direct.numerator)
        sign = -1 if direct.numerator < 0 else 1
        return sign * magnitude, 1, direct.denominator

    square = Fraction(x * x).limit_denominator(max_denominator)
    if not math.isclose(x * x, float(square), rel_tol=rtol, abs_tol=atol):
        return None
    num_outer, num_inner = _squarefree_decomposition(square.numerator)
    den_outer, den_inner = _squarefree_decomposition(square.denominator)
    # x = ±(num_outer √num_inner) / (den_outer √den_inner); rationalize by √den_inner.
    combined_outer, inner = _squarefree_decomposition(num_inner * den_inner)
    num = num_outer * combined_outer
    den = den_outer * den_inner
    divisor = math.gcd(num, den)
    num //= divisor
    den //= divisor
    sign = -1 if x < 0 else 1
    return sign * num, inner, den


def _imaginary_unit(output: OutputFormat) -> str:
    return r"\mathrm{i}" if output == OutputFormat.LaTeX else "i"


def _sqrt_str(inner: int, output: OutputFormat) -> str:
    """Return the string for ``sqrt(inner)`` (empty when ``inner == 1``)."""
    if inner == 1:
        return ""
    if output == OutputFormat.LaTeX:
        return rf"\sqrt{{{inner}}}"
    if output == OutputFormat.Unicode:
        return f"√{inner}"
    return f"sqrt({inner})"


def _fraction_str(num: str, den: str, output: OutputFormat) -> str:
    if output == OutputFormat.LaTeX:
        return rf"\frac{{{num}}}{{{den}}}"
    return f"{num}/{den}"


def _render_real(signed_num: int, inner: int, den: int, output: OutputFormat) -> str:
    """Render ``signed_num * sqrt(inner) / den`` produced by :func:`_recognize_sqrt`."""
    if signed_num == 0:
        return "0"
    sign = "-" if signed_num < 0 else ""
    magnitude = abs(signed_num)
    sqrt_part = _sqrt_str(inner, output)
    if inner == 1:
        numerator = f"{magnitude}"
    elif magnitude == 1:
        numerator = sqrt_part
    else:
        numerator = f"{magnitude}{sqrt_part}"
    if den == 1:
        return f"{sign}{numerator}"
    return f"{sign}{_fraction_str(numerator, str(den), output)}"


def _real_to_str(x: float, output: OutputFormat, max_denominator: int, atol: float, rtol: float) -> str | None:
    rec = _recognize_sqrt(x, max_denominator, atol, rtol)
    if rec is None:
        return None
    return _render_real(*rec, output)


def _render_imaginary(signed_num: int, inner: int, den: int, output: OutputFormat) -> str:
    """Render ``signed_num * sqrt(inner) / den * i`` with the unit leading the numerator.

    The imaginary unit is placed at the front of the numerator (e.g. ``i/2``,
    ``i√2/2``, ``3i/4``) so that it reads as ``i`` times a real magnitude, with a
    unit coefficient collapsing to a bare ``±i``.
    """
    sign = "-" if signed_num < 0 else ""
    magnitude = abs(signed_num)
    unit = _imaginary_unit(output)
    coefficient = "" if magnitude == 1 else f"{magnitude}"
    numerator = f"{coefficient}{unit}{_sqrt_str(inner, output)}"
    if den == 1:
        return f"{sign}{numerator}"
    return f"{sign}{_fraction_str(numerator, str(den), output)}"


def _imaginary_to_str(x: float, output: OutputFormat, max_denominator: int, atol: float, rtol: float) -> str | None:
    """Render a purely imaginary value ``x * i``."""
    rec = _recognize_sqrt(x, max_denominator, atol, rtol)
    if rec is None:
        return None
    return _render_imaginary(*rec, output)


def _recognize_angle_over_pi(theta: float, max_denominator: int, atol: float, rtol: float) -> Fraction | None:
    """Return ``theta / pi`` as a simple fraction, or ``None`` if it is not one."""
    value = theta / pi
    frac = Fraction(value).limit_denominator(max_denominator)
    if math.isclose(value, float(frac), rel_tol=rtol, abs_tol=atol):
        return frac
    return None


def _exponential_to_str(z: complex, output: OutputFormat, max_denominator: int, atol: float, rtol: float) -> str | None:
    """Render ``z`` as ``r e^{iθ}`` when both ``r`` and ``θ / π`` are recognized."""
    theta = math.atan2(z.imag, z.real)
    angle_frac = _recognize_angle_over_pi(theta, max_denominator, atol, rtol)
    if angle_frac is None or angle_frac == 0:
        return None
    radius = _real_to_str(math.hypot(z.real, z.imag), output, max_denominator, atol, rtol)
    if radius is None:
        return None
    sign = "-" if angle_frac < 0 else ""
    angle_str = angle_to_str(float(abs(angle_frac)), output)
    unit = _imaginary_unit(output)
    e_sym = r"\mathrm{e}" if output == OutputFormat.LaTeX else "e"
    unit_sep = " " if output == OutputFormat.LaTeX else "*" if output == OutputFormat.ASCII else ""
    exponent = f"{sign}{unit}{unit_sep}{angle_str}"
    body = f"{e_sym}^{{{exponent}}}" if output == OutputFormat.LaTeX else f"{e_sym}^({exponent})"
    if radius == "1":
        return body
    prefix_sep = " " if output == OutputFormat.LaTeX else "·" if output == OutputFormat.Unicode else "*"
    return f"{radius}{prefix_sep}{body}"


def _cartesian_to_str(
    re: float, im: float, output: OutputFormat, max_denominator: int, atol: float, rtol: float
) -> str | None:
    """Render ``re + im i`` when both parts are recognized as nice reals."""
    re_str = _real_to_str(re, output, max_denominator, atol, rtol)
    im_rec = _recognize_sqrt(im, max_denominator, atol, rtol)
    if re_str is None or im_rec is None:
        return None
    signed_num, inner, den = im_rec
    connector = " - " if signed_num < 0 else " + "
    unit = _imaginary_unit(output)
    if abs(signed_num) == 1 and inner == 1 and den == 1:
        imag = unit
    else:
        imag = f"{_render_real(abs(signed_num), inner, den, output)}{unit}"
    return f"{re_str}{connector}{imag}"


def _decimal_to_str(z: complex, output: OutputFormat, precision: int, atol: float) -> str:
    """Fallback formatting using rounded decimals with ``precision`` significant digits."""
    unit = _imaginary_unit(output)
    if abs(z.imag) <= atol:
        return f"{z.real:.{precision}g}"
    if abs(z.real) <= atol:
        return f"{z.imag:.{precision}g}{unit}"
    return f"{z.real:.{precision}g}{z.imag:+.{precision}g}{unit}"


def complex_to_str(
    value: object,
    output: OutputFormat,
    *,
    max_denominator: int = _DEFAULT_MAX_DENOMINATOR,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
    precision: int = _DEFAULT_PRECISION,
) -> str:
    r"""Return a human-friendly string representation of a complex number.

    Common values are rendered exactly rather than as floating-point numbers:
    fractions (``0.25`` → ``1/4``), square roots (``0.7071…`` → ``√2/2``) and
    complex exponentials (``0.5 + 0.866…j`` → ``e^(iπ/3)``). Values that are not
    recognized fall back to a rounded decimal representation, and inputs that
    cannot be interpreted as complex numbers (e.g. symbolic parameters) are
    returned via :func:`str`.

    Parameters
    ----------
    value : object
        The number to format. Anything supporting conversion to ``complex`` is
        accepted; other objects are stringified.
    output : OutputFormat
        Desired formatting style: ``Unicode`` (``√``, ``π``), ``LaTeX``
        (``\sqrt``, ``\pi``) or ``ASCII`` (``sqrt``, ``pi``).
    max_denominator : int, optional
        Maximum denominator used when recognizing rational magnitudes and phases
        (default: ``1000``).
    atol : float, optional
        Absolute tolerance for the recognition heuristics (default: ``1e-9``).
    rtol : float, optional
        Relative tolerance for the recognition heuristics (default: ``0.0``).
    precision : int, optional
        Number of significant digits to use for the decimal fallback when a
        value is not recognized as an exact form (default: ``4``).

    Returns
    -------
    str
        The formatted complex number.

    Examples
    --------
    >>> complex_to_str(0.25, OutputFormat.ASCII)
    '1/4'
    >>> complex_to_str(2**-0.5, OutputFormat.Unicode)
    '√2/2'
    >>> complex_to_str(0.5 + 0.8660254037844386j, OutputFormat.Unicode)
    'e^(iπ/3)'
    >>> complex_to_str(0.123456 + 0.234567j, OutputFormat.ASCII, precision=2)
    '0.12+0.23i'
    """
    if not isinstance(value, (bool, int, float, complex, SupportsComplex)):
        return str(value)
    z = complex(value)
    if abs(z.real) <= atol and abs(z.imag) <= atol:
        return "0"
    if abs(z.imag) <= atol:
        return _real_to_str(z.real, output, max_denominator, atol, rtol) or _decimal_to_str(z, output, precision, atol)
    if abs(z.real) <= atol:
        return _imaginary_to_str(z.imag, output, max_denominator, atol, rtol) or _decimal_to_str(
            z, output, precision, atol
        )
    exponential = _exponential_to_str(z, output, max_denominator, atol, rtol)
    # The exponential form is the clearest representation on the unit circle, but for
    # other moduli a Cartesian form (e.g. ``1 + i`` rather than ``√2·e^(iπ/4)``) reads
    # better, so it is preferred when both parts are recognized.
    if exponential is not None and math.isclose(math.hypot(z.real, z.imag), 1.0, rel_tol=rtol, abs_tol=atol):
        return exponential
    cartesian = _cartesian_to_str(z.real, z.imag, output, max_denominator, atol, rtol)
    if cartesian is not None:
        return cartesian
    # Fall back to a radius-prefixed exponential when the Cartesian parts are not nice.
    if exponential is not None:
        return exponential
    return _decimal_to_str(z, output, precision, atol)


def _ket_str(ket: str, output: OutputFormat) -> str:
    if output == OutputFormat.LaTeX:
        return rf"\ket{{{ket}}}"
    if output == OutputFormat.Unicode:
        return f"|{ket}⟩"
    return f"|{ket}>"


def _needs_parentheses(coefficient: str) -> bool:
    """Whether a coefficient is a sum and must be parenthesized before a ket."""
    return " + " in coefficient or " - " in coefficient


def _factor_uniform_magnitude(
    amplitudes: Sequence[object],
    kets: Sequence[str],
    output: OutputFormat,
    *,
    max_denominator: int,
    atol: float,
    rtol: float,
    precision: int,
) -> str | None:
    """Factor a modulus shared by every amplitude, e.g. ``√2/2(|0⟩ + i|1⟩)``.

    Amplitudes that share a common modulus (up to a relative phase) are written as
    ``r(φ₀|k₀⟩ + φ₁|k₁⟩ + …)``, with ``r`` the shared modulus and each ``φ`` the
    per-component phase ``amplitude / r`` (so a relative phase such as ``i`` is kept
    inside the parentheses). Returns ``None`` when the moduli differ, the shared
    modulus is trivial (``1``), or the amplitudes are not numeric (e.g. symbolic),
    in which case the caller renders term-by-term.
    """
    if len(amplitudes) < 2:
        return None
    try:
        values = [complex(amplitude) for amplitude in amplitudes]  # type: ignore[call-overload]
    except (TypeError, ValueError):
        # Non-numeric (e.g. symbolic) amplitudes have no modulus to factor.
        return None
    radius = abs(values[0])
    if radius == 0 or any(not math.isclose(abs(v), radius, rel_tol=rtol, abs_tol=atol) for v in values):
        return None
    radius_str = complex_to_str(
        radius, output, max_denominator=max_denominator, atol=atol, rtol=rtol, precision=precision
    )
    # A unit modulus is not worth factoring (it would read as ``1(...)``).
    if radius_str == "1":
        return None
    result = ""
    for index, (value, ket) in enumerate(zip(values, kets, strict=True)):
        phase = complex_to_str(
            value / radius, output, max_denominator=max_denominator, atol=atol, rtol=rtol, precision=precision
        )
        # A per-component phase is unit-modulus, so it is rendered as 1, -1, ±i, an
        # exponential, or a decimal — never a parenthesizable sum.
        if phase == "1":
            term = ket
        elif phase == "-1":
            term = f"-{ket}"
        else:
            term = f"{phase}{ket}"
        if index == 0:
            result = term
        elif term.startswith("-"):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    return f"{radius_str}({result})"


def statevec_to_str(
    statevec: Statevector,
    output: OutputFormat,
    *,
    encoding: _ENCODING = "MSB",
    max_denominator: int = _DEFAULT_MAX_DENOMINATOR,
    atol: float = _DEFAULT_ATOL,
    rtol: float = 0.0,
    precision: int = _DEFAULT_PRECISION,
) -> str:
    r"""Return a ket-notation string representation of a statevector.

    Amplitudes close to zero are omitted (see :meth:`graphix.sim.statevec.Statevector.to_dict`)
    and the remaining ones are pretty-printed with :func:`complex_to_str`.

    Parameters
    ----------
    statevec : Statevector
        The statevector to format.
    output : OutputFormat
        Desired formatting style (``ASCII``, ``LaTeX`` or ``Unicode``).
    encoding : {"LSB", "MSB"}, optional
        Bit-ordering convention for the basis kets (default: ``"MSB"``).
        See :meth:`graphix.sim.statevec.Statevector.to_dict`.
    max_denominator : int, optional
        Maximum denominator used by the amplitude recognition (default: ``1000``).
    atol : float, optional
        Absolute tolerance used both to drop near-zero amplitudes and for the
        recognition heuristics (default: ``1e-9``).
    rtol : float, optional
        Relative tolerance used both to drop near-zero amplitudes and for the
        recognition heuristics (default: ``0.0``).
    precision : int, optional
        Number of significant digits to use for amplitudes that fall back to a
        decimal representation (default: ``4``).

    Returns
    -------
    str
        The formatted statevector, e.g. ``√2/2(|00⟩ + |01⟩)``.
    """
    amplitudes = statevec.to_dict(encoding, rtol=rtol, atol=atol)
    if not amplitudes:
        return "0"
    amps = list(amplitudes.values())
    kets = [_ket_str(ket, output) for ket in amplitudes]
    # When every amplitude shares a modulus (up to a relative phase), factor it out,
    # e.g. ``√2/2(|0⟩ + i|1⟩)``.
    factored = _factor_uniform_magnitude(
        amps, kets, output, max_denominator=max_denominator, atol=atol, rtol=rtol, precision=precision
    )
    if factored is not None:
        return factored
    coefficients = [
        complex_to_str(amplitude, output, max_denominator=max_denominator, atol=atol, rtol=rtol, precision=precision)
        for amplitude in amps
    ]
    result = ""
    for index, (coefficient, ket_str) in enumerate(zip(coefficients, kets, strict=True)):
        if coefficient == "1":
            term = ket_str
        elif coefficient == "-1":
            term = f"-{ket_str}"
        elif _needs_parentheses(coefficient):
            term = f"({coefficient}){ket_str}"
        else:
            term = f"{coefficient}{ket_str}"
        if index == 0:
            result = term
        elif term.startswith("-"):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    return result


def density_matrix_to_str(
    density_matrix: DensityMatrix,
    output: OutputFormat,
    *,
    max_denominator: int = _DEFAULT_MAX_DENOMINATOR,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
    precision: int = _DEFAULT_PRECISION,
) -> str:
    r"""Return a matrix-form string representation of a density matrix.

    Each entry is pretty-printed with :func:`complex_to_str`. ``LaTeX`` output
    uses a ``pmatrix`` environment; ``ASCII`` and ``Unicode`` outputs produce a
    column-aligned grid.

    Parameters
    ----------
    density_matrix : DensityMatrix
        The density matrix to format.
    output : OutputFormat
        Desired formatting style (``ASCII``, ``LaTeX`` or ``Unicode``).
    max_denominator : int, optional
        Maximum denominator used by the entry recognition (default: ``1000``).
    atol : float, optional
        Absolute tolerance for the recognition heuristics (default: ``1e-9``).
    rtol : float, optional
        Relative tolerance for the recognition heuristics (default: ``0.0``).
    precision : int, optional
        Number of significant digits to use for entries that fall back to a
        decimal representation (default: ``4``).

    Returns
    -------
    str
        The formatted density matrix.
    """
    rows = [
        [
            complex_to_str(entry, output, max_denominator=max_denominator, atol=atol, rtol=rtol, precision=precision)
            for entry in row
        ]
        for row in density_matrix.rho
    ]
    if output == OutputFormat.LaTeX:
        body = r" \\ ".join(" & ".join(row) for row in rows)
        return rf"\begin{{pmatrix}}{body}\end{{pmatrix}}"
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))] if rows else []
    lines = ["  ".join(entry.rjust(widths[col]) for col, entry in enumerate(row)) for row in rows]
    return "\n".join(f"[ {line} ]" for line in lines)

"""Pretty-printing utilities."""

from __future__ import annotations

import enum
import math
import string
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, SupportsFloat

from graphix import command

if TYPE_CHECKING:
    from collections.abc import Container

    from graphix.command import Node
    from graphix.pattern import Pattern


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


def domain_to_str(domain: set[Node]) -> str:
    """Return the string representation of a domain."""
    return f"{{{','.join(str(node) for node in domain)}}}"


SUBSCRIPTS = str.maketrans(string.digits, "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPTS = str.maketrans(string.digits, "⁰¹²³⁴⁵⁶⁷⁸⁹")


def command_to_str(cmd: command.Command, output: OutputFormat) -> str:
    """Return the string representation of a command according to the given format.

    Parameters
    ----------
    cmd: Command
        The command to pretty print.
    output: OutputFormat
        The expected format.
    """
    # Circumvent circular import
    from graphix.fundamentals import Plane

    out = [cmd.kind.name]

    if cmd.kind == command.CommandKind.E:
        u, v = cmd.nodes
        if output == OutputFormat.LaTeX:
            out.append(f"_{{{u},{v}}}")
        elif output == OutputFormat.Unicode:
            u_subscripts = str(u).translate(SUBSCRIPTS)
            v_subscripts = str(v).translate(SUBSCRIPTS)
            out.append(f"{u_subscripts}₋{v_subscripts}")
        else:
            out.append(f"({u},{v})")
    elif cmd.kind == command.CommandKind.T:
        pass
    else:
        # All other commands have a field `node` to print, together
        # with some other arguments and/or domains.
        arguments = []
        if cmd.kind == command.CommandKind.M:
            if cmd.plane != Plane.XY:
                arguments.append(cmd.plane.name)
            # We use `SupportsFloat` since `isinstance(cmd.angle, float)`
            # is `False` if `cmd.angle` is an integer.
            if isinstance(cmd.angle, SupportsFloat):
                angle = float(cmd.angle)
                if not math.isclose(angle, 0.0):
                    arguments.append(angle_to_str(angle, output))
            else:
                # If the angle is a symbolic expression, we can only delegate the printing
                # TODO: We should have a mean to specify the format
                arguments.append(str(cmd.angle * math.pi))
        elif cmd.kind == command.CommandKind.C:
            arguments.append(str(cmd.clifford))
        # Use of `==` here for mypy
        command_domain = (
            cmd.domain
            if cmd.kind == command.CommandKind.X  # noqa: PLR1714
            or cmd.kind == command.CommandKind.Z
            or cmd.kind == command.CommandKind.S
            else None
        )
        if output == OutputFormat.LaTeX:
            out.append(f"_{{{cmd.node}}}")
            if arguments:
                out.append(f"^{{{','.join(arguments)}}}")
        elif output == OutputFormat.Unicode:
            node_subscripts = str(cmd.node).translate(SUBSCRIPTS)
            out.append(f"{node_subscripts}")
            if arguments:
                out.append(f"({','.join(arguments)})")
        else:
            arguments = [str(cmd.node), *arguments]
            if command_domain:
                arguments.append(domain_to_str(command_domain))
                command_domain = None
            out.append(f"({','.join(arguments)})")
        if cmd.kind == command.CommandKind.M and (cmd.s_domain or cmd.t_domain):
            out = ["[", *out, "]"]
            if cmd.t_domain:
                if output == OutputFormat.LaTeX:
                    t_domain_str = f"{{}}_{{{','.join(str(node) for node in cmd.t_domain)}}}"
                elif output == OutputFormat.Unicode:
                    t_domain_subscripts = [str(node).translate(SUBSCRIPTS) for node in cmd.t_domain]
                    t_domain_str = "₊".join(t_domain_subscripts)
                else:
                    t_domain_str = f"{{{','.join(str(node) for node in cmd.t_domain)}}}"
                out = [t_domain_str, *out]
            command_domain = cmd.s_domain
        if command_domain:
            if output == OutputFormat.LaTeX:
                domain_str = f"^{{{','.join(str(node) for node in command_domain)}}}"
            elif output == OutputFormat.Unicode:
                domain_superscripts = [str(node).translate(SUPERSCRIPTS) for node in command_domain]
                domain_str = "⁺".join(domain_superscripts)
            else:
                domain_str = f"{{{','.join(str(node) for node in command_domain)}}}"
            out.append(domain_str)
    return f"{''.join(out)}"


def pattern_to_str(
    pattern: Pattern,
    output: OutputFormat,
    left_to_right: bool = False,
    limit: int = 40,
    target: Container[command.CommandKind] | None = None,
) -> str:
    """Return the string representation of a pattern according to the given format.

    Parameters
    ----------
    pattern: Pattern
        The pattern to pretty print.
    output: OutputFormat
        The expected format.
    left_to_right: bool
        Optional. If `True`, the first command will appear on the beginning of
        the resulting string. If `False` (the default), the first command will
        appear at the end of the string.
    """
    separator = r"\," if output == OutputFormat.LaTeX else " "
    command_list = list(pattern)
    if target is not None:
        command_list = [command for command in command_list if command.kind in target]
    if not left_to_right:
        command_list.reverse()
    truncated = len(command_list) > limit
    short_command_list = command_list[: limit - 1] if truncated else command_list
    result = separator.join(command_to_str(command, output) for command in short_command_list)
    if output == OutputFormat.LaTeX:
        result = f"\\({result}\\)"
    if truncated:
        return f"{result}...({len(command_list) - limit + 1} more commands)"
    return result

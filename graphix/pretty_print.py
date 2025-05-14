"""Pretty-printing utilities."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import MISSING
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, SupportsFloat

from graphix import command
from graphix.fundamentals import Plane

if TYPE_CHECKING:
    from collections.abc import Container

    # these live only in the stub package, not at runtime
    from _typeshed import DataclassInstance

    from graphix.command import Node
    from graphix.pattern import Pattern


class OutputFormat(Enum):
    """Enumeration of the output format for pretty-printing."""

    ASCII = "ASCII"
    LaTeX = "LaTeX"
    Unicode = "Unicode"


def angle_to_str(angle: float, output: OutputFormat) -> str:
    """Return the string of an angle according to the given format.

    Parameters
    ----------
    angle: float
        The input angle, in unit of π.
    output: OutputFormat
        The expected format.
    """
    # We check whether the angle is close to a "simple" fraction of π,
    # where "simple" is defined has "having a denominator less than
    # or equal to 1000.

    frac = Fraction(angle).limit_denominator(1000)

    if not math.isclose(angle, float(frac)):
        rad = angle * math.pi

        return f"{rad:.2f}"

    num, den = frac.numerator, frac.denominator
    sign = "-" if num < 0 else ""
    num = abs(num)

    if output == OutputFormat.LaTeX:
        pi = r"\pi"

        def mkfrac(num: str, den: str) -> str:
            return f"\\frac{{{num}}}{{{den}}}"
    else:
        pi = "π" if output == OutputFormat.Unicode else "pi"

        def mkfrac(num: str, den: str) -> str:
            return f"{num}/{den}"

    if den == 1:
        if num == 1:
            return f"{sign}{pi}"
        return f"{sign}{num}{pi}"

    den_str = f"{den}"
    num_str = pi if num == 1 else f"{num}{pi}"
    return f"{sign}{mkfrac(num_str, den_str)}"


def domain_to_str(domain: set[Node]) -> str:
    """Return the string representation of a domain."""
    return f"{{{','.join(str(node) for node in domain)}}}"


SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def command_to_str(cmd: command.Command, output: OutputFormat) -> str:
    """Return the string representation of a command according to the given format.

    Parameters
    ----------
    cmd: Command
        The command to pretty print.
    output: OutputFormat
        The expected format.
    """
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
    if target:
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


def pretty_repr_dataclass(instance: DataclassInstance, unit_of_pi: bool = False) -> str:
    """Return a representation string for a dataclass."""
    cls_name = type(instance).__name__
    arguments = []
    saw_omitted = False
    for field in dataclasses.fields(instance):
        if field.name == "kind":
            continue
        value = getattr(instance, field.name)
        if field.default is not MISSING or field.default_factory is not MISSING:
            default = field.default_factory() if field.default_factory is not MISSING else field.default
            if value == default:
                saw_omitted = True
                continue
        if field.name == "angle" and not unit_of_pi:
            angle: float = instance.angle  # type: ignore[attr-defined]
            value_str = angle_to_str(angle / math.pi, OutputFormat.ASCII)
        else:
            value_str = repr(value)
        if saw_omitted:
            arguments.append(f"{field.name}={value_str}")
        else:
            arguments.append(value_str)
    arguments_str = ", ".join(arguments)
    return f"{cls_name}({arguments_str})"


def pretty_repr_enum(value: Enum) -> str:
    """Return a "pretty" representation of an Enum member.

    This returns the display string for `value` in the form
    `ClassName.MEMBER_NAME`, which is a valid Python expression
    (assuming `ClassName` is in scope) to recreate that member.

    By contrast, the default Enum repr is
    `<ClassName.MEMBER_NAME: value>`, which isn't directly
    evaluable as Python code.

    Parameters
    ----------
        value: An instance of `Enum`.

    Returns
    -------
        A string in the form `ClassName.MEMBER_NAME`.
    """
    # Equivalently (as of Python 3.12), `str(value)` also produces
    # "ClassName.MEMBER_NAME", but we build it explicitly here for
    # clarity.
    return f"{value.__class__.__name__}.{value.name}"

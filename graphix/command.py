"""Data validator command classes."""

from __future__ import annotations

import dataclasses
import enum
import math
import string
import sys
from enum import Enum
from typing import ClassVar, Literal, SupportsFloat, Union

import numpy as np

from graphix import utils
from graphix.clifford import Clifford
from graphix.fundamentals import Plane, Sign
from graphix.measurements import Domains

# Ruff suggests to move this import to a type-checking block, but dataclass requires it here
from graphix.parameter import ExpressionOrFloat  # noqa: TC001
from graphix.pauli import Pauli
from graphix.pretty_print import DataclassPrettyPrintMixin, OutputFormat, angle_to_str, domain_to_str
from graphix.states import BasicStates, State

Node = int


class CommandKind(Enum):
    """Tag for command kind."""

    N = enum.auto()
    M = enum.auto()
    E = enum.auto()
    C = enum.auto()
    X = enum.auto()
    Z = enum.auto()
    S = enum.auto()
    T = enum.auto()


SUBSCRIPTS = str.maketrans(string.digits, "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPTS = str.maketrans(string.digits, "⁰¹²³⁴⁵⁶⁷⁸⁹")


def command_to_str(cmd: Command, output: OutputFormat) -> str:
    """Return the string representation of a command according to the given format.

    Parameters
    ----------
    cmd: Command
        The command to pretty print.
    output: OutputFormat
        The expected format.
    """
    out = [cmd.kind.name]

    if cmd.kind == CommandKind.E:
        u, v = cmd.nodes
        if output == OutputFormat.LaTeX:
            out.append(f"_{{{u},{v}}}")
        elif output == OutputFormat.Unicode:
            u_subscripts = str(u).translate(SUBSCRIPTS)
            v_subscripts = str(v).translate(SUBSCRIPTS)
            out.append(f"{u_subscripts}₋{v_subscripts}")
        else:
            out.append(f"({u},{v})")
    elif cmd.kind == CommandKind.T:
        pass
    else:
        # All other commands have a field `node` to print, together
        # with some other arguments and/or domains.
        arguments = []
        if cmd.kind == CommandKind.M:
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
        elif cmd.kind == CommandKind.C:
            arguments.append(str(cmd.clifford))
        # Use of `==` here for mypy
        command_domain = (
            cmd.domain
            if cmd.kind == CommandKind.X  # noqa: PLR1714
            or cmd.kind == CommandKind.Z
            or cmd.kind == CommandKind.S
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
        if cmd.kind == CommandKind.M and (cmd.s_domain or cmd.t_domain):
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


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        utils.check_kind(cls, {"CommandKind": CommandKind, "Clifford": Clifford})


@dataclasses.dataclass(repr=False)
class N(_KindChecker, DataclassPrettyPrintMixin):
    r"""Preparation command.

    Parameters
    ----------
    node : int
        Index of the qubit to prepare.
    state : ~graphix.states.State, optional
        Initial state, defaults to :class:`~graphix.states.BasicStates.PLUS`.
    """

    node: Node
    state: State = dataclasses.field(default_factory=lambda: BasicStates.PLUS)
    kind: ClassVar[Literal[CommandKind.N]] = dataclasses.field(default=CommandKind.N, init=False)


@dataclasses.dataclass(repr=False)
class M(_KindChecker, DataclassPrettyPrintMixin):
    r"""Measurement command.

    Parameters
    ----------
    node : int
        Node index of the measured qubit.
    plane : Plane, optional
        Measurement plane, defaults to :class:`~graphix.fundamentals.Plane.XY`.
    angle : ExpressionOrFloat, optional
        Rotation angle divided by :math:`\pi`.
    s_domain : set[int], optional
        Domain for the X byproduct operator.
    t_domain : set[int], optional
        Domain for the Z byproduct operator.
    """

    node: Node
    plane: Plane = Plane.XY
    angle: ExpressionOrFloat = 0.0
    s_domain: set[Node] = dataclasses.field(default_factory=set)
    t_domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.M]] = dataclasses.field(default=CommandKind.M, init=False)

    def clifford(self, clifford_gate: Clifford) -> M:
        r"""Return a new measurement command with a Clifford applied.

        Parameters
        ----------
        clifford_gate : ~graphix.clifford.Clifford
            Clifford gate to apply before the measurement.

        Returns
        -------
        :class:`~graphix.command.M`
            Equivalent command representing the pattern ``MC``.
        """
        domains = clifford_gate.commute_domains(Domains(self.s_domain, self.t_domain))
        update = MeasureUpdate.compute(self.plane, False, False, clifford_gate)
        return M(
            self.node,
            update.new_plane,
            self.angle * update.coeff + update.add_term / np.pi,
            domains.s_domain,
            domains.t_domain,
        )


@dataclasses.dataclass(repr=False)
class E(_KindChecker, DataclassPrettyPrintMixin):
    r"""Entanglement command between two qubits.

    Parameters
    ----------
    nodes : tuple[int, int]
        Pair of nodes to entangle.
    """

    nodes: tuple[Node, Node]
    kind: ClassVar[Literal[CommandKind.E]] = dataclasses.field(default=CommandKind.E, init=False)


@dataclasses.dataclass(repr=False)
class C(_KindChecker, DataclassPrettyPrintMixin):
    r"""Local Clifford gate command.

    Parameters
    ----------
    node : int
        Node index on which to apply the gate.
    clifford : ~graphix.clifford.Clifford
        Clifford operator to apply.
    """

    node: Node
    clifford: Clifford
    kind: ClassVar[Literal[CommandKind.C]] = dataclasses.field(default=CommandKind.C, init=False)


@dataclasses.dataclass(repr=False)
class X(_KindChecker, DataclassPrettyPrintMixin):
    r"""X correction command.

    Parameters
    ----------
    node : int
        Node to correct.
    domain : set[int], optional
        Domain for the byproduct operator.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.X]] = dataclasses.field(default=CommandKind.X, init=False)


@dataclasses.dataclass(repr=False)
class Z(_KindChecker, DataclassPrettyPrintMixin):
    r"""Z correction command.

    Parameters
    ----------
    node : int
        Node to correct.
    domain : set[int], optional
        Domain for the byproduct operator.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.Z]] = dataclasses.field(default=CommandKind.Z, init=False)


@dataclasses.dataclass(repr=False)
class S(_KindChecker, DataclassPrettyPrintMixin):
    r"""S command.

    Parameters
    ----------
    node : int
        Node for the byproduct operator.
    domain : set[int], optional
        Domain on which to apply the operator.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.S]] = dataclasses.field(default=CommandKind.S, init=False)


@dataclasses.dataclass(repr=False)
class T(_KindChecker):
    r"""T command.

    Parameters
    ----------
    None
        The T command acts globally without parameters.
    """

    kind: ClassVar[Literal[CommandKind.T]] = dataclasses.field(default=CommandKind.T, init=False)


if sys.version_info >= (3, 10):
    Command = N | M | E | C | X | Z | S | T
    Correction = X | Z
else:
    Command = Union[N, M, E, C, X, Z, S, T]
    Correction = Union[X, Z]

BaseM = M


@dataclasses.dataclass
class MeasureUpdate:
    r"""Describe how a measure is changed by signals and a vertex operator.

    Parameters
    ----------
    new_plane : Plane
        Updated measurement plane after commuting gates.
    coeff : int
        Coefficient by which the angle is multiplied.
    add_term : float
        Additional term to add to the measurement angle.
    """

    new_plane: Plane
    coeff: int
    add_term: float

    @staticmethod
    def compute(plane: Plane, s: bool, t: bool, clifford_gate: Clifford) -> MeasureUpdate:
        r"""Compute the measurement update.

        Parameters
        ----------
        plane : ~graphix.fundamentals.Plane
            Measurement plane of the command.
        s : bool
            Whether an :math:`X` signal is present.
        t : bool
            Whether a :math:`Z` signal is present.
        clifford_gate : ~graphix.clifford.Clifford
            Vertex operator applied before the measurement.

        Returns
        -------
        MeasureUpdate
            Update describing the new measurement.
        """
        gates = list(map(Pauli.from_axis, plane.axes))
        if s:
            clifford_gate = Clifford.X @ clifford_gate
        if t:
            clifford_gate = Clifford.Z @ clifford_gate
        gates = list(map(clifford_gate.measure, gates))
        new_plane = Plane.from_axes(*(gate.axis for gate in gates))
        cos_pauli = clifford_gate.measure(Pauli.from_axis(plane.cos))
        sin_pauli = clifford_gate.measure(Pauli.from_axis(plane.sin))
        exchange = cos_pauli.axis != new_plane.cos
        coeff = -1 if exchange == (cos_pauli.unit.sign == sin_pauli.unit.sign) else 1
        add_term: float = 0
        if cos_pauli.unit.sign == Sign.MINUS:
            add_term += np.pi
        if exchange:
            add_term = np.pi / 2 - add_term
        return MeasureUpdate(new_plane, coeff, add_term)

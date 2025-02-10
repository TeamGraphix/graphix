"""Data validator command classes."""

from __future__ import annotations

import dataclasses
import enum
import sys
from enum import Enum
from typing import ClassVar, Literal, Union

import numpy as np

from graphix import utils
from graphix.clifford import Clifford
from graphix.fundamentals import Plane, Sign
from graphix.measurements import Domains
from graphix.pauli import Pauli
from graphix.states import BasicStates, State

Node = int


def command_to_latex(cmd: N | M | E | C | X | Z | S | T) -> str:
    """Get the latex string representation of a command."""
    kind = cmd.kind
    out = kind.name

    if kind == CommandKind.N:
        out += "_{" + str(cmd.node) + "}"
    if kind == CommandKind.M:
        out += "_" + str(cmd.node) + "^{" + cmd.plane.name + "," + str(round(cmd.angle, 2)) + "}"
    if kind == CommandKind.E:
        out += "_{" + str(cmd.nodes[0]) + "," + str(cmd.nodes[1]) + "}"
    if kind == CommandKind.C:
        out += "_" + str(cmd.node)
    if kind in {CommandKind.X, CommandKind.Z, CommandKind.S, CommandKind.T}:
        out += "_" + str(cmd.node) + "^{[" + "".join([str(dom) for dom in cmd.domain]) + "]}"

    return "$" + out + "$"


def command_to_str(cmd: N | M | E | C | X | Z | S | T) -> str:
    """Get the string representation of a command."""
    kind = cmd.kind
    out = kind.name

    if kind == CommandKind.N:
        out += "(" + str(cmd.node) + ")"
    if kind == CommandKind.M:
        out += "(" + str(cmd.node) + "," + cmd.plane.name + "," + str(round(cmd.angle, 2)) + ")"
    if kind == CommandKind.E:
        out += "(" + str(cmd.nodes[0]) + "," + str(cmd.nodes[1]) + ")"
    if kind == CommandKind.C:
        out += "(" + str(cmd.node)
    if kind in {CommandKind.X, CommandKind.Z, CommandKind.S, CommandKind.T}:
        out += "(" + str(cmd.node) + ")"

    return out


def command_to_unicode(cmd: N | M | E | C | X | Z | S | T) -> str:
    """Get the unicode representation of a command."""
    kind = cmd.kind
    out = kind.name

    subscripts = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

    def _get_subscript_from_number(number: int) -> str:
        strnum = str(number)
        if len(strnum) == 0:
            return ""
        if len(strnum) == 1:
            return subscripts[int(number)]
        sub = int(strnum[0])
        next_sub = strnum[1:]
        return subscripts[sub] + _get_subscript_from_number(int(next_sub))

    if kind == CommandKind.N:
        out += _get_subscript_from_number(cmd.node)
    if kind == CommandKind.M:
        out += _get_subscript_from_number(cmd.node)
    if kind == CommandKind.E:
        out += _get_subscript_from_number(cmd.nodes[0]) + _get_subscript_from_number(cmd.nodes[1])
    if kind == CommandKind.C:
        out += _get_subscript_from_number(cmd.node)
    if kind in {CommandKind.X, CommandKind.Z, CommandKind.S, CommandKind.T}:
        out += _get_subscript_from_number(cmd.node)

    return out


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


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        utils.check_kind(cls, {"CommandKind": CommandKind, "Clifford": Clifford})

    def to_latex(self) -> str:
        return command_to_latex(self)

    def to_unicode(self) -> str:
        return command_to_unicode(self)

    def __str__(self) -> str:
        return command_to_str(self)


@dataclasses.dataclass
class N(_KindChecker):
    """Preparation command."""

    node: Node
    state: State = dataclasses.field(default_factory=lambda: BasicStates.PLUS)
    kind: ClassVar[Literal[CommandKind.N]] = dataclasses.field(default=CommandKind.N, init=False)


@dataclasses.dataclass
class M(_KindChecker):
    """Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop."""

    node: Node
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = dataclasses.field(default_factory=set)
    t_domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.M]] = dataclasses.field(default=CommandKind.M, init=False)

    def clifford(self, clifford_gate: Clifford) -> M:
        """Apply a Clifford gate to the measure command.

        The returned `M` command is equivalent to the pattern `MC`.
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


@dataclasses.dataclass
class E(_KindChecker):
    """Entanglement command."""

    nodes: tuple[Node, Node]
    kind: ClassVar[Literal[CommandKind.E]] = dataclasses.field(default=CommandKind.E, init=False)


@dataclasses.dataclass
class C(_KindChecker):
    """Clifford command."""

    node: Node
    clifford: Clifford
    kind: ClassVar[Literal[CommandKind.C]] = dataclasses.field(default=CommandKind.C, init=False)


@dataclasses.dataclass
class X(_KindChecker):
    """X correction command."""

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.X]] = dataclasses.field(default=CommandKind.X, init=False)


@dataclasses.dataclass
class Z(_KindChecker):
    """Z correction command."""

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.Z]] = dataclasses.field(default=CommandKind.Z, init=False)


@dataclasses.dataclass
class S(_KindChecker):
    """S command."""

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.S]] = dataclasses.field(default=CommandKind.S, init=False)


@dataclasses.dataclass
class T(_KindChecker):
    """T command."""

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
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
    """Describe how a measure is changed by the signals and/or a vertex operator."""

    new_plane: Plane
    coeff: int
    add_term: float

    @staticmethod
    def compute(plane: Plane, s: bool, t: bool, clifford_gate: Clifford) -> MeasureUpdate:
        """Compute the update for a given plane, signals and vertex operator."""
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

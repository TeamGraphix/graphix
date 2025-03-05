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

# Ruff suggests to move this import to a type-checking block, but dataclass requires it here
from graphix.parameter import ExpressionOrFloat  # noqa: TC001
from graphix.pauli import Pauli
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


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        utils.check_kind(cls, {"CommandKind": CommandKind, "Clifford": Clifford})


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
    angle: ExpressionOrFloat = 0.0
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

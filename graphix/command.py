"""Data validator command classes."""

from __future__ import annotations

import abc
import enum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from graphix import clifford
from graphix.clifford import Domains
from graphix.pauli import Pauli, Plane, Sign
from graphix.states import BasicStates, State

if TYPE_CHECKING:
    from graphix.clifford import Clifford

Node = int


class CommandKind(enum.Enum):
    """Tag for command kind."""

    N = "N"
    M = "M"
    E = "E"
    C = "C"
    X = "X"
    Z = "Z"
    T = "T"
    S = "S"


class Command(BaseModel, abc.ABC):
    """Base command class."""

    kind: CommandKind = None


class N(Command):
    """Preparation command."""

    kind: CommandKind = CommandKind.N
    node: Node
    state: State = BasicStates.PLUS


class BaseM(Command):
    """Base measurement command.

    Represent a measurement of a node. In MBQC, a measure is an instance of `M`,
    with given plane, angles, and domains. In the context of blind computations,
    the server only knows which node is measured, and the parameters are given
    by the :class:`graphix.simulator.MeasureMethod` provided by the client.
    """

    kind: CommandKind = CommandKind.M
    node: Node


class M(BaseM):
    """Measurement command.

    By default the plane is set to 'XY', the angle to 0, empty domains.
    """

    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = set()
    t_domain: set[Node] = set()

    def clifford(self, clifford_gate: Clifford) -> M:
        """Apply a Clifford gate to the measure command.

        The returned `M` command is equivalent to the pattern `MC`.
        """
        domains = clifford_gate.commute_domains(Domains(self.s_domain, self.t_domain))
        update = MeasureUpdate.compute(self.plane, False, False, clifford_gate)
        return M(
            node=self.node,
            plane=update.new_plane,
            angle=self.angle * update.coeff + update.add_term / np.pi,
            s_domain=domains.s_domain,
            t_domain=domains.t_domain,
        )


class E(Command):
    """Entanglement command."""

    kind: CommandKind = CommandKind.E
    nodes: tuple[Node, Node]


class C(Command):
    """Clifford command."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for the `clifford` field

    kind: CommandKind = CommandKind.C
    node: Node
    clifford: clifford.Clifford


class Correction(Command):
    """Base class for correction command (either X or Z)."""

    node: Node
    domain: set[Node] = set()


class X(Correction):
    """X correction command."""

    kind: CommandKind = CommandKind.X


class Z(Correction):
    """Z correction command."""

    kind: CommandKind = CommandKind.Z


class S(Command):
    """S command."""

    kind: CommandKind = CommandKind.S
    node: Node
    domain: set[Node] = set()


class T(Command):
    """T command."""

    kind: CommandKind = CommandKind.T


class MeasureUpdate(BaseModel):
    """Describe how a measure is changed by the signals and/or a vertex operator."""

    new_plane: Plane
    coeff: int
    add_term: float

    @staticmethod
    def compute(plane: Plane, s: bool, t: bool, clifford_gate: Clifford) -> MeasureUpdate:
        """Compute the update for a given plane, signals and vertex operator."""
        gates = list(map(Pauli.from_axis, plane.axes))
        if s:
            clifford_gate = clifford.X @ clifford_gate
        if t:
            clifford_gate = clifford.Z @ clifford_gate
        gates = list(map(clifford_gate.measure, gates))
        new_plane = Plane.from_axes(*(gate.axis for gate in gates))
        cos_pauli = clifford_gate.measure(Pauli.from_axis(plane.cos))
        sin_pauli = clifford_gate.measure(Pauli.from_axis(plane.sin))
        exchange = cos_pauli.axis != new_plane.cos
        if exchange == (cos_pauli.unit.sign == sin_pauli.unit.sign):
            coeff = -1
        else:
            coeff = 1
        add_term: float = 0
        if cos_pauli.unit.sign == Sign.Minus:
            add_term += np.pi
        if exchange:
            add_term = np.pi / 2 - add_term
        return MeasureUpdate(new_plane=new_plane, coeff=coeff, add_term=add_term)

"""Data validator command classes."""

from __future__ import annotations

import abc
import enum

import numpy as np
from pydantic import BaseModel

import graphix.clifford
from graphix.clifford import Clifford
from graphix.pauli import Plane

Node = int


class CommandKind(enum.Enum):
    N = "N"
    M = "M"
    E = "E"
    C = "C"
    X = "X"
    Z = "Z"
    T = "T"
    S = "S"


class Command(BaseModel, abc.ABC):
    """
    Base command class.
    """

    kind: CommandKind = None


class N(Command):
    """
    Preparation command.
    """

    kind: CommandKind = CommandKind.N
    node: Node


class M(Command):
    """
    Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop.
    """

    kind: CommandKind = CommandKind.M
    node: Node
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: list[Node] = []
    t_domain: list[Node] = []

    def clifford(self, clifford: Clifford) -> M:
        s_domain = self.s_domain
        t_domain = self.t_domain
        for gate in clifford.hsz:
            if gate == graphix.clifford.I:
                pass
            elif gate == graphix.clifford.H:
                t_domain, s_domain = s_domain, t_domain
            elif gate == graphix.clifford.S:
                t_domain = s_domain + t_domain
            elif gate == graphix.clifford.Z:
                pass
            else:
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        update = graphix.pauli.MeasureUpdate.compute(self.plane, False, False, clifford)
        result = M(
            node=self.node,
            plane=update.new_plane,
            angle=self.angle * update.coeff + update.add_term / np.pi,
            s_domain=s_domain,
            t_domain=t_domain,
        )
        return result


class E(Command):
    """
    Entanglement command.
    """

    kind: CommandKind = CommandKind.E
    nodes: tuple[Node, Node]


class C(Command):
    """
    Clifford command.
    """

    kind: CommandKind = CommandKind.C
    node: Node
    cliff_index: int


class Correction(Command):
    """
    Correction command.
    Either X or Z.
    """

    node: Node
    domain: list[Node] = []


class X(Correction):
    """
    X correction command.
    """

    kind: CommandKind = CommandKind.X


class Z(Correction):
    """
    Z correction command.
    """

    kind: CommandKind = CommandKind.Z


class S(Command):
    """
    S command
    """

    kind: CommandKind = CommandKind.S
    node: Node
    domain: list[Node] = []


class T(Command):
    """
    T command
    """

    kind: CommandKind = CommandKind.T

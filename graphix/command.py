"""Data validator command classes."""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

import numpy as np

import graphix.clifford
from graphix import utils
from graphix.pauli import MeasureUpdate, Plane

if TYPE_CHECKING:
    from graphix.clifford import Clifford

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


@utils.disable_init
class Command:
    """
    Base command class.
    """


@dataclasses.dataclass
class N(Command):
    """
    Preparation command.
    """

    node: Node


@dataclasses.dataclass
class M(Command):
    """
    Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop.
    """

    node: Node
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = dataclasses.field(default_factory=set)
    t_domain: set[Node] = dataclasses.field(default_factory=set)

    def clifford(self, clifford: Clifford) -> M:
        s_domain = self.s_domain
        t_domain = self.t_domain
        for gate in clifford.hsz:
            if gate == graphix.clifford.I:
                pass
            elif gate == graphix.clifford.H:
                t_domain, s_domain = s_domain, t_domain
            elif gate == graphix.clifford.S:
                t_domain ^= s_domain
            elif gate == graphix.clifford.Z:
                pass
            else:
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        update = MeasureUpdate.compute(self.plane, False, False, clifford)
        return M(
            self.node,
            update.new_plane,
            self.angle * update.coeff + update.add_term / np.pi,
            s_domain,
            t_domain,
        )


@dataclasses.dataclass
class E(Command):
    """
    Entanglement command.
    """

    nodes: tuple[Node, Node]


@dataclasses.dataclass
class C(Command):
    """
    Clifford command.
    """

    node: Node
    cliff_index: int


@dataclasses.dataclass
class Correction(Command):
    """
    Correction command.
    Either X or Z.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class X(Correction):
    """
    X correction command.
    """


@dataclasses.dataclass
class Z(Correction):
    """
    Z correction command.
    """


@dataclasses.dataclass
class S(Command):
    """
    S command
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class T(Command):
    """
    T command
    """

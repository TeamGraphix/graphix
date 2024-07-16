"""Data validator command classes."""

from __future__ import annotations

import abc
import enum

from pydantic import BaseModel

from graphix.clifford import Clifford
from graphix.pauli import Plane
from graphix.states import BasicStates, State

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
    state: State = BasicStates.PLUS


class BaseM(Command):
    kind: CommandKind = CommandKind.M
    node: Node


class M(BaseM):
    """
    Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop.
    """

    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: list[Node] = []
    t_domain: list[Node] = []
    vop: int = 0


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
    clifford: Clifford


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

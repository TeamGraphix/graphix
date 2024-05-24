"""Data validator command classes."""

from pydantic import BaseModel
from typing import Union, Literal, List, Tuple
import enum

Node = int
Plane = Union[Literal["XY"], Literal["YZ"], Literal["XZ"]]


class CommandKind(str, enum.Enum):
    N = "N"
    M = "M"
    E = "E"
    C = "C"
    X = "X"
    Z = "Z"
    T = "T"
    S = "S"


class Command(BaseModel):
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
    plane: Plane = "XY"
    angle: float = 0.0
    s_domain: List[Node] = []
    t_domain: List[Node] = []
    vop: int = 0


class E(Command):
    """
    Entanglement command.
    """

    kind: CommandKind = CommandKind.E
    nodes: Tuple[Node, Node]


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
    domain: List[Node] = []


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
    domain: List[Node] = []


class T(Command):
    """
    T command
    """

    kind: CommandKind = CommandKind.T

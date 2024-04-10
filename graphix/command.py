"""Data validator command classes."""

from pydantic import BaseModel
from typing import Union, Literal

Node = int
Plane = Union[Literal["XY"], Literal["YZ"], Literal["XZ"]]
Name = Union[
    Literal["N"], Literal["M"], Literal["E"], Literal["X"], Literal["Z"], Literal["C"]
]


class Command(BaseModel):
    """
    Base command class.
    """

    pass


class N(Command):
    """
    Preparation command.
    """

    node: Node

    @property
    def name(self):
        return "N"

    def __lt__(self, other):
        return self.node < other.node


class M(Command):
    """
    Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop.
    """

    node: Node
    plane: Plane = "XY"
    angle: float = 0.0
    s_domain: list[Node] = []
    t_domain: list[Node] = []
    vop: int = 0

    @property
    def name(self):
        return "M"


class E(Command):
    """
    Entanglement command.
    """

    nodes: tuple[Node, Node]

    @property
    def name(self):
        return "E"


class C(Command):
    """
    Clifford command.
    """

    node: Node
    cliff_index: int

    @property
    def name(self):
        return "C"


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

    @property
    def name(self):
        return "X"


class Z(Correction):
    """
    Z correction command.
    """

    @property
    def name(self):
        return "Z"


class S(Command):
    """
    S command.s
    """

    node: Node
    domain: list[Node] = []

    @property
    def name(self):
        return "S"


class T(Command):
    pass

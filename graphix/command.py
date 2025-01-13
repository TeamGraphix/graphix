"""Data validator command classes."""

from __future__ import annotations

import abc
import enum

from pydantic import BaseModel

from graphix.pauli import Plane

Node = int

def _command_to_latex(cmd: Command) -> str:
    kind = cmd.kind
    out = kind.name

    if kind == CommandKind.N:
        out += '_{' + str(cmd.node) + '}'
    if kind == CommandKind.M:
        out += '_' + str(cmd.node) + '^{' + cmd.plane.name + ',' + str(round(cmd.angle, 2)) + '}'
    if kind == CommandKind.E:
        out += '_{' + str(cmd.nodes[0]) + ',' + str(cmd.nodes[1]) + '}'
    if kind == CommandKind.C:
        out += '_' + str(cmd.node) + '^{' + ''.join(cmd.domain) + '}'
    
    return '$' + out + '$'

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

    def to_latex(self) -> str:
        return _command_to_latex(self)


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

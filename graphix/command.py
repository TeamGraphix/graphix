"""Data validator command classes."""

from __future__ import annotations

import abc
import enum
import math
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

import graphix.clifford
from graphix.clifford import Clifford
from graphix.pauli import Axis, Plane, Sign

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


class Pauli(BaseModel):
    """
    Pauli measurement.

    Pauli measurement is not a pattern command in itself, but can be obtained from a general `M` measurement with `is_close_to_pauli`.
    """

    axis: Axis
    sign: Sign


class M(Command):
    """
    Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop.
    """

    kind: CommandKind = CommandKind.M
    node: Node
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = set()
    t_domain: set[Node] = set()

    def is_close_to_pauli(self, *, rel_tol: float | None = None, abs_tol: float | None = None) -> Pauli | None:
        angle_double = 2 * self.angle
        angle_double_int = round(angle_double)
        kwargs = {k: v for k, v in [("rel_tol", rel_tol), ("abs_tol", abs_tol)] if v is not None}
        if not math.isclose(angle_double, angle_double_int, **kwargs):
            return None
        angle_double_mod_4 = angle_double_int % 4
        if angle_double_mod_4 % 2 == 0:
            axis = self.plane.cos
        else:
            axis = self.plane.sin
        sign = Sign.minus_if(angle_double_mod_4 >= 2)
        return Pauli(axis=axis, sign=sign)

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
        update = graphix.pauli.MeasureUpdate.compute(self.plane, False, False, clifford)
        return M(
            node=self.node,
            plane=update.new_plane,
            angle=self.angle * update.coeff + update.add_term / np.pi,
            s_domain=s_domain,
            t_domain=t_domain,
        )


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
    domain: set[Node] = set()


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
    domain: set[Node] = set()


class T(Command):
    """
    T command
    """

    kind: CommandKind = CommandKind.T

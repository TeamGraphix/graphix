"""Data validator command classes."""

from __future__ import annotations

import abc
import dataclasses
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import typing_extensions
from pydantic import BaseModel, ConfigDict

from graphix import clifford
from graphix.pauli import Pauli, Plane, Sign
from graphix.states import BasicStates, State

if TYPE_CHECKING:
    from graphix.clifford import Clifford

Node = int


class CommandKind(Enum):
    N = "N"
    M = "M"
    E = "E"
    C = "C"
    X = "X"
    Z = "Z"
    S = "S"


class Command(ABC):
    """
    Base command class.
    """

    @property
    @abc.abstractmethod
    def kind(self) -> Any: ...


# Decorator required
@dataclasses.dataclass
class Correction(Command):
    """
    Correction command.
    Either X or Z.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class N(Command):
    """
    Preparation command.
    """

    node: Node
    state: State = BasicStates.PLUS

    @property
    @typing_extensions.override
    def kind(self) -> Literal[CommandKind.N]:
        return CommandKind.N


# Decorator required
@dataclasses.dataclass
class BaseM(Command):
    node: Node


@dataclasses.dataclass
class M(BaseM):
    """
    Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop.
    """

    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = dataclasses.field(default_factory=set)
    t_domain: set[Node] = dataclasses.field(default_factory=set)

    @property
    @typing_extensions.override
    def kind(self) -> Literal[CommandKind.M]:
        return CommandKind.M

    def clifford(self, clifford_gate: Clifford) -> M:
        s_domain = self.s_domain
        t_domain = self.t_domain
        for gate in clifford_gate.hsz:
            if gate == clifford.I:
                pass
            elif gate == clifford.H:
                t_domain, s_domain = s_domain, t_domain
            elif gate == clifford.S:
                t_domain ^= s_domain
            elif gate == clifford.Z:
                pass
            else:
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        update = MeasureUpdate.compute(self.plane, False, False, clifford_gate)
        return M(self.node, update.new_plane, self.angle * update.coeff + update.add_term / np.pi, s_domain, t_domain)


@dataclasses.dataclass
class E(Command):
    """
    Entanglement command.
    """

    nodes: tuple[Node, Node]

    @property
    @typing_extensions.override
    def kind(self) -> Literal[CommandKind.E]:
        return CommandKind.E


@dataclasses.dataclass
class C(Command):
    """
    Clifford command.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for the `clifford` field
    node: Node
    clifford: clifford.Clifford

    @property
    @typing_extensions.override
    def kind(self) -> Literal[CommandKind.C]:
        return CommandKind.C


@dataclasses.dataclass
class X(Correction):
    """
    X correction command.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[CommandKind.X]:
        return CommandKind.X


@dataclasses.dataclass
class Z(Correction):
    """
    Z correction command.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[CommandKind.Z]:
        return CommandKind.Z


@dataclasses.dataclass
class S(Command):
    """
    S command
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)

    @property
    @typing_extensions.override
    def kind(self) -> Literal[CommandKind.S]:
        return CommandKind.S


class MeasureUpdate(BaseModel):
    new_plane: Plane
    coeff: int
    add_term: float

    @staticmethod
    def compute(plane: Plane, s: bool, t: bool, clifford_gate: Clifford) -> MeasureUpdate:
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

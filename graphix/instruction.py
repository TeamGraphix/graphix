"""Instruction classes."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Literal, SupportsFloat

# Self introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import Self, override

from graphix import utils
from graphix.fundamentals import (
    Axis,
    ParameterizedAngle,
    Plane,
)
from graphix.pretty_print import OutputFormat, angle_to_str
from graphix.repr_mixins import DataclassReprMixin


def repr_angle(angle: ParameterizedAngle) -> str:
    """
    Return the representation string of an angle in radians.

    This is used for pretty-printing instructions with `angle` parameters.
    Delegates to :func:`pretty_print.angle_to_str`.
    """
    # Non-float-supporting objects are returned as-is
    if not isinstance(angle, SupportsFloat):
        return str(angle)

    return angle_to_str(angle, OutputFormat.ASCII)


class InstructionKind(Enum):
    """Tag for instruction kind."""

    CCX = enum.auto()
    RZZ = enum.auto()
    CNOT = enum.auto()
    SWAP = enum.auto()
    CZ = enum.auto()
    H = enum.auto()
    S = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()
    I = enum.auto()
    M = enum.auto()
    RX = enum.auto()
    RY = enum.auto()
    RZ = enum.auto()


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define the ``kind`` attribute."""
        super().__init_subclass__()
        utils.check_kind(cls, {"InstructionKind": InstructionKind, "Plane": Plane})


class InstructionVisitor:
    """Visitor for instruction.

    This base class can be subclassed to rewrite instructions by
    overriding some of the following functions:

    - ``visit_qubit``: rewrite qubit indices.

    - ``visit_angle``: rewrite angles.

    - ``visit_axis``: rewrite axes.
    """

    def visit_qubit(self, qubit: int) -> int:  # noqa: PLR6301
        """Rewrite a qubit index."""
        return qubit

    def visit_angle(self, angle: ParameterizedAngle) -> ParameterizedAngle:  # noqa: PLR6301
        """Rewrite an angle."""
        return angle

    def visit_axis(self, axis: Axis) -> Axis:  # noqa: PLR6301
        """Rewrite an axis."""
        return axis


class BaseInstruction(ABC, DataclassReprMixin):
    """Base class for circuit instruction."""

    @abstractmethod
    def visit(self, visitor: InstructionVisitor) -> Self:
        """Rewrite the instruction according to the given visitor."""


@dataclass(repr=False)
class CCX(_KindChecker, BaseInstruction):
    """Toffoli circuit instruction."""

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = field(default=InstructionKind.CCX, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> CCX:
        u, v = self.controls
        return CCX(visitor.visit_qubit(self.target), (visitor.visit_qubit(u), visitor.visit_qubit(v)))


@dataclass(repr=False)
class RZZ(_KindChecker, BaseInstruction):
    """RZZ circuit instruction."""

    target: int
    control: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RZZ]] = field(default=InstructionKind.RZZ, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> RZZ:
        return RZZ(visitor.visit_qubit(self.target), visitor.visit_qubit(self.control), visitor.visit_angle(self.angle))


@dataclass(repr=False)
class CNOT(_KindChecker, BaseInstruction):
    """CNOT circuit instruction."""

    target: int
    control: int
    kind: ClassVar[Literal[InstructionKind.CNOT]] = field(default=InstructionKind.CNOT, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> CNOT:
        return CNOT(visitor.visit_qubit(self.target), visitor.visit_qubit(self.control))


@dataclass(repr=False)
class CZ(_KindChecker, BaseInstruction):
    """CZ circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CZ]] = field(default=InstructionKind.CZ, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> CZ:
        u, v = self.targets
        return CZ((visitor.visit_qubit(u), visitor.visit_qubit(v)))


@dataclass(repr=False)
class SWAP(_KindChecker, BaseInstruction):
    """SWAP circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = field(default=InstructionKind.SWAP, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> SWAP:
        u, v = self.targets
        return SWAP((visitor.visit_qubit(u), visitor.visit_qubit(v)))


@dataclass(repr=False)
class H(_KindChecker, BaseInstruction):
    """H circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.H]] = field(default=InstructionKind.H, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> H:
        return H(visitor.visit_qubit(self.target))


@dataclass(repr=False)
class S(_KindChecker, BaseInstruction):
    """S circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.S]] = field(default=InstructionKind.S, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> S:
        return S(visitor.visit_qubit(self.target))


@dataclass(repr=False)
class X(_KindChecker, BaseInstruction):
    """X circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.X]] = field(default=InstructionKind.X, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> X:
        return X(visitor.visit_qubit(self.target))


@dataclass(repr=False)
class Y(_KindChecker, BaseInstruction):
    """Y circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Y]] = field(default=InstructionKind.Y, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> Y:
        return Y(visitor.visit_qubit(self.target))


@dataclass(repr=False)
class Z(_KindChecker, BaseInstruction):
    """Z circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Z]] = field(default=InstructionKind.Z, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> Z:
        return Z(visitor.visit_qubit(self.target))


@dataclass(repr=False)
class I(_KindChecker, BaseInstruction):
    """I circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.I]] = field(default=InstructionKind.I, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> I:
        return I(visitor.visit_qubit(self.target))


@dataclass(repr=False)
class M(_KindChecker, BaseInstruction):
    """M circuit instruction."""

    target: int
    axis: Axis
    kind: ClassVar[Literal[InstructionKind.M]] = field(default=InstructionKind.M, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> M:
        return M(visitor.visit_qubit(self.target), visitor.visit_axis(self.axis))


@dataclass(repr=False)
class RX(_KindChecker, BaseInstruction):
    """X rotation circuit instruction."""

    target: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RX]] = field(default=InstructionKind.RX, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> RX:
        return RX(visitor.visit_qubit(self.target), visitor.visit_angle(self.angle))


@dataclass(repr=False)
class RY(_KindChecker, BaseInstruction):
    """Y rotation circuit instruction."""

    target: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RY]] = field(default=InstructionKind.RY, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> RY:
        return RY(visitor.visit_qubit(self.target), visitor.visit_angle(self.angle))


@dataclass(repr=False)
class RZ(_KindChecker, BaseInstruction):
    """Z rotation circuit instruction."""

    target: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RZ]] = field(default=InstructionKind.RZ, init=False)

    @override
    def visit(self, visitor: InstructionVisitor) -> RZ:
        return RZ(visitor.visit_qubit(self.target), visitor.visit_angle(self.angle))


InstructionWithoutRZZ = CCX | CNOT | SWAP | CZ | H | S | X | Y | Z | I | M | RX | RY | RZ
Instruction = InstructionWithoutRZZ | RZZ

"""Instruction classes."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal, SupportsFloat, TypeAlias

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
    J = enum.auto()
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

    def visit_qubit(self, qubit: int) -> int:
        """Rewrite a qubit index."""
        return qubit

    def visit_angle(self, angle: ParameterizedAngle) -> ParameterizedAngle:
        """Rewrite an angle."""
        return angle

    def visit_axis(self, axis: Axis) -> Axis:
        """Rewrite an axis."""
        return axis


class BaseInstruction(ABC, DataclassReprMixin):
    """Base class for circuit instructions."""

    @abstractmethod
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        """Rewrite the instruction according to the given visitor.

        Parameters
        ----------
        visitor : InstructionVisitor
            The visitor specifying the rewriting.

        copy : bool, optional
            If ``True``, the current instruction remains unchanged, and a
            new instruction is returned. The default is ``False``, meaning
            that changes are performed in place.

        Returns
        -------
        Self
            The rewritten instruction. Equal to ``self`` if ``copy`` is ``False``.
        """


@dataclass(repr=False)
class CCX(_KindChecker, BaseInstruction):
    """Toffoli circuit instruction."""

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = field(default=InstructionKind.CCX, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> CCX:
        u, v = self.controls
        target = visitor.visit_qubit(self.target)
        controls = (visitor.visit_qubit(u), visitor.visit_qubit(v))
        if controls[0] == controls[1]:
            raise RuntimeError(f"Control qubits cannot be the same. Qubit index: {controls[0]}.")
        for i, c in enumerate(controls):
            if target == c:
                raise RuntimeError(f"Target and control-{i} qubits cannot be the same. Qubit index: {target}.")

        if copy:
            return CCX(target, controls)
        self.target = target
        self.controls = controls
        return self


@dataclass(repr=False)
class RZZ(_KindChecker, BaseInstruction):
    """RZZ circuit instruction."""

    target: int
    control: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RZZ]] = field(default=InstructionKind.RZZ, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> RZZ:
        target = visitor.visit_qubit(self.target)
        control = visitor.visit_qubit(self.control)
        if target == control:
            raise RuntimeError(f"Target and control qubits cannot be the same. Qubit index: {target}.")
        angle = visitor.visit_angle(self.angle)
        if copy:
            return RZZ(target, control, angle)
        self.target = target
        self.control = control
        self.angle = angle
        return self


@dataclass(repr=False)
class CNOT(_KindChecker, BaseInstruction):
    """CNOT circuit instruction."""

    target: int
    control: int
    kind: ClassVar[Literal[InstructionKind.CNOT]] = field(default=InstructionKind.CNOT, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> CNOT:
        target = visitor.visit_qubit(self.target)
        control = visitor.visit_qubit(self.control)
        if target == control:
            raise RuntimeError(f"Target and control qubits cannot be the same. Qubit index: {target}.")
        if copy:
            return CNOT(target, control)
        self.target = target
        self.control = control
        return self


@dataclass(repr=False)
class CZ(_KindChecker, BaseInstruction):
    """CZ circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CZ]] = field(default=InstructionKind.CZ, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> CZ:
        u, v = self.targets
        targets = (visitor.visit_qubit(u), visitor.visit_qubit(v))
        if targets[0] == targets[1]:
            raise RuntimeError(f"Target qubits cannot be the same. Qubit index: {targets[0]}.")
        if copy:
            return CZ(targets)
        self.targets = targets
        return self


@dataclass(repr=False)
class SWAP(_KindChecker, BaseInstruction):
    """SWAP circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = field(default=InstructionKind.SWAP, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> SWAP:
        u, v = self.targets
        targets = (visitor.visit_qubit(u), visitor.visit_qubit(v))
        if targets[0] == targets[1]:
            raise RuntimeError(f"Target qubits cannot be the same. Qubit index: {targets[0]}.")
        if copy:
            return SWAP(targets)
        self.targets = targets
        return self


@dataclass(repr=False)
class SingleTargetInstruction(BaseInstruction):
    """Base class for single-target circuit instructions."""

    target: int

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        target = visitor.visit_qubit(self.target)
        if copy:
            return type(self)(target)
        self.target = target
        return self


@dataclass(repr=False)
class H(_KindChecker, SingleTargetInstruction):
    """H circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.H]] = field(default=InstructionKind.H, init=False)


@dataclass(repr=False)
class S(_KindChecker, SingleTargetInstruction):
    """S circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.S]] = field(default=InstructionKind.S, init=False)


@dataclass(repr=False)
class X(_KindChecker, SingleTargetInstruction):
    """X circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.X]] = field(default=InstructionKind.X, init=False)


@dataclass(repr=False)
class Y(_KindChecker, SingleTargetInstruction):
    """Y circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.Y]] = field(default=InstructionKind.Y, init=False)


@dataclass(repr=False)
class Z(_KindChecker, SingleTargetInstruction):
    """Z circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.Z]] = field(default=InstructionKind.Z, init=False)


@dataclass(repr=False)
class I(_KindChecker, SingleTargetInstruction):
    """I circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.I]] = field(default=InstructionKind.I, init=False)


@dataclass(repr=False)
class M(_KindChecker, BaseInstruction):
    """M circuit instruction."""

    target: int
    axis: Axis
    kind: ClassVar[Literal[InstructionKind.M]] = field(default=InstructionKind.M, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> M:
        target = visitor.visit_qubit(self.target)
        axis = visitor.visit_axis(self.axis)
        if copy:
            return M(target, axis)
        self.target = target
        self.axis = axis
        return self


@dataclass(repr=False)
class RotationInstruction(BaseInstruction):
    """Base class for rotation instructions."""

    target: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        target = visitor.visit_qubit(self.target)
        angle = visitor.visit_angle(self.angle)
        if copy:
            return type(self)(target, angle)
        self.target = target
        self.angle = angle
        return self


@dataclass(repr=False)
class RX(_KindChecker, RotationInstruction):
    """X rotation circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.RX]] = field(default=InstructionKind.RX, init=False)


@dataclass(repr=False)
class RY(_KindChecker, RotationInstruction):
    """Y rotation circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.RY]] = field(default=InstructionKind.RY, init=False)


@dataclass(repr=False)
class RZ(_KindChecker, RotationInstruction):
    """Z rotation circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.RZ]] = field(default=InstructionKind.RZ, init=False)


@dataclass(repr=False)
class J(_KindChecker, RotationInstruction):
    """J circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.J]] = field(default=InstructionKind.J, init=False)


class InstructionWithoutRZZ:
    """Grouping of all instructions except RZZ for namespace exposure.

    Notes
    -----
    This class is not meant to be instantiated, but rather serves as a namespace for all instructions except RZZ.
    The type alias for "any command" is :data:`InstructionKind`.
    """

    CCX: TypeAlias = CCX
    CNOT: TypeAlias = CNOT
    CZ: TypeAlias = CZ
    SWAP: TypeAlias = SWAP
    H: TypeAlias = H
    S: TypeAlias = S
    X: TypeAlias = X
    Y: TypeAlias = Y
    Z: TypeAlias = Z
    I: TypeAlias = I
    M: TypeAlias = M
    RX: TypeAlias = RX
    RY: TypeAlias = RY
    RZ: TypeAlias = RZ
    J: TypeAlias = J

    def __init__(self) -> None:
        raise TypeError("InstructionWithoutRZZ is a namespace, not a class.")


class Instruction(InstructionWithoutRZZ):
    """Grouping of all instructions for namespace exposure.

    Notes
    -----
    This class is not meant to be instantiated, but rather serves as a namespace for all instructions except RZZ.
    The type alias for "any command" is :data:`InstructionKind`.
    """

    RZZ: TypeAlias = RZZ

    def __init__(self) -> None:
        raise TypeError("Instruction is a namespace, not a class.")


if TYPE_CHECKING:
    InstructionTypeWithoutRZZ = CCX | CNOT | SWAP | CZ | H | S | X | Y | Z | I | M | RX | RY | RZ | J
    InstructionType = InstructionTypeWithoutRZZ | RZZ

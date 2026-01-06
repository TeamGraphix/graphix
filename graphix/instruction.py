"""Instruction classes."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Literal, SupportsFloat

from graphix import utils
from graphix.fundamentals import Axis, Plane

# Ruff suggests to move this import to a type-checking block, but dataclass requires it here
from graphix.parameter import ExpressionOrFloat  # noqa: TC001
from graphix.pretty_print import OutputFormat, angle_to_str
from graphix.repr_mixins import DataclassReprMixin


def repr_angle(angle: ExpressionOrFloat) -> str:
    """
    Return the representation string of an angle in radians.

    This is used for pretty-printing instructions with `angle` parameters.
    Delegates to :func:`pretty_print.angle_to_str`.
    """
    # Non-float-supporting objects are returned as-is
    if not isinstance(angle, SupportsFloat):
        return str(angle)

    # Convert to float, express in π units, and format in ASCII/plain mode
    pi_units = float(angle) / math.pi
    return angle_to_str(pi_units, OutputFormat.ASCII)


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


class BaseInstruction(DataclassReprMixin):
    """Base class for circuit instruction."""


@dataclass(repr=False)
class CCX(_KindChecker, BaseInstruction):
    """Toffoli circuit instruction."""

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = field(default=InstructionKind.CCX, init=False)


@dataclass(repr=False)
class RZZ(_KindChecker, BaseInstruction):
    """RZZ circuit instruction."""

    target: int
    control: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RZZ]] = field(default=InstructionKind.RZZ, init=False)


@dataclass(repr=False)
class CNOT(_KindChecker, BaseInstruction):
    """CNOT circuit instruction."""

    target: int
    control: int
    kind: ClassVar[Literal[InstructionKind.CNOT]] = field(default=InstructionKind.CNOT, init=False)


@dataclass(repr=False)
class CZ(_KindChecker, BaseInstruction):
    """CZ circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CZ]] = field(default=InstructionKind.CZ, init=False)


@dataclass(repr=False)
class SWAP(_KindChecker, BaseInstruction):
    """SWAP circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = field(default=InstructionKind.SWAP, init=False)


@dataclass(repr=False)
class H(_KindChecker, BaseInstruction):
    """H circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.H]] = field(default=InstructionKind.H, init=False)


@dataclass(repr=False)
class S(_KindChecker, BaseInstruction):
    """S circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.S]] = field(default=InstructionKind.S, init=False)


@dataclass(repr=False)
class X(_KindChecker, BaseInstruction):
    """X circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.X]] = field(default=InstructionKind.X, init=False)


@dataclass(repr=False)
class Y(_KindChecker, BaseInstruction):
    """Y circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Y]] = field(default=InstructionKind.Y, init=False)


@dataclass(repr=False)
class Z(_KindChecker, BaseInstruction):
    """Z circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Z]] = field(default=InstructionKind.Z, init=False)


@dataclass(repr=False)
class I(_KindChecker, BaseInstruction):
    """I circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.I]] = field(default=InstructionKind.I, init=False)


@dataclass(repr=False)
class M(_KindChecker, BaseInstruction):
    """M circuit instruction."""

    target: int
    axis: Axis
    kind: ClassVar[Literal[InstructionKind.M]] = field(default=InstructionKind.M, init=False)


@dataclass(repr=False)
class RX(_KindChecker, BaseInstruction):
    """X rotation circuit instruction."""

    target: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RX]] = field(default=InstructionKind.RX, init=False)


@dataclass(repr=False)
class RY(_KindChecker, BaseInstruction):
    """Y rotation circuit instruction."""

    target: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RY]] = field(default=InstructionKind.RY, init=False)


@dataclass(repr=False)
class RZ(_KindChecker, BaseInstruction):
    """Z rotation circuit instruction."""

    target: int
    angle: ExpressionOrFloat = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RZ]] = field(default=InstructionKind.RZ, init=False)


Instruction = CCX | RZZ | CNOT | SWAP | CZ | H | S | X | Y | Z | I | M | RX | RY | RZ

from __future__ import annotations

import abc
import dataclasses
from abc import ABC
from enum import Enum
from typing import Any, Literal

import typing_extensions
from typing_extensions import Never

if typing_extensions.TYPE_CHECKING:
    from graphix.pauli import Plane


class InstructionKind(Enum):
    CCX = "CCX"
    RZZ = "RZZ"
    CNOT = "CNOT"
    SWAP = "SWAP"
    H = "H"
    S = "S"
    X = "X"
    Y = "Y"
    Z = "Z"
    I = "I"
    M = "M"
    RX = "RX"
    RY = "RY"
    RZ = "RZ"
    # The two following instructions are used internally by the transpiler
    XC = "XC"
    ZC = "ZC"


# Decorator required
@dataclasses.dataclass
class Instruction(ABC):
    """
    Circuit instruction base class.
    """

    @property
    @abc.abstractmethod
    def kind(self) -> Any: ...

    @kind.setter
    def kind(self, _: InstructionKind) -> Never:
        raise AttributeError("kind is read-only.")


# Decorator required
@dataclasses.dataclass
class OneQubitInstruction(Instruction):
    """
    One qubit circuit instruction base class.
    """

    target: int


# Decorator required
@dataclasses.dataclass
class CorrectionInstruction(OneQubitInstruction):
    """
    Correction instruction base class.
    """

    domain: set[int]


# Decorator required
@dataclasses.dataclass
class RotationInstruction(OneQubitInstruction):
    """
    Rotation instruction base class.
    """

    angle: float
    meas_index: int | None = None


# Decorator required
@dataclasses.dataclass
class OneControlInstruction(OneQubitInstruction):
    """
    One control instruction base class.
    """

    control: int


# Decorator required
@dataclasses.dataclass
class TwoControlsInstruction(OneQubitInstruction):
    """
    Two controls instruction base class.
    """

    controls: tuple[int, int]


@dataclasses.dataclass
class XC(CorrectionInstruction):
    """
    X correction circuit instruction. Used internally by the transpiler.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.XC]:
        return InstructionKind.XC


@dataclasses.dataclass
class ZC(CorrectionInstruction):
    """
    Z correction circuit instruction. Used internally by the transpiler.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.ZC]:
        return InstructionKind.ZC


@dataclasses.dataclass
class CCX(TwoControlsInstruction):
    """
    Toffoli circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.CCX]:
        return InstructionKind.CCX


@dataclasses.dataclass
class RZZ(RotationInstruction, OneControlInstruction):
    """
    RZZ circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.RZZ]:
        return InstructionKind.RZZ


@dataclasses.dataclass
class CNOT(OneControlInstruction):
    """
    CNOT circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.CNOT]:
        return InstructionKind.CNOT


@dataclasses.dataclass
class SWAP(Instruction):
    """
    SWAP circuit instruction.
    """

    targets: tuple[int, int]

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.SWAP]:
        return InstructionKind.SWAP


@dataclasses.dataclass
class H(OneQubitInstruction):
    """
    H circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.H]:
        return InstructionKind.H


@dataclasses.dataclass
class S(OneQubitInstruction):
    """
    S circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.S]:
        return InstructionKind.S


@dataclasses.dataclass
class X(OneQubitInstruction):
    """
    X circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.X]:
        return InstructionKind.X


@dataclasses.dataclass
class Y(OneQubitInstruction):
    """
    Y circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.Y]:
        return InstructionKind.Y


@dataclasses.dataclass
class Z(OneQubitInstruction):
    """
    Z circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.Z]:
        return InstructionKind.Z


@dataclasses.dataclass
class I(OneQubitInstruction):
    """
    I circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.I]:
        return InstructionKind.I


@dataclasses.dataclass
class M(OneQubitInstruction):
    """
    M circuit instruction.
    """

    plane: Plane
    angle: float

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.M]:
        return InstructionKind.M


@dataclasses.dataclass
class RX(RotationInstruction):
    """
    X rotation circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.RX]:
        return InstructionKind.RX


@dataclasses.dataclass
class RY(RotationInstruction):
    """
    Y rotation circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.RY]:
        return InstructionKind.RY


@dataclasses.dataclass
class RZ(RotationInstruction):
    """
    Z rotation circuit instruction.
    """

    @property
    @typing_extensions.override
    def kind(self) -> Literal[InstructionKind.RZ]:
        return InstructionKind.RZ

from __future__ import annotations

import dataclasses
import enum

from graphix import utils
from graphix.pauli import Plane


class InstructionKind(enum.Enum):
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


@utils.disable_init
class Instruction:
    """
    Circuit instruction.
    """

    meas_index: int


@utils.disable_init
class OneQubitInstruction(Instruction):
    """
    One qubit circuit instruction.
    """

    target: int


@utils.disable_init
class CorrectionInstruction(OneQubitInstruction):
    """
    Correction instruction.
    """

    domain: set[int]


@utils.disable_init
class RotationInstruction(OneQubitInstruction):
    """
    Rotation instruction.
    """

    angle: float


@utils.disable_init
class OneControlInstruction(OneQubitInstruction):
    """
    One control instruction.
    """

    control: int


@utils.disable_init
class TwoControlsInstruction(OneQubitInstruction):
    """
    Two controls instruction.
    """

    controls: tuple[int, int]


@dataclasses.dataclass
class XC(CorrectionInstruction):
    """
    X correction circuit instruction. Used internally by the transpiler.
    """


@dataclasses.dataclass
class ZC(CorrectionInstruction):
    """
    Z correction circuit instruction. Used internally by the transpiler.
    """


@dataclasses.dataclass
class CCX(TwoControlsInstruction):
    """
    Toffoli circuit instruction.
    """


@dataclasses.dataclass
class RZZ(OneControlInstruction, RotationInstruction):
    """
    RZZ circuit instruction.
    """


@dataclasses.dataclass
class CNOT(OneControlInstruction):
    """
    CNOT circuit instruction.
    """


@dataclasses.dataclass
class SWAP(Instruction):
    """
    SWAP circuit instruction.
    """

    targets: tuple[int, int]


@dataclasses.dataclass
class H(OneQubitInstruction):
    """
    H circuit instruction.
    """


@dataclasses.dataclass
class S(OneQubitInstruction):
    """
    S circuit instruction.
    """


@dataclasses.dataclass
class X(OneQubitInstruction):
    """
    X circuit instruction.
    """


@dataclasses.dataclass
class Y(OneQubitInstruction):
    """
    Y circuit instruction.
    """


@dataclasses.dataclass
class Z(OneQubitInstruction):
    """
    Z circuit instruction.
    """


@dataclasses.dataclass
class I(OneQubitInstruction):
    """
    I circuit instruction.
    """


@dataclasses.dataclass
class M(OneQubitInstruction):
    """
    M circuit instruction.
    """

    plane: Plane
    angle: float


@dataclasses.dataclass
class RX(RotationInstruction):
    """
    X rotation circuit instruction.
    """


@dataclasses.dataclass
class RY(RotationInstruction):
    """
    Y rotation circuit instruction.
    """


@dataclasses.dataclass
class RZ(RotationInstruction):
    """
    Z rotation circuit instruction.
    """

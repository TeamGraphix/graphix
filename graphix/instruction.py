from enum import Enum
from pydantic import BaseModel


class InstructionName(Enum):
    CNOT = "CNOT"
    SWAP = "SWAP"
    H = "H"
    S = "S"
    X = "X"
    Y = "Y"
    Z = "Z"
    RX = "RX"
    RY = "RY"
    RZ = "RZ"
    RZZ = "RZZ"
    CCX = "CCX"
    I = "I"
    XC = "XC"
    ZC = "ZC"


class Instruction(BaseModel):
    """
    Circuit instruction base class model.
    """

    name: InstructionName
    meas_index: int = None


class OneQubitInstruction(Instruction):
    """
    One qubit circuit instruction base class model.
    """

    target: int


class CorrectionInstruction(OneQubitInstruction):
    """
    Correction instruction base class model.
    """

    domain: list[int]


class RotationInstruction(OneQubitInstruction):
    """
    Rotation instruction base class model.
    """

    angle: float


class OneControlInstruction(OneQubitInstruction):
    """
    One control instruction base class model.
    """

    control: int


class TwoControlsInstruction(OneQubitInstruction):
    """
    Two controls instruction base class model.
    """

    controls: tuple[int, int]


class XC(CorrectionInstruction):
    """
    X correction circuit instruction.
    """

    name: InstructionName = InstructionName.XC


class ZC(CorrectionInstruction):
    """
    Z correction circuit instruction.
    """

    name: InstructionName = InstructionName.ZC


class CCX(TwoControlsInstruction):
    """
    Toffoli circuit instruction.
    """

    name: InstructionName = InstructionName.CCX


class RZZ(OneControlInstruction, RotationInstruction):
    """
    RZZ circuit instruction.
    """

    name: InstructionName = InstructionName.RZZ


class CNOT(OneControlInstruction):
    """
    CNOT circuit instruction.
    """

    name: InstructionName = InstructionName.CNOT


class SWAP(Instruction):
    """
    SWAP circuit instruction.
    """

    name: InstructionName = InstructionName.SWAP
    targets: tuple[int, int]


class H(OneQubitInstruction):
    """
    H circuit instruction.
    """

    name: InstructionName = InstructionName.H


class S(OneQubitInstruction):
    """
    S circuit instruction.
    """

    name: InstructionName = InstructionName.S


class X(OneQubitInstruction):
    """
    X circuit instruction.
    """

    name: InstructionName = InstructionName.X


class Y(OneQubitInstruction):
    """
    Y circuit instruction.
    """

    name: InstructionName = InstructionName.Y


class Z(OneQubitInstruction):
    """
    Z circuit instruction.
    """

    name: InstructionName = InstructionName.Z


class I(OneQubitInstruction):
    """
    I circuit instruction.
    """

    name: InstructionName = InstructionName.I


class RX(RotationInstruction):
    """
    X rotation circuit instruction.
    """

    name: InstructionName = InstructionName.RX


class RY(RotationInstruction):
    """
    Y rotation circuit instruction.
    """

    name: InstructionName = InstructionName.RY


class RZ(RotationInstruction):
    """
    Z rotation circuit instruction.
    """

    name: InstructionName = InstructionName.RZ

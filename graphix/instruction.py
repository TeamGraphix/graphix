from pydantic import BaseModel
import enum

class InstructionKind(str, enum.Enum):
    XC = 'XC'
    ZC = 'ZC'
    CCX = 'CCX'
    RZZ = 'RZZ'
    CNOT = 'CNOT'
    SWAP = 'SWAP'
    H = 'H'
    S = 'S'
    X = 'X'
    Y = 'Y'
    Z = 'Z'
    I = 'I'
    RX = 'RX'
    RY = 'RY'
    RZ = 'RZ'


class Instruction(BaseModel):
    """
    Circuit instruction base class model.
    """
    
    kind: InstructionKind = None
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

    kind: InstructionKind = InstructionKind.XC


class ZC(CorrectionInstruction):
    """
    Z correction circuit instruction.
    """

    kind: InstructionKind = InstructionKind.ZC


class CCX(TwoControlsInstruction):
    """
    Toffoli circuit instruction.
    """

    kind: InstructionKind = InstructionKind.CCX


class RZZ(OneControlInstruction, RotationInstruction):
    """
    RZZ circuit instruction.
    """

    kind: InstructionKind = InstructionKind.RZZ


class CNOT(OneControlInstruction):
    """
    CNOT circuit instruction.
    """

    kind: InstructionKind = InstructionKind.CNOT


class SWAP(Instruction):
    """
    SWAP circuit instruction.
    """

    kind: InstructionKind = InstructionKind.SWAP
    targets: tuple[int, int]


class H(OneQubitInstruction):
    """
    H circuit instruction.
    """

    kind: InstructionKind = InstructionKind.H


class S(OneQubitInstruction):
    """
    S circuit instruction.
    """

    kind: InstructionKind = InstructionKind.S


class X(OneQubitInstruction):
    """
    X circuit instruction.
    """

    kind: InstructionKind = InstructionKind.X


class Y(OneQubitInstruction):
    """
    Y circuit instruction.
    """

    kind: InstructionKind = InstructionKind.Y


class Z(OneQubitInstruction):
    """
    Z circuit instruction.
    """

    kind: InstructionKind = InstructionKind.Z


class I(OneQubitInstruction):
    """
    I circuit instruction.
    """

    kind: InstructionKind = InstructionKind.I


class RX(RotationInstruction):
    """
    X rotation circuit instruction.
    """

    kind: InstructionKind = InstructionKind.RX


class RY(RotationInstruction):
    """
    Y rotation circuit instruction.
    """

    kind: InstructionKind = InstructionKind.RY


class RZ(RotationInstruction):
    """
    Z rotation circuit instruction.
    """

    kind: InstructionKind = InstructionKind.RZ

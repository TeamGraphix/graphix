from __future__ import annotations

import abc
import enum

from pydantic import BaseModel

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


class Instruction(BaseModel, abc.ABC):
    """
    Circuit instruction base class model.
    """

    kind: InstructionKind = None
    meas_index: int = None
    
    @abc.abstractmethod
    def to_qasm3(self) -> str:
        pass


class OneQubitInstruction(Instruction):
    """
    One qubit circuit instruction base class model.
    """

    target: int
    
    def to_qasm3(self) -> str:
        return f"{self.kind.value.lower()} {self.target};"


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

    def to_qasm3(self) -> str:
        return f"{self.kind.value.lower()}({self.angle}) q[{self.target}]"


class OneControlInstruction(OneQubitInstruction):
    """
    One control instruction base class model.
    """

    control: int

    def to_qasm3(self) -> str:
        return f"{self.kind.value.lower()} q[{self.control}], q[{self.target}]"


class TwoControlsInstruction(OneQubitInstruction):
    """
    Two controls instruction base class model.
    """

    controls: tuple[int, int]

    def to_qasm3(self) -> str:
        return f"{self.kind.value.lower()} q[{self.controls[0]}], q[{self.controls[1]}], q[{self.target}]"


class XC(CorrectionInstruction):
    """
    X correction circuit instruction. Used internally by the transpiler.
    """

    kind: InstructionKind = InstructionKind.XC


class ZC(CorrectionInstruction):
    """
    Z correction circuit instruction. Used internally by the transpiler.
    """

    kind: InstructionKind = InstructionKind.ZC


class CCX(TwoControlsInstruction):
    """
    Toffoli circuit instruction.
    """

    kind: InstructionKind = InstructionKind.CCX

    def to_qasm3(self):
        return super().to_qasm3()


class RZZ(OneControlInstruction, RotationInstruction):
    """
    RZZ circuit instruction.
    """

    kind: InstructionKind = InstructionKind.RZZ
    
    def to_qasm3(self) -> str:
        return f"{self.kind.value.lower()}({self.angle}) q[{self.control}], q[{self.target}]"


class CNOT(OneControlInstruction):
    """
    CNOT circuit instruction.
    """

    kind: InstructionKind = InstructionKind.CNOT

    def to_qasm3(self):
        return f"cx q[{self.control}], q[{self.target}]"


class SWAP(Instruction):
    """
    SWAP circuit instruction.
    """

    kind: InstructionKind = InstructionKind.SWAP
    targets: tuple[int, int]

    def to_qasm3(self):
        return f"{self.kind.value.lower()} q[{self.targets[0]}], q[{self.targets[1]}]"


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


class M(OneQubitInstruction):
    """
    M circuit instruction.
    """

    kind: InstructionKind = InstructionKind.M
    plane: Plane
    angle: float
    
    def to_qasm3(self):
        return f"measure q[{self.target}] -> b[{self.target}]"


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

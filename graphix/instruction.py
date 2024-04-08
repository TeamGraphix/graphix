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
    Base circuit instruction class. Used to represent any kind of instruction.
    If an instruction doesn't need some attributes like control, domain or angle, they are juste setted to None.
    """
    name: InstructionName
    target: int | tuple[int, int]
    control: int | list[int] | None
    angle: float | None
    domain: list[int] = []
    meas_index: int | None


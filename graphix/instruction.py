from pydantic import BaseModel


class Instruction(BaseModel):
    """
    Circuit instruction base class model.
    """

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

    pass


class ZC(CorrectionInstruction):
    """
    Z correction circuit instruction.
    """

    pass


class CCX(TwoControlsInstruction):
    """
    Toffoli circuit instruction.
    """

    pass


class RZZ(OneControlInstruction, RotationInstruction):
    """
    RZZ circuit instruction.
    """

    pass


class CNOT(OneControlInstruction):
    """
    CNOT circuit instruction.
    """

    pass


class SWAP(Instruction):
    """
    SWAP circuit instruction.
    """

    targets: tuple[int, int]


class H(OneQubitInstruction):
    """
    H circuit instruction.
    """

    pass


class S(OneQubitInstruction):
    """
    S circuit instruction.
    """

    pass


class X(OneQubitInstruction):
    """
    X circuit instruction.
    """

    pass


class Y(OneQubitInstruction):
    """
    Y circuit instruction.
    """

    pass


class Z(OneQubitInstruction):
    """
    Z circuit instruction.
    """

    pass


class I(OneQubitInstruction):
    """
    I circuit instruction.
    """

    pass


class RX(RotationInstruction):
    """
    X rotation circuit instruction.
    """

    pass


class RY(RotationInstruction):
    """
    Y rotation circuit instruction.
    """

    pass


class RZ(RotationInstruction):
    """
    Z rotation circuit instruction.
    """

    pass

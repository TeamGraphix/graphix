from __future__ import annotations

import dataclasses
import sys
from enum import Enum
from typing import ClassVar, Literal, Union

from graphix import types
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


class _KindChecker:
    """
    Enforce tag field declaration.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        types.check_kind(cls, {"InstructionKind": InstructionKind, "Plane": Plane})


@dataclasses.dataclass
class CCX(_KindChecker):
    """
    Toffoli circuit instruction.
    """

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = dataclasses.field(default=InstructionKind.CCX, init=False)


@dataclasses.dataclass
class RZZ(_KindChecker):
    """
    RZZ circuit instruction.
    """

    target: int
    control: int
    angle: float
    # FIXME: Remove `| None` from `meas_index`
    # - `None` makes codes messy/type-unsafe
    # - `= None` results in subtle MRO issues
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZZ]] = dataclasses.field(default=InstructionKind.RZZ, init=False)


@dataclasses.dataclass
class CNOT(_KindChecker):
    """
    CNOT circuit instruction.
    """

    target: int
    control: int
    kind: ClassVar[Literal[InstructionKind.CNOT]] = dataclasses.field(default=InstructionKind.CNOT, init=False)


@dataclasses.dataclass
class SWAP(_KindChecker):
    """
    SWAP circuit instruction.
    """

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = dataclasses.field(default=InstructionKind.SWAP, init=False)


@dataclasses.dataclass
class H(_KindChecker):
    """
    H circuit instruction.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.H]] = dataclasses.field(default=InstructionKind.H, init=False)


@dataclasses.dataclass
class S(_KindChecker):
    """
    S circuit instruction.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.S]] = dataclasses.field(default=InstructionKind.S, init=False)


@dataclasses.dataclass
class X(_KindChecker):
    """
    X circuit instruction.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.X]] = dataclasses.field(default=InstructionKind.X, init=False)


@dataclasses.dataclass
class Y(_KindChecker):
    """
    Y circuit instruction.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.Y]] = dataclasses.field(default=InstructionKind.Y, init=False)


@dataclasses.dataclass
class Z(_KindChecker):
    """
    Z circuit instruction.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.Z]] = dataclasses.field(default=InstructionKind.Z, init=False)


@dataclasses.dataclass
class I(_KindChecker):
    """
    I circuit instruction.
    """

    target: int
    kind: ClassVar[Literal[InstructionKind.I]] = dataclasses.field(default=InstructionKind.I, init=False)


@dataclasses.dataclass
class M(_KindChecker):
    """
    M circuit instruction.
    """

    target: int
    plane: Plane
    angle: float
    kind: ClassVar[Literal[InstructionKind.M]] = dataclasses.field(default=InstructionKind.M, init=False)


@dataclasses.dataclass
class RX(_KindChecker):
    """
    X rotation circuit instruction.
    """

    target: int
    angle: float
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RX]] = dataclasses.field(default=InstructionKind.RX, init=False)


@dataclasses.dataclass
class RY(_KindChecker):
    """
    Y rotation circuit instruction.
    """

    target: int
    angle: float
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RY]] = dataclasses.field(default=InstructionKind.RY, init=False)


@dataclasses.dataclass
class RZ(_KindChecker):
    """
    Z rotation circuit instruction.
    """

    target: int
    angle: float
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZ]] = dataclasses.field(default=InstructionKind.RZ, init=False)


@dataclasses.dataclass
class XC(_KindChecker):
    """
    X correction circuit instruction. Used internally by the transpiler.
    """

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind.XC]] = dataclasses.field(default=InstructionKind.XC, init=False)


@dataclasses.dataclass
class ZC(_KindChecker):
    """
    Z correction circuit instruction. Used internally by the transpiler.
    """

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind.ZC]] = dataclasses.field(default=InstructionKind.ZC, init=False)


if sys.version_info >= (3, 10):
    Instruction = CCX | RZZ | CNOT | SWAP | H | S | X | Y | Z | I | M | RX | RY | RZ | XC | ZC
else:
    Instruction = Union[CCX, RZZ, CNOT, SWAP, H, S, X, Y, Z, I, M, RX, RY, RZ, XC, ZC]

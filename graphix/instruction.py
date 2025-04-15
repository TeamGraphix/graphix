"""Instruction classes."""

from __future__ import annotations

import dataclasses
import enum
import sys
from enum import Enum
from fractions import Fraction
from typing import ClassVar, Literal, Union

import numpy as np
from typing_extensions import assert_never

from graphix import utils
from graphix.fundamentals import Plane

# Ruff suggests to move this import to a type-checking block, but dataclass requires it here
from graphix.parameter import Expression, ExpressionOrFloat


def to_qasm3(instruction: Instruction) -> str:
    """Get the qasm3 representation of a single circuit instruction."""
    kind = instruction.kind
    if kind == InstructionKind.M:
        return f"b[{instruction.target}] = measure q[{instruction.target}]"
    # Use of `==` here for mypy
    if kind in {InstructionKind.RX, InstructionKind.RY, InstructionKind.RZ}:
        if isinstance(instruction.angle, Expression):
            raise ValueError("QASM export of symbolic pattern is not supported")
        rad_over_pi = instruction.angle / np.pi
        tol = 1e-9
        frac = Fraction(rad_over_pi).limit_denominator(1000)
        if abs(rad_over_pi - float(frac)) > tol:
            angle = f"{rad_over_pi}*pi"
        num, den = frac.numerator, frac.denominator
        sign = "-" if num < 0 else ""
        num = abs(num)
        if den == 1:
            angle = f"{sign}pi" if num == 1 else f"{sign}{num}*pi"
        else:
            angle = f"{sign}pi/{den}" if num == 1 else f"{sign}{num}*pi/{den}"
        return f"{kind.name.lower()}({angle}) q[{instruction.target}]"

    # Use of `==` here for mypy
    if (
        kind == InstructionKind.H  # noqa: PLR1714
        or kind == InstructionKind.I
        or kind == InstructionKind.S
        or kind == InstructionKind.X
        or kind == InstructionKind.Y
        or kind == InstructionKind.Z
    ):
        return f"{kind.name.lower()} q[{instruction.target}]"
    if kind == InstructionKind.CNOT:
        return f"cx q[{instruction.control}], q[{instruction.target}]"
    if kind == InstructionKind.SWAP:
        return f"swap q[{instruction.targets[0]}], q[{instruction.targets[1]}]"
    if kind == InstructionKind.RZZ:
        return f"rzz q[{instruction.control}], q[{instruction.target}]"
    if kind == InstructionKind.CCX:
        return f"ccx q[{instruction.controls[0]}], q[{instruction.controls[1]}], q[{instruction.target}]"
    # Use of `==` here for mypy
    if kind == InstructionKind._XC or kind == InstructionKind._ZC:  # noqa: PLR1714
        raise ValueError("Internal instruction should not appear")
    assert_never(kind)


class InstructionKind(Enum):
    """Tag for instruction kind."""

    CCX = enum.auto()
    RZZ = enum.auto()
    CNOT = enum.auto()
    SWAP = enum.auto()
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
    # The two following instructions are used internally by the transpiler
    _XC = enum.auto()
    _ZC = enum.auto()


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        utils.check_kind(cls, {"InstructionKind": InstructionKind, "Plane": Plane})


@dataclasses.dataclass
class CCX(_KindChecker):
    """Toffoli circuit instruction."""

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = dataclasses.field(default=InstructionKind.CCX, init=False)


@dataclasses.dataclass
class RZZ(_KindChecker):
    """RZZ circuit instruction."""

    target: int
    control: int
    angle: ExpressionOrFloat
    # FIXME: Remove `| None` from `meas_index`
    # - `None` makes codes messy/type-unsafe
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZZ]] = dataclasses.field(default=InstructionKind.RZZ, init=False)


@dataclasses.dataclass
class CNOT(_KindChecker):
    """CNOT circuit instruction."""

    target: int
    control: int
    kind: ClassVar[Literal[InstructionKind.CNOT]] = dataclasses.field(default=InstructionKind.CNOT, init=False)


@dataclasses.dataclass
class SWAP(_KindChecker):
    """SWAP circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = dataclasses.field(default=InstructionKind.SWAP, init=False)


@dataclasses.dataclass
class H(_KindChecker):
    """H circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.H]] = dataclasses.field(default=InstructionKind.H, init=False)


@dataclasses.dataclass
class S(_KindChecker):
    """S circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.S]] = dataclasses.field(default=InstructionKind.S, init=False)


@dataclasses.dataclass
class X(_KindChecker):
    """X circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.X]] = dataclasses.field(default=InstructionKind.X, init=False)


@dataclasses.dataclass
class Y(_KindChecker):
    """Y circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Y]] = dataclasses.field(default=InstructionKind.Y, init=False)


@dataclasses.dataclass
class Z(_KindChecker):
    """Z circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.Z]] = dataclasses.field(default=InstructionKind.Z, init=False)


@dataclasses.dataclass
class I(_KindChecker):
    """I circuit instruction."""

    target: int
    kind: ClassVar[Literal[InstructionKind.I]] = dataclasses.field(default=InstructionKind.I, init=False)


@dataclasses.dataclass
class M(_KindChecker):
    """M circuit instruction."""

    target: int
    plane: Plane
    angle: ExpressionOrFloat
    kind: ClassVar[Literal[InstructionKind.M]] = dataclasses.field(default=InstructionKind.M, init=False)


@dataclasses.dataclass
class RX(_KindChecker):
    """X rotation circuit instruction."""

    target: int
    angle: ExpressionOrFloat
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RX]] = dataclasses.field(default=InstructionKind.RX, init=False)


@dataclasses.dataclass
class RY(_KindChecker):
    """Y rotation circuit instruction."""

    target: int
    angle: ExpressionOrFloat
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RY]] = dataclasses.field(default=InstructionKind.RY, init=False)


@dataclasses.dataclass
class RZ(_KindChecker):
    """Z rotation circuit instruction."""

    target: int
    angle: ExpressionOrFloat
    meas_index: int | None = None
    kind: ClassVar[Literal[InstructionKind.RZ]] = dataclasses.field(default=InstructionKind.RZ, init=False)


@dataclasses.dataclass
class _XC(_KindChecker):
    """X correction circuit instruction. Used internally by the transpiler."""

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind._XC]] = dataclasses.field(default=InstructionKind._XC, init=False)


@dataclasses.dataclass
class _ZC(_KindChecker):
    """Z correction circuit instruction. Used internally by the transpiler."""

    target: int
    domain: set[int]
    kind: ClassVar[Literal[InstructionKind._ZC]] = dataclasses.field(default=InstructionKind._ZC, init=False)


if sys.version_info >= (3, 10):
    Instruction = CCX | RZZ | CNOT | SWAP | H | S | X | Y | Z | I | M | RX | RY | RZ | _XC | _ZC
else:
    Instruction = Union[CCX, RZZ, CNOT, SWAP, H, S, X, Y, Z, I, M, RX, RY, RZ, _XC, _ZC]

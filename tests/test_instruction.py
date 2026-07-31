from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import pytest

# override introduced in Python 3.12
from typing_extensions import override

from graphix import ANGLE_PI, Axis, Clifford
from graphix.instruction import Instruction, InstructionVisitor

if TYPE_CHECKING:
    from graphix.fundamentals import ParameterizedAngle
    from graphix.instruction import InstructionType

ALL_INSTRUCTIONS = [
    Instruction.CCX(target=0, controls=(1, 2)),
    Instruction.RZZ(target=0, control=1, angle=ANGLE_PI / 4),
    Instruction.CNOT(target=0, control=1),
    Instruction.SWAP(targets=(0, 1)),
    Instruction.CZ(targets=(0, 1)),
    Instruction.H(target=0),
    Instruction.S(target=0),
    Instruction.X(target=0),
    Instruction.Y(target=0),
    Instruction.Z(target=0),
    Instruction.I(target=0),
    Instruction.RX(target=0, angle=ANGLE_PI / 4),
    Instruction.RY(target=0, angle=ANGLE_PI / 4),
    Instruction.RZ(target=0, angle=ANGLE_PI / 4),
    Instruction.J(target=0, angle=ANGLE_PI / 4),
    Instruction.M(target=0, axis=Axis.X),
]


class VisitQubit(InstructionVisitor):
    @override
    def visit_qubit(self, qubit: int) -> int:
        return qubit + 1


class VisitAngle(InstructionVisitor):
    @override
    def visit_angle(self, angle: ParameterizedAngle) -> ParameterizedAngle:
        return -angle


class VisitAxis(InstructionVisitor):
    @override
    def visit_axis(self, axis: Axis) -> Axis:
        return axis.clifford(Clifford.H)


@pytest.mark.parametrize("instruction", ALL_INSTRUCTIONS)
def test_visit_qubit(instruction: InstructionType) -> None:
    # Copy the instruction to keep ALL_INSTRUCTIONS unmodified
    instr_copy = copy(instruction)
    visitor = VisitQubit()
    instr_visited = instr_copy.visit(visitor, copy=True)
    assert instr_copy == instruction
    assert instr_visited != instruction
    instr_copy.visit(visitor, copy=False)
    assert instr_copy != instruction
    assert instr_visited == instr_copy


@pytest.mark.parametrize("instruction", ALL_INSTRUCTIONS)
def test_visit_angle(instruction: InstructionType) -> None:
    if not hasattr(instruction, "angle"):
        pytest.skip()
    # Copy the instruction to keep ALL_INSTRUCTIONS unmodified
    instr_copy = copy(instruction)
    visitor = VisitAngle()
    instr_visited = instr_copy.visit(visitor, copy=True)
    assert instr_copy == instruction
    assert instr_visited != instruction
    instr_copy.visit(visitor, copy=False)
    assert instr_copy != instruction
    assert instr_visited == instr_copy


@pytest.mark.parametrize("instruction", ALL_INSTRUCTIONS)
def test_visit_axis(instruction: InstructionType) -> None:
    if not hasattr(instruction, "axis"):
        pytest.skip()
    instr_copy = copy(instruction)
    visitor = VisitAxis()
    instr_visited = instr_copy.visit(visitor, copy=True)
    assert instr_copy == instruction
    assert instr_visited != instruction
    instr_copy.visit(visitor, copy=False)
    assert instr_copy != instruction
    assert instr_visited == instr_copy

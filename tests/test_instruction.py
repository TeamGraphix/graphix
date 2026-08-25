from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

# override introduced in Python 3.12
from typing_extensions import override

from graphix import ANGLE_PI, Axis, Clifford, Instruction
from graphix.fundamentals import angle_to_rad
from graphix.instruction import InstructionVisitor
from graphix.ops import Ops

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator

    from graphix.fundamentals import ParameterizedAngle
    from graphix.instruction import InstructionType


@dataclass(frozen=True)
class InstructionTestCase:
    name: str
    instruction: Callable[[Generator], InstructionType]


INSTRUCTION_TEST_CASES: tuple[InstructionTestCase, ...] = (
    InstructionTestCase("CCX", lambda _rng: Instruction.CCX(0, (1, 2))),
    InstructionTestCase("RZZ", lambda rng: Instruction.RZZ(0, 1, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("CZ", lambda _rng: Instruction.CZ((0, 1))),
    InstructionTestCase("CNOT", lambda _rng: Instruction.CNOT(0, 1)),
    InstructionTestCase("SWAP", lambda _rng: Instruction.SWAP((0, 1))),
    InstructionTestCase("H", lambda _rng: Instruction.H(0)),
    InstructionTestCase("S", lambda _rng: Instruction.S(0)),
    InstructionTestCase("SDG", lambda _rng: Instruction.SDG(0)),
    InstructionTestCase("T", lambda _rng: Instruction.T(0)),
    InstructionTestCase("TDG", lambda _rng: Instruction.TDG(0)),
    InstructionTestCase("SX", lambda _rng: Instruction.SX(0)),
    InstructionTestCase("SXDG", lambda _rng: Instruction.SXDG(0)),
    InstructionTestCase("X", lambda _rng: Instruction.X(0)),
    InstructionTestCase("Y", lambda _rng: Instruction.Y(0)),
    InstructionTestCase("Z", lambda _rng: Instruction.Z(0)),
    InstructionTestCase("I", lambda _rng: Instruction.I(0)),
    InstructionTestCase("RX", lambda rng: Instruction.RX(0, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("RY", lambda rng: Instruction.RY(0, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("RZ", lambda rng: Instruction.RZ(0, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("J", lambda rng: Instruction.J(0, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("P", lambda rng: Instruction.P(0, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase(
        "U",
        lambda rng: Instruction.U(
            0, rng.random() * 2 * ANGLE_PI, rng.random() * 2 * ANGLE_PI, rng.random() * 2 * ANGLE_PI
        ),
    ),
    InstructionTestCase("CY", lambda _rng: Instruction.CY(0, 1)),
    InstructionTestCase("CJ", lambda rng: Instruction.CJ(0, 1, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("CP", lambda rng: Instruction.CP(0, 1, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("CRX", lambda rng: Instruction.CRX(0, 1, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("CRY", lambda rng: Instruction.CRY(0, 1, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase("CRZ", lambda rng: Instruction.CRZ(0, 1, rng.random() * 2 * ANGLE_PI)),
    InstructionTestCase(
        "CU",
        lambda rng: Instruction.CU(
            0,
            1,
            rng.random() * 2 * ANGLE_PI,
            rng.random() * 2 * ANGLE_PI,
            rng.random() * 2 * ANGLE_PI,
            rng.random() * 2 * ANGLE_PI,
        ),
    ),
    InstructionTestCase("CSWAP", lambda _rng: Instruction.CSWAP(0, (1, 2))),
)


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


@pytest.mark.parametrize("test_case", INSTRUCTION_TEST_CASES)
def test_visit_qubit(fx_rng: Generator, test_case: InstructionTestCase) -> None:
    instr = test_case.instruction(fx_rng)
    instr_copy = copy(instr)
    visitor = VisitQubit()
    instr_visited = instr.visit(visitor, copy=True)
    assert instr == instr_copy
    assert instr_visited != instr_copy
    instr.visit(visitor, copy=False)
    assert instr != instr_copy
    assert instr_visited == instr


@pytest.mark.parametrize("test_case", INSTRUCTION_TEST_CASES)
def test_visit_angle(fx_rng: Generator, test_case: InstructionTestCase) -> None:
    instr = test_case.instruction(fx_rng)
    if not hasattr(instr, "angle"):
        pytest.skip()
    instr_copy = copy(instr)
    visitor = VisitAngle()
    instr_visited = instr.visit(visitor, copy=True)
    assert instr == instr_copy
    assert instr_visited != instr_copy
    instr.visit(visitor, copy=False)
    assert instr != instr_copy
    assert instr_visited == instr


@pytest.mark.parametrize("test_case", INSTRUCTION_TEST_CASES)
def test_visit_axis(fx_rng: Generator, test_case: InstructionTestCase) -> None:
    instr = test_case.instruction(fx_rng)
    if not hasattr(instr, "axis"):
        pytest.skip()
    instr_copy = copy(instr)
    visitor = VisitAxis()
    instr_visited = instr.visit(visitor, copy=True)
    assert instr == instr_copy
    assert instr_visited != instr_copy
    instr.visit(visitor, copy=False)
    assert instr != instr_copy
    assert instr_visited == instr


def test_u(fx_rng: Generator) -> None:
    theta = fx_rng.random()
    phi = fx_rng.random()
    lambda_ = fx_rng.random()
    np.testing.assert_allclose(
        Ops.u(theta, phi, lambda_),
        np.exp(-1j * angle_to_rad(theta) / 2)
        * (Ops.H @ Ops.j(phi + ANGLE_PI / 2) @ Ops.j(theta) @ Ops.j(lambda_ - ANGLE_PI / 2)),
    )


def test_cj(fx_rng: Generator) -> None:
    alpha = fx_rng.random()
    delta = (alpha + ANGLE_PI) / 2
    a = Ops.ry(ANGLE_PI / 4)
    b = Ops.ry(-ANGLE_PI / 4) @ Ops.rz(-delta)
    c = Ops.rz(delta)
    np.testing.assert_allclose(a @ b @ c, Ops.I, atol=1e-15)
    np.testing.assert_allclose(a @ Ops.X @ b @ Ops.X @ c, np.exp(-1j * angle_to_rad(delta)) * Ops.j(alpha))
    np.testing.assert_allclose(
        Ops.cj(alpha),
        np.kron(Ops.p(delta), Ops.I) @ np.kron(Ops.I, a) @ Ops.CNOT @ np.kron(Ops.I, b) @ Ops.CNOT @ np.kron(Ops.I, c),
        atol=1e-15,
    )


def test_cu(fx_rng: Generator) -> None:
    theta = fx_rng.random()
    phi = fx_rng.random()
    lambda_ = fx_rng.random()
    gamma = fx_rng.random()
    np.testing.assert_allclose(
        Ops.cu(theta, phi, lambda_, gamma),
        np.kron(Ops.p(gamma - theta / 2), Ops.I)
        @ Ops.cj(0)
        @ Ops.cj(phi + ANGLE_PI / 2)
        @ Ops.cj(theta)
        @ Ops.cj(lambda_ - ANGLE_PI / 2),
    )

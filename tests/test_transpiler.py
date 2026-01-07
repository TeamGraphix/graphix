from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix import instruction
from graphix.branch_selector import ConstBranchSelector
from graphix.fundamentals import ANGLE_PI, Axis, Sign
from graphix.instruction import InstructionKind
from graphix.random_objects import rand_circuit, rand_gate, rand_state_vector
from graphix.states import BasicStates
from graphix.transpiler import Circuit
from tests.test_branch_selector import CheckedBranchSelector

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from graphix.instruction import Instruction
    from graphix.measurements import Outcome

    InstructionTestCase: TypeAlias = Callable[[Generator], Instruction]

INSTRUCTION_TEST_CASES: list[InstructionTestCase] = [
    lambda _rng: instruction.CCX(0, (1, 2)),
    lambda rng: instruction.RZZ(0, 1, rng.random() * 2 * ANGLE_PI),
    lambda _rng: instruction.CZ((0, 1)),
    lambda _rng: instruction.CNOT(0, 1),
    lambda _rng: instruction.SWAP((0, 1)),
    lambda _rng: instruction.H(0),
    lambda _rng: instruction.S(0),
    lambda _rng: instruction.X(0),
    lambda _rng: instruction.Y(0),
    lambda _rng: instruction.Z(0),
    lambda _rng: instruction.I(0),
    lambda rng: instruction.RX(0, rng.random() * 2 * ANGLE_PI),
    lambda rng: instruction.RY(0, rng.random() * 2 * ANGLE_PI),
    lambda rng: instruction.RZ(0, rng.random() * 2 * ANGLE_PI),
]


class TestTranspilerUnitGates:
    def test_cz(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        circuit.cz(0, 1)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_cnot(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_hadamard(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_s(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_x(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_y(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_z(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_rx(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * ANGLE_PI
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_ry(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * ANGLE_PI
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_rz(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * ANGLE_PI
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_i(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_ccx(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 4
        depth = 6
        circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        state = circuit.simulate_statevector(rng=rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_transpiled(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rand_gate(nqubits, depth, pairs, fx_rng, use_rzz=True)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_measure(self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(2)
        circuit.m(0, axis)
        input_state = rand_state_vector(2, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate_statevector(rng=rng, input_state=input_state, branch_selector=branch_selector).statevec
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate_pattern(rng=rng, input_state=input_state, branch_selector=branch_selector)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("input_axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("input_sign", [Sign.PLUS, Sign.MINUS])
    @pytest.mark.parametrize("measurement_axis", [Axis.X, Axis.Y, Axis.Z])
    def test_measurement_expectation_value(self, input_axis: Axis, input_sign: Sign, measurement_axis: Axis) -> None:
        match input_axis, input_sign:
            case Axis.X, Sign.PLUS:
                input_state = BasicStates.PLUS
            case Axis.X, Sign.MINUS:
                input_state = BasicStates.MINUS
            case Axis.Y, Sign.PLUS:
                input_state = BasicStates.PLUS_I
            case Axis.Y, Sign.MINUS:
                input_state = BasicStates.MINUS_I
            case Axis.Z, Sign.PLUS:
                input_state = BasicStates.ZERO
            case Axis.Z, Sign.MINUS:
                input_state = BasicStates.ONE
        circuit = Circuit(1)
        circuit.m(0, measurement_axis)
        expectation_value0 = 0.5 if input_axis != measurement_axis else 1 if input_sign == Sign.PLUS else 0
        branch_selector = CheckedBranchSelector(expected={0: expectation_value0}, abs_tol=1e-15)
        circuit.simulate_statevector(input_state=input_state, branch_selector=branch_selector)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_transpile_measurements_to_z_axis(self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(2)
        circuit.m(0, axis)
        input_state = rand_state_vector(2, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate_statevector(rng=rng, input_state=input_state, branch_selector=branch_selector).statevec
        circuit_z = circuit.transpile_measurements_to_z_axis()
        assert all(instr.axis == Axis.Z for instr in circuit_z.instruction if instr.kind == InstructionKind.M)
        state_z = circuit.simulate_statevector(
            rng=rng, input_state=input_state, branch_selector=branch_selector
        ).statevec
        assert np.abs(np.dot(state_z.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_add_extend(self) -> None:
        circuit = Circuit(3)
        circuit.ccx(0, 1, 2)
        circuit.rzz(0, 1, 2)
        circuit.cz(0, 1)
        circuit.cnot(0, 1)
        circuit.swap(0, 1)
        circuit.h(0)
        circuit.s(0)
        circuit.x(0)
        circuit.y(0)
        circuit.z(0)
        circuit.i(0)
        circuit.m(0, Axis.X)
        circuit.rx(1, 0.5)
        circuit.ry(2, 0.5)
        circuit.rz(1, 0.5)
        circuit2 = Circuit(3, instr=circuit.instruction)
        assert circuit.instruction == circuit2.instruction

    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instruction_flow(self, fx_rng: Generator, instruction: InstructionTestCase) -> None:
        circuit = Circuit(3, instr=[instruction(fx_rng)])
        pattern = circuit.transpile().pattern
        flow = pattern.extract_causal_flow()
        flow.check_well_formed()

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instructions(self, fx_bg: PCG64, jumps: int, instruction: InstructionTestCase) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(3, instr=[instruction(rng)])
        pattern = circuit.transpile().pattern
        input_state = rand_state_vector(3, rng=rng)
        state = circuit.simulate_statevector(input_state=input_state).statevec
        state_mbqc = pattern.simulate_pattern(input_state=input_state, rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

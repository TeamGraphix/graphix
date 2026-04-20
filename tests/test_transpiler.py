from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix import instruction
from graphix.branch_selector import ConstBranchSelector, FixedBranchSelector
from graphix.fundamentals import ANGLE_PI, Axis, Sign
from graphix.instruction import I, InstructionKind
from graphix.random_objects import rand_circuit, rand_gate, rand_state_vector
from graphix.simulator import DefaultMeasureMethod
from graphix.states import BasicStates
from graphix.transpiler import Circuit, decompose_ccx, transpile_swaps
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
    lambda rng: instruction.J(0, rng.random() * 2 * ANGLE_PI),
]


class TestTranspilerUnitGates:
    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instruction_flow(self, fx_rng: Generator, instruction: InstructionTestCase) -> None:
        circuit = Circuit(3, instr=[instruction(fx_rng)])
        pattern = circuit.transpile().pattern
        flow = pattern.to_bloch().extract_causal_flow()
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
        assert state_mbqc.isclose(state)

    def test_simple(self) -> None:
        rng = np.random.default_rng(420)
        circuit = Circuit(3, instr=[instruction.CCX(0, (1, 2))])
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        input_state = rand_state_vector(3, rng=rng)
        state = circuit.simulate_statevector(input_state=input_state).statevec
        state_mbqc = pattern.simulate_pattern(input_state=input_state, rng=rng)
        assert state_mbqc.isclose(state)

    def test_transpiled(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rand_gate(nqubits, depth, pairs, fx_rng, use_rzz=True)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert state_mbqc.isclose(state)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_measure(self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        circuit.m(0, axis)
        input_state = rand_state_vector(2, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate_statevector(rng=rng, input_state=input_state, branch_selector=branch_selector).statevec
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate_pattern(rng=rng, input_state=input_state, branch_selector=branch_selector)
        assert state_mbqc.isclose(state)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_measure_early(self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(3)
        circuit.m(0, axis)
        circuit.cnot(1, 2)
        input_state = rand_state_vector(3, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate_statevector(rng=rng, input_state=input_state, branch_selector=branch_selector).statevec
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate_pattern(rng=rng, input_state=input_state, branch_selector=branch_selector)
        assert state_mbqc.isclose(state)

    @pytest.mark.parametrize("input_axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("input_sign", [Sign.PLUS, Sign.MINUS])
    @pytest.mark.parametrize("measurement_axis", [Axis.X, Axis.Y, Axis.Z])
    def test_measurement_expectation_value(
        self, fx_rng: Generator, input_axis: Axis, input_sign: Sign, measurement_axis: Axis
    ) -> None:
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
        circuit.simulate_statevector(input_state=input_state, branch_selector=branch_selector, rng=fx_rng)

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
        assert state_z.isclose(state)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_transpile_swaps(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 4
        depth = 6
        circuit = rand_circuit(nqubits, depth, rng, use_ccx=True, use_rzz=True)
        assert any(instr.kind == InstructionKind.SWAP for instr in circuit.instruction)
        transpiled_swaps = transpile_swaps(circuit)
        circuit2 = transpiled_swaps.circuit
        assert not any(instr.kind == InstructionKind.SWAP for instr in circuit2.instruction)
        state = circuit.simulate_statevector(rng=rng).statevec
        state2 = circuit2.simulate_statevector(rng=rng).statevec
        qubits: list[int] = []
        for qubit in transpiled_swaps.qubits:
            assert qubit is not None
            qubits.append(qubit)
        state2.psi = np.transpose(state2.psi, qubits)
        assert state.isclose(state2)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_transpile_swaps_with_measurements(self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(3)
        circuit.swap(0, 1)
        circuit.swap(0, 2)
        circuit.cnot(1, 2)
        circuit.m(1, axis)
        circuit.i(0)
        transpiled_swaps = transpile_swaps(circuit)
        circuit2 = transpiled_swaps.circuit
        assert not any(instr.kind == InstructionKind.SWAP for instr in circuit2.instruction)
        assert I(2) in circuit2.instruction
        input_state = rand_state_vector(3, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate_statevector(rng=rng, input_state=input_state, branch_selector=branch_selector).statevec
        state2 = circuit2.simulate_statevector(
            rng=rng, input_state=input_state, branch_selector=branch_selector
        ).statevec
        assert transpiled_swaps.qubits == (2, None, 1)
        state2.swap((0, 1))
        assert state.isclose(state2)

    def test_cz_ccx(self) -> None:
        """Test case reported in issue #2.

        https://github.com/qat-inria/graphix-jcz-transpiler/issues/2
        """
        circuit = Circuit(width=3)
        circuit.cz(2, 0)
        circuit.ccx(0, 1, 2)
        ref_state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        state = pattern.simulate_pattern()
        assert state.isclose(ref_state)

    def test_ccx_decomposition(self) -> None:
        circuit = Circuit(width=3)
        circuit.cz(2, 0)
        circuit.ccx(0, 1, 2)
        circuit2 = Circuit(width=3)
        circuit2.cz(2, 0)
        circuit2.extend(decompose_ccx(instruction.CCX(controls=(0, 1), target=2)))
        state = circuit.simulate_statevector().statevec
        state2 = circuit2.simulate_statevector().statevec
        assert state.isclose(state2)

    def test_cnot_cz(self) -> None:
        """Test regression about output node reordering."""
        circuit = Circuit(width=3, instr=[instruction.CNOT(0, 1), instruction.CZ((0, 1))])
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate_pattern()
        assert state.isclose(state_mbqc)

    @pytest.mark.parametrize("jumps", range(1, 6))
    @pytest.mark.parametrize("axes", [[Axis.X, Axis.Y], [Axis.X, Axis.Y, Axis.Z]])
    def test_classical_outputs_consistency(self, fx_bg: PCG64, jumps: int, axes: list[Axis]) -> None:
        """Check that `classical_outputs` are in the same order as `classical_measures`."""
        rng = Generator(fx_bg.jumped(jumps))
        n = len(axes)
        width = n + 1
        circuit = Circuit(width)
        for q in range(n):
            circuit.cnot(q, q + 1)
        for q, axis in enumerate(axes):
            circuit.m(q, axis)

        transpile_result = circuit.transpile()
        pattern = transpile_result.pattern
        expected_outcomes: list[Outcome] = [1 if q % 2 else 0 for q in range(n)]
        results_circuit: dict[int, Outcome] = dict(zip(range(n), expected_outcomes, strict=False))
        m_outcomes = dict(zip(transpile_result.classical_outputs, expected_outcomes, strict=False))
        non_output_nodes = pattern.extract_nodes() - set(pattern.output_nodes)
        results_pattern: dict[int, Outcome] = {node: m_outcomes.get(node, 0) for node in non_output_nodes}
        input_state = rand_state_vector(width, rng=rng)
        measure_method = DefaultMeasureMethod()
        circuit_result = circuit.simulate_statevector(
            rng=rng,
            input_state=input_state,
            branch_selector=FixedBranchSelector(results=results_circuit),
        )
        pattern.simulate_pattern(
            rng=rng,
            input_state=input_state,
            branch_selector=FixedBranchSelector(results=results_pattern),
            measure_method=measure_method,
        )
        assert len(transpile_result.classical_outputs) == len(circuit_result.classical_measures)
        pattern_measures = [measure_method.results[node] for node in transpile_result.classical_outputs]
        assert pattern_measures == list(circuit_result.classical_measures)
        assert pattern_measures == expected_outcomes

    def test_classical_outputs_empty(self) -> None:
        """Circuits with no M instructions produce empty classical_outputs."""
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        circuit.h(0)
        result = circuit.transpile()
        assert len(result.classical_outputs) == 0
        assert len(circuit.simulate_statevector().classical_measures) == 0


class TestCircuits:
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

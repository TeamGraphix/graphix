from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix import instruction
from graphix.branch_selector import ConstBranchSelector, FixedBranchSelector
from graphix.fundamentals import ANGLE_PI, Axis, Plane, Sign
from graphix.instruction import I, InstructionKind
from graphix.random_objects import rand_circuit, rand_gate, rand_state_vector
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevector, StatevectorBackend
from graphix.simulator import DefaultMeasureMethod
from graphix.states import BasicStates, PlanarState
from graphix.transpiler import Circuit, OutputIndex, OutputKind, decompose_ccx, transpile_swaps
from tests.test_branch_selector import CheckedBranchSelector
from tests.test_instruction import VisitAngle

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, TypeAlias

    from graphix.instruction import InstructionType
    from graphix.measurements import Outcome
    from graphix.states import State

    InstructionTestCase: TypeAlias = Callable[[Generator], InstructionType]
    _DenseStateBackendLiteral = Literal["statevector", "densitymatrix"]


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
    @pytest.mark.parametrize("with_ancillas", [False, True])
    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instruction_flow(self, fx_rng: Generator, instruction: InstructionTestCase, with_ancillas: bool) -> None:
        instr = [instruction(fx_rng)]
        circuit = Circuit(1, instr=instr, ancillas=2) if with_ancillas else Circuit(3, instr=instr)
        pattern = circuit.transpile().pattern
        circuit.transpile_to_causalflow().flow.check_well_formed()
        flow = pattern.to_bloch().to_causalflow()
        flow.check_well_formed()

    @pytest.mark.parametrize("with_ancillas", [False, True])
    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instructions(
        self, fx_bg: PCG64, jumps: int, instruction: InstructionTestCase, with_ancillas: bool
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        instr = [instruction(rng)]
        circuit = Circuit(1, instr=instr, ancillas=2) if with_ancillas else Circuit(3, instr=instr)
        pattern = circuit.transpile().pattern
        input_state = rand_state_vector(1, rng=rng) if with_ancillas else rand_state_vector(3, rng=rng)
        state = circuit.simulate(input_state=input_state).state
        state_mbqc = pattern.simulate(input_state=input_state, rng=rng)
        assert state_mbqc.isclose(state)

    def test_cond_instr(self) -> None:
        circuit = Circuit(2)
        circuit.m(0, Axis.Z)
        circuit.cond_instr([instruction.H(1)], {0})
        with pytest.raises(NotImplementedError):
            circuit.transpile()

    @pytest.mark.parametrize(
        "ancilla_state",
        [
            BasicStates.PLUS,
            BasicStates.MINUS,
            BasicStates.ZERO,
            BasicStates.ONE,
            BasicStates.PLUS_I,
            BasicStates.MINUS_I,
        ],
    )
    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instruction_ancilla_state(
        self, fx_rng: Generator, instruction: InstructionTestCase, ancilla_state: State
    ) -> None:
        instr = [instruction(fx_rng)]
        circuit = Circuit(1, instr=instr, ancillas=2, ancilla_state=ancilla_state)

        circuit.transpile_to_causalflow().flow.check_well_formed()

        pattern = circuit.transpile().pattern
        input_state = rand_state_vector(1, rng=fx_rng)
        state = circuit.simulate(input_state=input_state).state
        state_mbqc = pattern.simulate(input_state=input_state, rng=fx_rng)
        assert state_mbqc.isclose(state)

    def test_transpiled(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rand_gate(nqubits, depth, pairs, fx_rng, use_rzz=True)
        pattern = circuit.transpile().pattern
        state = circuit.simulate(rng=fx_rng).state
        state_mbqc = pattern.simulate(rng=fx_rng)
        assert state_mbqc.isclose(state)

    @pytest.mark.parametrize("with_ancillas", [False, True])
    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_measure(
        self,
        fx_bg: PCG64,
        jumps: int,
        axis: Axis,
        outcome: Outcome,
        backend: _DenseStateBackendLiteral,
        with_ancillas: bool,
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(2, ancillas=2) if with_ancillas else Circuit(2)
        circuit.cnot(0, 1)
        circuit.m(0, axis)
        if with_ancillas:
            circuit.cnot(1, 2)
            circuit.cnot(2, 3)
            if axis is Axis.X and outcome == 1:
                circuit.ry(
                    3, 0.2
                )  # This rotation is crucial to avoid errors measuring |+> along X and selecting outcome 1.
            circuit.m(3, axis)
        input_state = rand_state_vector(2, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate(
            rng=rng, input_state=input_state, branch_selector=branch_selector, backend=backend
        ).state
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate(
            rng=rng, input_state=input_state, branch_selector=branch_selector, backend=backend
        )
        if isinstance(state_mbqc, Statevector) and isinstance(state, Statevector):
            assert state_mbqc.isclose(state)
        elif isinstance(state_mbqc, DensityMatrix) and isinstance(state, DensityMatrix):
            assert np.allclose(state_mbqc.rho, state.rho)

    @pytest.mark.parametrize(
        "ancilla_state",
        [
            BasicStates.PLUS,
            BasicStates.MINUS,
            BasicStates.ZERO,
            BasicStates.ONE,
            BasicStates.PLUS_I,
            BasicStates.MINUS_I,
        ],
    )
    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_measure_ancilla_state(
        self,
        fx_bg: PCG64,
        jumps: int,
        axis: Axis,
        outcome: Outcome,
        backend: _DenseStateBackendLiteral,
        ancilla_state: State,
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(2, ancillas=2, ancilla_state=ancilla_state)
        circuit.cnot(0, 1)
        circuit.m(0, axis)
        circuit.cnot(1, 2)
        circuit.cnot(2, 3)
        # Rotations of ancilla qubit are crucial to avoid issues with selecting state with 0 probability.
        circuit.rx(3, 0.1)
        circuit.ry(3, 0.2)
        circuit.rz(3, 0.3)
        circuit.m(3, axis)
        input_state = rand_state_vector(2, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate(
            rng=rng, input_state=input_state, branch_selector=branch_selector, backend=backend
        ).state
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate(
            rng=rng, input_state=input_state, branch_selector=branch_selector, backend=backend
        )
        if isinstance(state_mbqc, Statevector) and isinstance(state, Statevector):
            assert state_mbqc.isclose(state)
        elif isinstance(state_mbqc, DensityMatrix) and isinstance(state, DensityMatrix):
            assert np.allclose(state_mbqc.rho, state.rho)

    @pytest.mark.parametrize("with_ancillas", [False, True])
    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_measure_early(self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome, with_ancillas: bool) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(3, ancillas=2) if with_ancillas else Circuit(3)
        circuit.m(0, axis)
        if with_ancillas:
            if axis is Axis.X and outcome == 1:
                circuit.ry(3, 0.2)
            circuit.m(3, axis)
        circuit.cnot(1, 2)
        if with_ancillas:
            circuit.cnot(2, 4)
        input_state = rand_state_vector(3, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate(rng=rng, input_state=input_state, branch_selector=branch_selector).state
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate(rng=rng, input_state=input_state, branch_selector=branch_selector)
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
        circuit.simulate(input_state=input_state, branch_selector=branch_selector, rng=fx_rng)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_transpile_measurements_to_z_axis(self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(2)
        circuit.m(0, axis)
        input_state = rand_state_vector(2, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate(rng=rng, input_state=input_state, branch_selector=branch_selector).state
        circuit_z = circuit.transpile_measurements_to_z_axis()
        assert all(instr.axis == Axis.Z for instr in circuit_z.instruction if instr.kind == InstructionKind.M)
        state_z = circuit.simulate(rng=rng, input_state=input_state, branch_selector=branch_selector).state
        assert state_z.isclose(state)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_transpile_j_to_rzh(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 2
        circuit = rand_circuit(nqubits, depth, rng, use_j=True, use_ccx=True, use_rzz=True)
        circuit.j(0, 0.5)  # Ensure that there is at least one J instruction
        assert any(instr.kind == InstructionKind.J for instr in circuit.instruction)
        circuit2 = circuit.transpile_j_to_rzh()
        assert not any(instr.kind == InstructionKind.J for instr in circuit2.instruction)
        state = circuit.simulate(rng=rng).state
        state2 = circuit2.simulate(rng=rng).state
        assert state.fidelity(state2) == pytest.approx(1)

    @pytest.mark.parametrize("with_ancillas", [False, True])
    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_transpile_swaps_with_measurements(
        self, fx_bg: PCG64, jumps: int, axis: Axis, outcome: Outcome, with_ancillas: bool
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = Circuit(3, ancillas=2) if with_ancillas else Circuit(3)
        circuit.swap(0, 1)
        circuit.swap(0, 2)
        circuit.cnot(1, 2)
        circuit.m(1, axis)
        circuit.i(0)
        if with_ancillas:
            circuit.swap(2, 3)
            circuit.swap(3, 4)
            if axis is Axis.X and outcome == 1:
                circuit.ry(3, 0.2)
            circuit.m(3, axis)

        transpiled_swaps = transpile_swaps(circuit, copy=True)
        circuit2 = transpiled_swaps.circuit
        assert not any(instr.kind == InstructionKind.SWAP for instr in circuit2.instruction)
        assert I(2) in circuit2.instruction
        assert any(instr.kind == InstructionKind.SWAP for instr in circuit.instruction)
        circuit_copy = deepcopy(circuit)
        transpile_swaps(circuit_copy)
        assert circuit_copy.instruction == circuit2.instruction
        input_state = rand_state_vector(3, rng=rng)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate(rng=rng, input_state=input_state, branch_selector=branch_selector).state
        state2 = circuit2.simulate(rng=rng, input_state=input_state, branch_selector=branch_selector).state
        if not with_ancillas:
            assert transpiled_swaps.outputs == (
                OutputIndex(OutputKind.Qubit, 2),
                OutputIndex(OutputKind.Bit, 0),
                OutputIndex(OutputKind.Qubit, 1),
            )
            assert transpiled_swaps.extract_output_node_indices() == (1, 0)
            state2.swap((0, 1))
        else:
            assert transpiled_swaps.outputs == (
                OutputIndex(OutputKind.Qubit, 2),
                OutputIndex(OutputKind.Bit, 0),
                OutputIndex(OutputKind.Qubit, 3),
                OutputIndex(OutputKind.Bit, 1),
                OutputIndex(OutputKind.Qubit, 1),
            )
            assert transpiled_swaps.extract_output_node_indices() == (1, 2, 0)
            state2.swap((0, 1))
            state2.swap((1, 2))

        assert state.isclose(state2)

    def test_cz_ccx(self, fx_rng: Generator) -> None:
        """Test case reported in issue #2.

        https://github.com/qat-inria/graphix-jcz-transpiler/issues/2
        """
        circuit = Circuit(width=3)
        circuit.cz(2, 0)
        circuit.ccx(0, 1, 2)
        ref_state = circuit.simulate(rng=fx_rng).state
        pattern = circuit.transpile().pattern
        state = pattern.simulate(rng=fx_rng)
        assert state.isclose(ref_state)

    def test_ccx_decomposition(self) -> None:
        circuit = Circuit(width=3)
        circuit.cz(2, 0)
        circuit.ccx(0, 1, 2)
        circuit2 = Circuit(width=3)
        circuit2.cz(2, 0)
        circuit2.extend(decompose_ccx(instruction.CCX(controls=(0, 1), target=2)))
        state = circuit.simulate().state
        state2 = circuit2.simulate().state
        assert state.isclose(state2)

    def test_cnot_cz(self, fx_rng: Generator) -> None:
        """Test regression about output node reordering."""
        circuit = Circuit(width=3, instr=[instruction.CNOT(0, 1), instruction.CZ((0, 1))])
        state = circuit.simulate(rng=fx_rng).state
        pattern = circuit.transpile().pattern
        state_mbqc = pattern.simulate(rng=fx_rng)
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
        non_output_nodes = pattern.nodes() - set(pattern.output_nodes)
        results_pattern: dict[int, Outcome] = {node: m_outcomes.get(node, 0) for node in non_output_nodes}
        input_state = rand_state_vector(width, rng=rng)
        measure_method = DefaultMeasureMethod()
        circuit_result = circuit.simulate(
            rng=rng,
            input_state=input_state,
            branch_selector=FixedBranchSelector(results=results_circuit),
        )
        pattern.simulate(
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
        assert len(circuit.simulate().classical_measures) == 0


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

    @pytest.mark.parametrize(
        ("instruction", "msg"),
        [
            (instruction.S(0), r"Qubit 0 is not an active qubit."),
            (instruction.RZ(4, 0.3), r"Qubit 4 is not an active qubit."),
            (instruction.CNOT(0, 1), r"Qubit 0 is not an active qubit."),
            (instruction.CONDINSTR((instruction.H(0), instruction.Z(2)), {0}), r"Qubit 0 is not an active qubit."),
            (instruction.CONDINSTR((instruction.H(1), instruction.X(0)), {0}), r"Qubit 0 is not an active qubit."),
            (
                instruction.CONDINSTR((instruction.H(1), instruction.X(2), instruction.SWAP((1, 4))), {0}),
                r"Qubit 4 is not an active qubit.",
            ),
            (instruction.CONDINSTR((instruction.RX(2, 0.2),), {0, 1}), r"Qubit 1 is not a measured qubit."),
            (instruction.CONDINSTR((instruction.J(1, 0.2),), {0, 4}), r"Qubit 4 is not a measured qubit."),
        ],
    )
    def test_wrong_qubits(self, instruction: InstructionType, msg: str) -> None:
        circuit = Circuit(3)
        circuit.m(0, Axis.X)
        with pytest.raises(ValueError, match=msg):
            circuit.add(instruction)

    @pytest.mark.parametrize(
        ("instruction", "msg"),
        [
            (instruction.CCX(0, (1, 1)), r"Control qubits cannot be the same. Qubit index: 1"),
            (instruction.CCX(0, (0, 1)), r"Target and control-0 qubits cannot be the same. Qubit index: 0"),
            (instruction.RZZ(2, 2, 0.3), r"Target and control qubits cannot be the same. Qubit index: 2"),
            (instruction.CZ((1, 1)), r"Target qubits cannot be the same. Qubit index: 1"),
            (instruction.CNOT(0, 0), r"Target and control qubits cannot be the same. Qubit index: 0"),
            (instruction.SWAP((1, 1)), r"Target qubits cannot be the same. Qubit index: 1"),
            (instruction.CONDINSTR((instruction.SWAP((1, 1)),)), r"Target qubits cannot be the same. Qubit index: 1"),
        ],
    )
    def test_repeated_qubits(self, instruction: InstructionType, msg: str) -> None:
        circuit = Circuit(3)
        with pytest.raises(ValueError, match=msg):
            circuit.add(instruction)

    def test_simple(self) -> None:
        rng = np.random.default_rng(420)
        circuit = Circuit(3, instr=[instruction.CCX(0, (1, 2))])
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        input_state = rand_state_vector(3, rng=rng)
        state = circuit.simulate(input_state=input_state).state
        state_mbqc = pattern.simulate(input_state=input_state, rng=rng)
        assert state_mbqc.isclose(state)

    def test_simple_ancillas(self, fx_rng: Generator) -> None:
        circuit = Circuit(1, ancillas=1)
        circuit.x(0)
        circuit.h(1)
        state = circuit.simulate(rng=fx_rng, input_state=BasicStates.ZERO).state
        state_ref = Statevector([BasicStates.ONE, BasicStates.ZERO])
        assert state.isclose(state_ref)

    @pytest.mark.parametrize(
        "ancilla_state",
        [
            BasicStates.PLUS,
            BasicStates.MINUS,
            BasicStates.ZERO,
            BasicStates.ONE,
            BasicStates.PLUS_I,
            BasicStates.MINUS_I,
            PlanarState(Plane.XY, 0.3),
        ],
    )
    def test_ancilla_state(self, fx_rng: Generator, ancilla_state: State) -> None:
        circuit = Circuit(1, ancillas=1, ancilla_state=ancilla_state)
        circuit.cz(0, 1)
        state = circuit.simulate(rng=fx_rng, input_state=BasicStates.ZERO).state
        state_ref = Statevector([BasicStates.ZERO, ancilla_state])
        assert state.isclose(state_ref)

    @pytest.mark.parametrize(
        ("domain", "outcome", "output"),
        [
            (set(), 0, BasicStates.ZERO),
            (set(), 1, BasicStates.ZERO),
            ({0}, 0, BasicStates.ZERO),
            ({0}, 1, BasicStates.MINUS),
            ({0, 1}, 0, BasicStates.ZERO),
            ({0, 1}, 1, BasicStates.ZERO),
        ],
    )
    def test_cond_instr(self, fx_rng: Generator, domain: set[int], outcome: Outcome, output: State) -> None:
        circuit = Circuit(3)
        circuit.m(0, Axis.Z)
        circuit.m(1, Axis.Z)
        circuit.cond_instr((instruction.H(2), instruction.Z(2)), domain)
        branch_selector = ConstBranchSelector(outcome)
        state = circuit.simulate(
            rng=fx_rng,
            input_state=[BasicStates.PLUS, BasicStates.PLUS, BasicStates.ZERO],
            branch_selector=branch_selector,
        ).state
        assert state.isclose(Statevector(nqubit=1, data=output))

    def test_ancilla_error(self) -> None:
        state = PlanarState(Plane.XY, 0.3)
        circuit = Circuit(1, ancillas=1, ancilla_state=state)
        circuit.cnot(0, 1)
        with pytest.raises(NotImplementedError):
            circuit.transpile()

    @pytest.mark.parametrize("jumps", range(1, 3))
    def test_dm_backend(self, fx_bg: PCG64, jumps: int) -> None:
        nqubits = 2
        rng = Generator(fx_bg.jumped(jumps))
        circuit = rand_circuit(nqubits, 3, rng)
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        input_state = rand_state_vector(nqubits, rng=rng)
        state = circuit.simulate(input_state=input_state, backend="densitymatrix").state
        state_mbqc = pattern.simulate(input_state=input_state, backend="densitymatrix", rng=rng)
        assert np.allclose(state_mbqc.rho, state.rho)

    def test_dm_backend_with_ancillas(self, fx_rng: Generator) -> None:
        circuit = Circuit(1, ancillas=1)
        circuit.x(0)
        circuit.cnot(0, 1)
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        input_state = rand_state_vector(1, rng=fx_rng)
        state = circuit.simulate(input_state=input_state, backend="densitymatrix").state
        state_mbqc = pattern.simulate(input_state=input_state, backend="densitymatrix", rng=fx_rng)
        assert np.allclose(state_mbqc.rho, state.rho)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_transpile_swaps(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True, use_rzz=True)
    assert any(instr.kind == InstructionKind.SWAP for instr in circuit.instruction)
    transpiled_swaps = transpile_swaps(circuit, copy=True)
    circuit2 = transpiled_swaps.circuit
    assert not any(instr.kind == InstructionKind.SWAP for instr in circuit2.instruction)
    state = circuit.simulate(rng=rng).state
    state2 = circuit2.simulate(rng=rng).state
    state2.permute(transpiled_swaps.extract_output_node_indices())
    assert state.isclose(state2)


def test_transpile_double_cz() -> None:
    circuit = Circuit(2)
    circuit.cz(0, 1)
    circuit.cz(1, 0)
    cf = circuit.transpile_to_causalflow()
    assert len(cf.flow.og.graph.edges) == 0


@pytest.mark.parametrize("with_ancillas", [False, True])
def test_transpile_swaps_vs_no_transpile_swaps(fx_rng: Generator, with_ancillas: bool) -> None:
    circuit = Circuit(2, ancillas=1) if with_ancillas else Circuit(2)
    circuit.rx(0, 0.25)
    circuit.ry(0, 0.25)
    circuit.cz(0, 1)
    circuit.swap(0, 1)
    if with_ancillas:
        circuit.rz(2, 0.3)
        circuit.swap(0, 2)
    pattern_without_swap = circuit.transpile().pattern
    pattern_with_swap = circuit.transpile(transpile_swaps=False).pattern
    state_without_swap = pattern_without_swap.simulate(rng=fx_rng)
    state_with_swap = pattern_with_swap.simulate(rng=fx_rng)
    assert state_without_swap.isclose(state_with_swap)


@pytest.mark.parametrize("transpile_swaps", [True, False])
def test_transpile_pattern_swaps_with_measurements_simple(fx_rng: Generator, transpile_swaps: bool) -> None:
    # See issue
    # https://github.com/TeamGraphix/graphix/issues/584

    circuit = Circuit(2)
    circuit.swap(0, 1)
    circuit.m(1, Axis.Z)

    pattern = circuit.transpile(transpile_swaps=transpile_swaps).pattern
    state_qc = circuit.simulate(rng=fx_rng).state
    state_mbqc = pattern.simulate(rng=fx_rng)
    assert state_mbqc.isclose(state_qc)


@pytest.mark.parametrize("transpile_swaps", [True, False])
def test_transpile_pattern_swaps_with_measurements(fx_rng: Generator, transpile_swaps: bool) -> None:
    circuit = Circuit(4)
    circuit.swap(0, 1)
    circuit.swap(1, 3)
    circuit.cnot(0, 2)
    circuit.rx(2, 0.2)

    circuit.m(1, Axis.Z)
    circuit.m(2, Axis.Z)
    circuit.m(3, Axis.Z)

    pattern = circuit.transpile(transpile_swaps=transpile_swaps).pattern
    state_qc = circuit.simulate(rng=fx_rng).state
    state_mbqc = pattern.simulate(rng=fx_rng)
    assert state_mbqc.isclose(state_qc)


def test_backend_branch_selector() -> None:
    circ = Circuit(1)
    with pytest.raises(ValueError, match="already instantiated"):
        circ.simulate(backend=StatevectorBackend(), branch_selector=ConstBranchSelector(0))


def test_visit() -> None:
    circ = Circuit(1)
    circ.rx(0, 0.5)
    visitor = VisitAngle()
    circ2 = circ.visit(visitor, copy=True)
    assert circ.instruction != circ2.instruction
    assert circ.visit(visitor) is circ
    assert circ.instruction == circ2.instruction

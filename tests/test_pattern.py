from __future__ import annotations

import copy
import itertools
import typing
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.branch_selector import ConstBranchSelector, FixedBranchSelector
from graphix.clifford import Clifford
from graphix.command import C, Command, CommandKind, E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.measurements import Outcome, PauliMeasurement
from graphix.pattern import Pattern, shift_outcomes
from graphix.random_objects import rand_circuit, rand_gate
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from graphix.sim.tensornet import MBQCTensorNet
from graphix.simulator import PatternSimulator
from graphix.states import PlanarState
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.sim.base_backend import BackendState


def compare_backend_result_with_statevec(backend_state: BackendState, statevec: Statevec) -> float:
    if isinstance(backend_state, Statevec):
        return float(np.abs(np.dot(backend_state.flatten().conjugate(), statevec.flatten())))
    if isinstance(backend_state, DensityMatrix):
        return float(np.abs(np.dot(backend_state.rho.flatten().conjugate(), DensityMatrix(statevec).rho.flatten())))
    raise NotImplementedError(backend_state)


class TestPattern:
    def test_manual_generation(self) -> None:
        pattern = Pattern()
        pattern.add(N(node=0))
        pattern.add(N(node=1))
        pattern.add(M(node=0))

    def test_init(self) -> None:
        pattern = Pattern(input_nodes=[1, 0], cmds=[N(node=2), M(node=1)], output_nodes=[2, 0])
        assert pattern.input_nodes == [1, 0]
        assert pattern.output_nodes == [2, 0]
        with pytest.raises(ValueError):
            Pattern(input_nodes=[1, 0], cmds=[N(node=2), M(node=1)], output_nodes=[0, 1, 2])

    def test_eq(self) -> None:
        pattern1 = Pattern(input_nodes=[1, 0], cmds=[N(node=2), M(node=1)], output_nodes=[2, 0])
        pattern2 = Pattern(input_nodes=[1, 0], cmds=[N(node=2), M(node=1)], output_nodes=[2, 0])
        assert pattern1 == pattern2
        pattern1 = Pattern(input_nodes=[1, 0], cmds=[N(node=2), M(node=1)])
        pattern2 = Pattern(input_nodes=[1, 0], cmds=[N(node=2), M(node=1)], output_nodes=[2, 0])
        assert pattern1 != pattern2

    def test_standardize(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern

        pattern.standardize()
        assert pattern.is_standard()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_minimize_space(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_pauli_non_contiguous(self) -> None:
        pattern = Pattern(input_nodes=[0])
        pattern.extend(
            [
                N(node=2, state=PlanarState(plane=Plane.XY, angle=0.0)),
                E(nodes=(0, 2)),
                M(node=0, plane=Plane.XY, angle=0.0, s_domain=set(), t_domain=set()),
            ]
        )
        pattern.perform_pauli_measurements()

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_minimize_space_with_gflow(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rand_gate(nqubits, depth, pairs, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="mc")
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    @pytest.mark.parametrize("backend_type", ["statevector", "densitymatrix", "tensornetwork"])
    def test_empty_output_nodes(
        self, backend_type: typing.Literal["statevector", "densitymatrix", "tensornetwork"]
    ) -> None:
        pattern = Pattern(input_nodes=[0])
        pattern.add(M(node=0, angle=0.5))

        def simulate_and_measure() -> int:
            sim = PatternSimulator(pattern, backend_type)
            sim.run()
            state = sim.backend.state
            if isinstance(state, Statevec):
                assert state.dims() == ()
            elif isinstance(state, DensityMatrix):
                assert state.dims() == (1, 1)
            elif isinstance(state, MBQCTensorNet):
                assert state.to_statevector().shape == (1,)
            return sim.measure_method.get_measure_result(0)

        nb_shots = 1000
        nb_ones = sum(1 for _ in range(nb_shots) if simulate_and_measure())
        assert abs(nb_ones - nb_shots / 2) < nb_shots / 10

    def test_minimize_space_graph_maxspace_with_flow(self, fx_rng: Generator) -> None:
        max_qubits = 20
        for nqubits in range(2, max_qubits):
            depth = 5
            pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
            circuit = rand_gate(nqubits, depth, pairs, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize()
            pattern.minimize_space()
            assert pattern.max_space() == nqubits + 1

    def test_parallelize_pattern(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.parallelize_pattern()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="mc")
        assert pattern.is_standard()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    # TODO: tensor network backend is excluded because "parallel preparation strategy does not support not-standardized pattern".
    def test_pauli_measurement_random_circuit(self, fx_bg: PCG64, jumps: int, backend: str) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="mc")
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(backend, rng=rng)
        assert compare_backend_result_with_statevec(state_mbqc, state) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("ignore_pauli_with_deps", [False, True])
    def test_pauli_measurement_random_circuit_all_paulis(
        self, fx_bg: PCG64, jumps: int, ignore_pauli_with_deps: bool
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="mc")
        pattern.perform_pauli_measurements(ignore_pauli_with_deps=ignore_pauli_with_deps)
        assert ignore_pauli_with_deps or not any(
            PauliMeasurement.try_from(cmd.plane, cmd.angle) for cmd in pattern if cmd.kind == CommandKind.M
        )

    @pytest.mark.parametrize("plane", Plane)
    @pytest.mark.parametrize("angle", [0.0, 0.5, 1.0, 1.5])
    def test_pauli_measurement_single(self, plane: Plane, angle: float) -> None:
        pattern = Pattern(input_nodes=[0, 1])
        pattern.add(E(nodes=(0, 1)))
        pattern.add(M(node=0, plane=plane, angle=angle))
        pattern_ref = pattern.copy()
        pattern.perform_pauli_measurements()
        state = pattern.simulate_pattern()
        state_ref = pattern_ref.simulate_pattern(branch_selector=ConstBranchSelector(0))
        assert np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_leave_input_random_circuit(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="mc")
        pattern.perform_pauli_measurements(leave_input=True)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_pauli_measurement(self) -> None:
        # test pattern is obtained from 3-qubit QFT with pauli measurement
        circuit = Circuit(3)
        for i in range(3):
            circuit.h(i)
        circuit.x(1)
        circuit.x(2)

        # QFT
        circuit.h(2)
        cp(circuit, np.pi / 4, 0, 2)
        cp(circuit, np.pi / 2, 1, 2)
        circuit.h(1)
        cp(circuit, np.pi / 2, 0, 1)
        circuit.h(0)
        swap(circuit, 0, 2)

        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="mc")
        pattern.perform_pauli_measurements()

        isolated_nodes = pattern.get_isolated_nodes()
        # 48-node is the isolated and output node.
        isolated_nodes_ref = {48}

        assert isolated_nodes == isolated_nodes_ref

    def test_pauli_measurement_leave_input(self) -> None:
        # test pattern is obtained from 3-qubit QFT with pauli measurement
        circuit = Circuit(3)
        for i in range(3):
            circuit.h(i)
        circuit.x(1)
        circuit.x(2)

        # QFT
        circuit.h(2)
        cp(circuit, np.pi / 4, 0, 2)
        cp(circuit, np.pi / 2, 1, 2)
        circuit.h(1)
        cp(circuit, np.pi / 2, 0, 1)
        circuit.h(0)
        swap(circuit, 0, 2)

        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="mc")
        pattern.perform_pauli_measurements(leave_input=True)

        isolated_nodes = pattern.get_isolated_nodes()
        # There is no isolated node.
        isolated_nodes_ref: set[int] = set()

        assert isolated_nodes == isolated_nodes_ref

    def test_get_meas_plane(self) -> None:
        preset_meas_plane = [
            Plane.XY,
            Plane.XY,
            Plane.XY,
            Plane.YZ,
            Plane.YZ,
            Plane.YZ,
            Plane.XZ,
            Plane.XZ,
            Plane.XZ,
        ]
        vop_list = [0, 5, 6]  # [identity, S gate, H gate]
        pattern = Pattern(input_nodes=list(range(len(preset_meas_plane))))
        for i in range(len(preset_meas_plane)):
            pattern.add(M(node=i, plane=preset_meas_plane[i]).clifford(Clifford(vop_list[i % 3])))
        ref_meas_plane = {
            0: Plane.XY,
            1: Plane.XY,
            2: Plane.YZ,
            3: Plane.YZ,
            4: Plane.XZ,
            5: Plane.XY,
            6: Plane.XZ,
            7: Plane.YZ,
            8: Plane.XZ,
        }
        meas_plane = pattern.get_meas_plane()
        assert meas_plane == ref_meas_plane

    @pytest.mark.parametrize("plane", Plane)
    @pytest.mark.parametrize("method", ["mc", "direct"])
    def test_shift_signals_plane(self, plane: Plane, method: str) -> None:
        pattern = Pattern(input_nodes=[0])
        for i in (1, 2, 3):
            pattern.add(N(node=i))
            pattern.add(E(nodes=(0, i)))
        pattern.add(M(node=0, angle=0.5))
        pattern.add(M(node=1, angle=0.5))
        pattern.add(M(node=2, angle=0.5, plane=plane, s_domain={0}, t_domain={1}))
        pattern.add(Z(node=3, domain={2}))
        pattern_ref = copy.deepcopy(pattern)
        pattern.standardize()
        signal_dict = pattern.shift_signals(method=method)
        # Test for every possible outcome of each measure
        zero_one: list[Outcome] = [0, 1]
        for outcomes_ref_list in itertools.product(*([zero_one] * 3)):
            outcomes_ref = dict(enumerate(outcomes_ref_list))
            branch_selector = FixedBranchSelector(results=outcomes_ref)
            state_ref = pattern_ref.simulate_pattern(branch_selector=branch_selector)
            outcomes_p = shift_outcomes(outcomes_ref, signal_dict)
            branch_selector = FixedBranchSelector(results=outcomes_p)
            state_p = pattern.simulate_pattern(branch_selector=branch_selector)
            assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize_direct(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals_direct(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="direct")
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_then_standardize(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.perform_pauli_measurements()
        pattern.standardize()
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert compare_backend_result_with_statevec(state_mbqc, state) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize_two_cliffords(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        c0, c1 = rng.integers(len(Clifford), size=2)
        pattern = Pattern(input_nodes=[0])
        pattern.add(C(node=0, clifford=Clifford(c0)))
        pattern.add(C(node=0, clifford=Clifford(c1)))
        pattern_ref = pattern.copy()
        pattern.standardize()
        state_ref = pattern_ref.simulate_pattern()
        state_p = pattern.simulate_pattern()
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 48))
    def test_standardize_domains_and_clifford(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        x, z = rng.integers(2, size=2)
        c = rng.integers(len(Clifford))
        pattern = Pattern(input_nodes=[0])
        pattern.results[1] = x
        pattern.add(X(node=0, domain={1}))
        pattern.results[2] = z
        pattern.add(Z(node=0, domain={2}))
        pattern.add(C(node=0, clifford=Clifford(c)))
        pattern_ref = pattern.copy()
        pattern.standardize()
        state_ref = pattern_ref.simulate_pattern()
        state_p = pattern.simulate_pattern()
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    # Simple pattern composition
    def test_compose_1(self) -> None:
        i1_lst = [0]
        o1_lst = [1]
        cmds1: list[Command] = [N(1), E((0, 1)), M(0), Z(1, {0}), X(1, {0})]
        p1 = Pattern(input_nodes=i1_lst, output_nodes=o1_lst, cmds=cmds1)

        i2_lst = [0]
        o2_lst = [2]
        cmds2: list[Command] = [N(2), E((0, 2)), M(0), Z(2, {0}), X(2, {0})]
        p2 = Pattern(input_nodes=i2_lst, output_nodes=o2_lst, cmds=cmds2)

        mapping = {0: 1, 2: 5}
        pc, mapping_c = p1.compose(p2, mapping)

        i_lst = [0]
        o_lst = [5]
        cmds: list[Command] = [N(1), E((0, 1)), M(0), Z(1, {0}), X(1, {0}), N(5), E((1, 5)), M(1), Z(5, {1}), X(5, {1})]
        p = Pattern(input_nodes=i_lst, output_nodes=o_lst, cmds=cmds)

        assert pc == p
        assert mapping_c == {0: 1, 2: 5}

        with pytest.raises(ValueError, match=r"Keys of `mapping` must correspond to the nodes of `other`."):
            p1.compose(p2, mapping={0: 1, 2: 5, 1: 2})

        with pytest.raises(ValueError, match=r"Values of `mapping` contain duplicates."):
            p1.compose(p2, mapping={0: 1, 2: 1})

        with pytest.raises(ValueError, match=r"Values of `mapping` must not contain measured nodes of pattern `self`."):
            p1.compose(p2, mapping={0: 1, 2: 0})

        with pytest.raises(
            ValueError,
            match=r"Mapping 2 -> 1 is not valid. 1 is an output of pattern `self` but 2 is not an input of pattern `other`.",
        ):
            p1.compose(p2, mapping={2: 1})

    # Pattern composition (more involved than test_compose_1)
    def test_compose_2(self) -> None:
        i1 = [1, 4]
        o1 = [4]
        cmds1: list[Command] = [
            N(0),
            N(2),
            N(3),
            E((1, 2)),
            E((0, 4)),
            M(0),
            M(1),
            M(2),
            M(3, t_domain={1}, s_domain={2}),
            Z(4, {0}),
            X(4, {3, 1}),
        ]
        p1 = Pattern(cmds=cmds1, input_nodes=i1, output_nodes=o1)

        i2 = [0, 3]
        o2 = [3]
        cmds2: list[Command] = [N(1), N(2), M(1), M(2), M(0, t_domain={1}, s_domain={2}), Z(3, {1, 0}), X(3, {2})]
        p2 = Pattern(cmds=cmds2, input_nodes=i2, output_nodes=o2)

        mapping = {0: 4, 3: 100}
        pc, mapping_complete = p1.compose(other=p2, mapping=mapping)

        i = [1, 4, 100]
        o = [100]
        cmds: list[Command] = [
            N(0),
            N(2),
            N(3),
            E((1, 2)),
            E((0, 4)),
            M(0),
            M(1),
            M(2),
            M(3, t_domain={1}, s_domain={2}),
            Z(4, {0}),
            X(4, {3, 1}),
            N(101),
            N(102),
            M(101),
            M(102),
            M(4, t_domain={101}, s_domain={102}),
            Z(100, {4, 101}),
            X(100, {102}),
        ]
        p = Pattern(cmds=cmds, input_nodes=i, output_nodes=o)

        assert p == pc
        assert mapping_complete == {0: 4, 3: 100, 1: 101, 2: 102}

    #  Pattern composition preserving output order
    def test_compose_3(self) -> None:
        i1 = [0, 1, 2, 3]
        o1 = [0, 1, 2, 3]
        cmds1: list[Command] = [
            E((0, 1)),
            E((1, 2)),
            E((2, 3)),
            C(0, Clifford.H),
            C(1, Clifford.X),
        ]
        p1 = Pattern(cmds=cmds1, input_nodes=i1, output_nodes=o1)

        i2 = [0, 1]
        o2 = [2, 3]
        cmds2: list[Command] = [N(2), N(3), E((0, 1)), E((0, 2)), E((1, 3)), M(0), M(1), X(2, {0}), Z(3, {1})]
        p2 = Pattern(cmds=cmds2, input_nodes=i2, output_nodes=o2)

        mapping_1 = {0: 1, 1: 2, 2: 100, 3: 101}
        mapping_2 = {0: 2, 1: 1, 2: 100, 3: 101}

        with pytest.warns(
            UserWarning,
            match=r"Pattern `self` contains Clifford commands and pattern `other` contains E commands. Standardization might not be possible for the resulting composed pattern.",
        ):
            pc, _ = p1.compose(other=p2, mapping=mapping_1, preserve_mapping=False)

        with pytest.warns(
            UserWarning,
            match=r"Pattern `self` contains Clifford commands and pattern `other` contains E commands. Standardization might not be possible for the resulting composed pattern.",
        ):
            pc_1, _ = p1.compose(other=p2, mapping=mapping_1, preserve_mapping=True)

        with pytest.warns(
            UserWarning,
            match=r"Pattern `self` contains Clifford commands and pattern `other` contains E commands. Standardization might not be possible for the resulting composed pattern.",
        ):
            pc_2, _ = p1.compose(other=p2, mapping=mapping_2, preserve_mapping=True)

        i = [0, 1, 2, 3]
        o = [0, 3, 100, 101]
        o_1 = [0, 100, 101, 3]
        o_2 = [0, 101, 100, 3]
        cmds: list[Command] = [
            E((0, 1)),
            E((1, 2)),
            E((2, 3)),
            C(0, Clifford.H),
            C(1, Clifford.X),
            N(100),
            N(101),
            E((1, 2)),
            E((1, 100)),
            E((2, 101)),
            M(1),
            M(2),
            X(100, {1}),
            Z(101, {2}),
        ]
        cmds_2: list[Command] = [
            E((0, 1)),
            E((1, 2)),
            E((2, 3)),
            C(0, Clifford.H),
            C(1, Clifford.X),
            N(100),
            N(101),
            E((2, 1)),
            E((2, 100)),
            E((1, 101)),
            M(2),
            M(1),
            X(100, {2}),
            Z(101, {1}),
        ]
        p = Pattern(cmds=cmds, input_nodes=i, output_nodes=o)
        p_1 = Pattern(cmds=cmds, input_nodes=i, output_nodes=o_1)
        p_2 = Pattern(cmds=cmds_2, input_nodes=i, output_nodes=o_2)

        assert p == pc
        assert p_1 == pc_1
        assert p_2 == pc_2

    #  Pattern composition with Pauli preprocessing
    def test_compose_4(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        i1 = [0]
        o1 = [2]
        cmds1: list[Command] = [N(1), N(2), E((0, 1)), E((1, 2)), M(0, angle=-alpha), M(1), X(2, {1}), Z(2, {0})]
        p1 = Pattern(cmds=cmds1, input_nodes=i1, output_nodes=o1)
        p2 = Pattern(cmds=cmds1, input_nodes=i1, output_nodes=o1)

        p1.perform_pauli_measurements()
        p2.perform_pauli_measurements()

        mapping = {0: 2, 1: 3, 2: 4}
        pc, mapping_complete = p1.compose(p2, mapping=mapping)

        i = [0]
        o = [4]
        cmds: list[Command] = [
            N(2),
            E((0, 2)),
            M(0, plane=Plane.YZ, angle=alpha),
            Z(2, {0}),
            X(2, {1}),
            N(4),
            E((2, 4)),
            M(2, plane=Plane.YZ, angle=alpha),
            Z(4, {2}),
            X(4, {3}),
        ]
        p = Pattern(cmds=cmds, input_nodes=i, output_nodes=o)
        p.results = {1: 0, 3: 0}

        assert p == pc
        assert mapping_complete == mapping

    # Equivalence between pattern and circuit composition
    def test_compose_5(self, fx_rng: Generator) -> None:
        circuit_1 = Circuit(1)
        circuit_1.h(0)
        p1 = circuit_1.transpile().pattern  # outputs: [1]

        alpha = 2 * np.pi * fx_rng.random()

        circuit_2 = Circuit(1)
        circuit_2.rz(0, alpha)
        p2 = circuit_2.transpile().pattern  # inputs: [0]

        p, _ = p1.compose(p2, mapping={0: 1, 1: 2, 2: 3})

        circuit_12 = Circuit(1)
        circuit_12.h(0)
        circuit_12.rz(0, alpha)
        p12 = circuit_12.transpile().pattern

        assert p == p12

    # Compose random circuits
    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_compose_6(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 4
        depth = 2
        circuit_1 = rand_circuit(nqubits, depth, rng, use_ccx=True)
        circuit_2 = rand_circuit(nqubits, depth, rng, use_ccx=True)
        circuit = Circuit(width=nqubits, instr=circuit_1.instruction + circuit_2.instruction)
        p1 = circuit_1.transpile().pattern
        p2 = circuit_2.transpile().pattern
        p = circuit.transpile().pattern
        p_compose, _ = p1.compose(p2, mapping=dict(zip(p2.input_nodes, p1.output_nodes)), preserve_mapping=True)
        p.minimize_space()
        p_compose.minimize_space()
        s = p.simulate_pattern()
        s_compose = p_compose.simulate_pattern()
        assert np.abs(np.dot(s.flatten().conjugate(), s_compose.flatten())) == pytest.approx(1)

    # Test warning composition after standardization
    def test_compose_7(self, fx_rng: Generator) -> None:
        alpha = 2 * np.pi * fx_rng.random()

        circuit_1 = Circuit(1)
        circuit_1.h(0)
        circuit_1.rz(0, alpha)
        p1 = circuit_1.transpile().pattern
        p1.perform_pauli_measurements()

        circuit_2 = Circuit(1)
        circuit_2.rz(0, alpha)
        p2 = circuit_2.transpile().pattern

        with pytest.warns(
            UserWarning,
            match=r"Pattern `self` contains Clifford commands and pattern `other` contains E commands. Standardization might not be possible for the resulting composed pattern.",
        ):
            p1.compose(p2, mapping={0: 3})


def cp(circuit: Circuit, theta: float, control: int, target: int) -> None:
    """Controlled rotation gate, decomposed."""  # noqa: D401
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit: Circuit, a: int, b: int) -> None:
    """Swap gate, decomposed."""
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


class TestMCOps:
    @pytest.mark.parametrize(
        "test",
        [
            ((0, 1), (0, 1), True),
            ((1, 0), (0, 1), True),
            ((0, 1), (2, 0), False),
            ((0, 1), (2, 3), False),
            ((1, 3), (4, 1), False),
        ],
    )
    def test_assert_equal_edge(self, test: tuple[tuple[int, int], tuple[int, int], bool]) -> None:
        assert assert_equal_edge(test[0], test[1]) == test[2]

    def test_no_gate(self) -> None:
        n = 3
        circuit = Circuit(n)
        pattern = circuit.transpile().pattern
        assert len(list(iter(pattern))) == 0

    def test_get_graph(self) -> None:
        n = 3
        g = nx.complete_graph(n)
        circuit = Circuit(n)
        for u, v in g.edges:
            circuit.cnot(u, v)
            circuit.rz(v, np.pi / 4)
            circuit.cnot(u, v)
        for v in g.nodes:
            circuit.rx(v, np.pi / 9)

        pattern = circuit.transpile().pattern
        nodes, edges = pattern.get_graph()

        nodes_ref = list(range(27))
        edges_ref = [
            (1, 3),
            (0, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (0, 7),
            (7, 8),
            (2, 9),
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (0, 13),
            (13, 14),
            (14, 15),
            (8, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (8, 19),
            (19, 20),
            (0, 21),
            (21, 22),
            (8, 23),
            (23, 24),
            (20, 25),
            (25, 26),
        ]

        # nodes check
        nodes_check1 = True
        nodes_check2 = True
        for node in nodes:
            if node not in nodes_ref:
                nodes_check1 = False
        for node in nodes_ref:
            if node not in nodes:
                nodes_check2 = False
        assert nodes_check1
        assert nodes_check2

        # edges check
        edges_check1 = True
        edges_check2 = True
        for edge in edges:
            edge_match = False
            for edge_ref in edges_ref:
                edge_match |= assert_equal_edge(edge, edge_ref)
            if not edge_match:
                edges_check1 = False
        for edge in edges_ref:
            edge_match = False
            for edge_ref in edges:
                edge_match |= assert_equal_edge(edge, edge_ref)
            if not edge_match:
                edges_check2 = False
        assert edges_check1
        assert edges_check2

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern_mc = circuit.transpile().pattern
        pattern_mc.standardize()
        assert pattern.is_standard()
        pattern.minimize_space()
        pattern_mc.minimize_space()
        state_d = pattern.simulate_pattern(rng=rng)
        state_ref = pattern_mc.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_d.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="direct")
        pattern_mc = circuit.transpile().pattern
        pattern_mc.standardize()
        pattern_mc.shift_signals(method="mc")
        assert pattern.is_standard()
        pattern.minimize_space()
        pattern_mc.minimize_space()
        state_d = pattern.simulate_pattern(rng=rng)
        state_ref = pattern_mc.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_d.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize_and_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern(rng=rng)
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 4))
    def test_mixed_pattern_operations(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        processes = [
            [["standardize"]],
            [["standardize"], ["signal", "mc"], ["signal", "direct"]],
            [
                ["standardize"],
                ["signal", "mc"],
                ["standardize"],
                ["signal", "direct"],
            ],
        ]
        nqubits = 3
        depth = 2
        circuit = rand_circuit(nqubits, depth, rng)
        state_ref = circuit.simulate_statevector().statevec
        for process in processes:
            pattern = circuit.transpile().pattern
            for operation in process:
                if operation[0] == "standardize":
                    pattern.standardize()
                elif operation[0] == "signal":
                    pattern.shift_signals(method=operation[1])
            assert pattern.is_standard()
            pattern.minimize_space()
            state_p = pattern.simulate_pattern(rng=rng)
            assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    def test_pauli_measurement_end_with_measure(self) -> None:
        # https://github.com/TeamGraphix/graphix/issues/153
        p = Pattern(input_nodes=[0])
        p.add(N(node=1))
        p.add(M(node=1, plane=Plane.XY))
        p.perform_pauli_measurements()

    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    def test_arbitrary_inputs(self, fx_rng: Generator, nqb: int, rand_circ: Circuit, backend: str) -> None:
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array(Plane), nqb)
        states = [PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        randpattern = rand_circ.transpile().pattern
        out = randpattern.simulate_pattern(backend=backend, input_state=states, rng=fx_rng)
        out_circ = rand_circ.simulate_statevector(input_state=states).statevec
        assert compare_backend_result_with_statevec(out, out_circ) == pytest.approx(1)

    def test_arbitrary_inputs_tn(self, fx_rng: Generator, nqb: int, rand_circ: Circuit) -> None:
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array(Plane), nqb)
        states = [PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        randpattern = rand_circ.transpile().pattern
        with pytest.raises(NotImplementedError):
            randpattern.simulate_pattern(
                backend="tensornetwork", graph_prep="sequential", input_state=states, rng=fx_rng
            )

    def test_remove_qubit(self) -> None:
        p = Pattern(input_nodes=[0, 1])
        p.add(M(node=0))
        p.add(C(node=0, clifford=Clifford.X))
        with pytest.raises(KeyError):
            p.simulate_pattern()


def assert_equal_edge(edge: Sequence[int], ref: Sequence[int]) -> bool:
    return any(all(ei == ri for ei, ri in zip(edge, other)) for other in (ref, reversed(ref)))

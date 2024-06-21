from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

import graphix.ops
import graphix.sim.base_backend
import graphix.states
import tests.random_circuit as rc
from graphix.pattern import CommandNode, Pattern
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from graphix.simulator import PatternSimulator
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

from numpy.random import PCG64, Generator


def compare_backend_result_with_statevec(backend: str, backend_state, statevec: Statevec) -> float:
    if backend == "statevector":
        return np.abs(np.dot(backend_state.flatten().conjugate(), statevec.flatten()))
    elif backend == "densitymatrix":
        return np.abs(np.dot(backend_state.rho.flatten().conjugate(), DensityMatrix(statevec).rho.flatten()))
    else:
        raise NotImplementedError(backend)


class TestPattern:
    # this fails without behaviour modification
    def test_manual_generation(self) -> None:
        pattern = Pattern()
        pattern.add(["N", 0])
        pattern.add(["N", 1])
        pattern.add(["M", 0, "XY", 0, [], []])

    def test_standardize(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        assert pattern.is_standard()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_minimize_space(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 5
        circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_minimize_space_with_gflow(self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix", "tensornetwork"])
    def test_empty_output_nodes(self, backend: Literal["statevector", "densitymatrix", "tensornetwork"]) -> None:
        pattern = Pattern(input_nodes=[0])
        pattern.add(["M", 0, "XY", 0.5, [], []])

        def simulate_and_measure():
            sim = PatternSimulator(pattern, backend)
            sim.run()
            if backend == "statevector":
                assert sim.state.dims() == ()
            elif backend == "densitymatrix":
                assert sim.state.dims() == (1, 1)
            elif backend == "tensornetwork":
                assert sim.state.to_statevector().shape == (1,)
            return sim.results[0]

        nb_shots = 1000
        nb_ones = sum(1 for _ in range(nb_shots) if simulate_and_measure())
        assert abs(nb_ones - nb_shots / 2) < nb_shots / 10

    def test_minimize_space_graph_maxspace_with_flow(self, fx_rng: Generator) -> None:
        max_qubits = 20
        for nqubits in range(2, max_qubits):
            depth = 5
            pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
            circuit = rc.generate_gate(nqubits, depth, pairs, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")
            pattern.minimize_space()
            assert pattern.max_space() == nqubits + 1

    def test_parallelize_pattern(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.parallelize_pattern()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        assert pattern.is_standard()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    # TODO: tensor network backend is excluded because "parallel preparation strategy does not support not-standardized pattern".
    def test_pauli_measurement_random_circuit(
        self, fx_bg: PCG64, jumps: int, backend: graphix.sim.base_backend.Backend, use_rustworkx: bool = True
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(backend)
        assert compare_backend_result_with_statevec(backend, state_mbqc, state) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_leave_input_random_circuit(
        self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx, leave_input=True)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_opt_gate(self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rc.get_rand_circuit(nqubits, depth, rng, use_rzz=True)
        pattern = circuit.transpile(opt=True).pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_opt_gate_transpiler(self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rc.get_rand_circuit(nqubits, depth, rng, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True).pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_opt_gate_transpiler_without_signalshift(
        self,
        fx_bg: PCG64,
        jumps: int,
        use_rustworkx: bool = True,
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rc.get_rand_circuit(nqubits, depth, rng, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True).pattern
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize(
        "use_rustworkx",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(sys.modules.get("rustworkx") is None, reason="rustworkx not installed"),
            ),
        ],
    )
    def test_pauli_measurement(self, use_rustworkx: bool) -> None:
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
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)

        isolated_nodes = pattern.get_isolated_nodes()
        # 48-node is the isolated and output node.
        isolated_nodes_ref = {48}

        assert isolated_nodes == isolated_nodes_ref

    @pytest.mark.parametrize(
        "use_rustworkx",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(sys.modules.get("rustworkx") is None, reason="rustworkx not installed"),
            ),
        ],
    )
    def test_pauli_measurement_leave_input(self, use_rustworkx: bool) -> None:
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
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx, leave_input=True)

        isolated_nodes = pattern.get_isolated_nodes()
        # There is no isolated node.
        isolated_nodes_ref = set()

        assert isolated_nodes == isolated_nodes_ref

    def test_get_meas_plane(self) -> None:
        preset_meas_plane = ["XY", "XY", "XY", "YZ", "YZ", "YZ", "XZ", "XZ", "XZ"]
        vop_list = [0, 5, 6]  # [identity, S gate, H gate]
        pattern = Pattern(input_nodes=list(range(len(preset_meas_plane))))
        for i in range(len(preset_meas_plane)):
            pattern.add(["M", i, preset_meas_plane[i], 0, [], [], vop_list[i % 3]])
        ref_meas_plane = {
            0: "XY",
            1: "XY",
            2: "YZ",
            3: "YZ",
            4: "XZ",
            5: "XY",
            6: "XZ",
            7: "YZ",
            8: "XZ",
        }
        meas_plane = pattern.get_meas_plane()
        assert meas_plane == ref_meas_plane


def cp(circuit: Circuit, theta: float, control: int, target: int) -> None:
    """Controlled rotation gate, decomposed"""
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit: Circuit, a: int, b: int) -> None:
    """swap gate, decomposed"""
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


class TestLocalPattern:
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
        localpattern = pattern.get_local_pattern()
        for node in localpattern.nodes.values():
            assert node.seq == []

    def test_get_graph(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, fx_rng)
        pattern = circuit.transpile().pattern
        nodes_ref, edges_ref = pattern.get_graph()

        localpattern = pattern.get_local_pattern()
        nodes, edges = localpattern.get_graph()

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
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        localpattern = pattern.get_local_pattern()
        localpattern.standardize()
        pattern = localpattern.get_pattern()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        localpattern = pattern.get_local_pattern()
        localpattern.standardize()
        localpattern.shift_signals()
        pattern = localpattern.get_pattern()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize_and_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize_and_shift_signals()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 4))
    def test_mixed_pattern_operations(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        processes = [
            [["standardize", "global"], ["standardize", "local"]],
            [["standardize", "local"], ["signal", "global"], ["signal", "local"]],
            [
                ["standardize", "local"],
                ["signal", "global"],
                ["standardize", "global"],
                ["signal", "local"],
            ],
        ]
        nqubits = 3
        depth = 2
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        state_ref = circuit.simulate_statevector().statevec
        for process in processes:
            pattern = circuit.transpile().pattern
            for operation in process:
                if operation[0] == "standardize":
                    pattern.standardize(method=operation[1])
                elif operation[0] == "signal":
                    pattern.shift_signals(method=operation[1])
            assert pattern.is_standard()
            pattern.minimize_space()
            state_p = pattern.simulate_pattern()
            assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_opt_transpile_standardize(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile(opt=True).pattern
        pattern.standardize(method="local")
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_opt_transpile_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile(opt=True).pattern
        pattern.standardize(method="local")
        pattern.shift_signals(method="local")
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize(
        "test",
        [
            ([1, 2, 3, -1], True),
            ([1, 2, 3, -2, -3, -2, -4], True),
            ([1, -4, 2, -3, -1, 3], False),
            ([1, 2, 3, -1, -4, 2], False),
        ],
    )
    def test_node_is_standardized(self, test: tuple[list[int], bool]) -> None:
        seq, ref = test
        node = CommandNode(0, seq, [], [], False, [], [])
        result = node.is_standard()
        assert result == ref

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_localpattern_is_standard(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rc.get_rand_circuit(nqubits, depth, rng)
        localpattern = circuit.transpile().pattern.get_local_pattern()
        result1 = localpattern.is_standard()
        localpattern.standardize()
        result2 = localpattern.is_standard()
        assert not result1
        assert result2

    def test_pauli_measurement_end_with_measure(self) -> None:
        # https://github.com/TeamGraphix/graphix/issues/153
        p = Pattern(input_nodes=[0])
        p.add(["N", 1])
        p.add(["M", 1, "XY", 0, [], []])
        p.perform_pauli_measurements()

    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    def test_arbitrary_inputs(self, fx_rng: Generator, nqb: int, rand_circ: Circuit, backend: str) -> None:
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        randpattern = rand_circ.transpile().pattern
        out = randpattern.simulate_pattern(backend=backend, input_state=states)
        out_circ = rand_circ.simulate_statevector(input_state=states).statevec
        assert compare_backend_result_with_statevec(backend, out, out_circ) == pytest.approx(1)

    def test_arbitrary_inputs_tn(self, fx_rng: Generator, nqb: int, rand_circ: Circuit) -> None:
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        randpattern = rand_circ.transpile().pattern
        with pytest.raises(NotImplementedError):
            randpattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential", input_state=states)


def assert_equal_edge(edge: Sequence[int], ref: Sequence[int]) -> bool:
    return any(all(ei == ri for ei, ri in zip(edge, other)) for other in (ref, reversed(ref)))

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

import tests.random_circuit as rc
from graphix.pattern import CommandNode, Pattern
from graphix.simulator import PatternSimulator
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator


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
        np.testing.assert_equal(pattern.is_standard(), True)
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_minimize_space(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 5
        circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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
    def test_minimize_space_with_gflow(self, use_rustworkx: bool, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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
        assert abs(nb_ones - nb_shots / 2) < nb_shots / 20

    def test_minimize_space_graph_maxspace_with_flow(self, fx_rng: Generator) -> None:
        max_qubits = 20
        for nqubits in range(2, max_qubits):
            depth = 5
            pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
            circuit = rc.generate_gate(nqubits, depth, pairs, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")
            pattern.minimize_space()
            np.testing.assert_equal(pattern.max_space(), nqubits + 1)

    def test_parallelize_pattern(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.parallelize_pattern()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_shift_signals(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")
            pattern.shift_signals(method="global")
            np.testing.assert_equal(pattern.is_standard(), True)
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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
    def test_pauli_measurment(self, use_rustworkx: bool, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 3
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")
            pattern.shift_signals(method="global")
            pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
            pattern.minimize_space()
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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
    def test_pauli_measurment_leave_input(self, use_rustworkx: bool, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 3
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")
            pattern.shift_signals(method="global")
            pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx, leave_input=True)
            pattern.minimize_space()
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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
    def test_pauli_measurment_opt_gate(self, use_rustworkx: bool, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 3
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng, use_rzz=True)
            pattern = circuit.transpile(opt=True).pattern
            pattern.standardize(method="global")
            pattern.shift_signals(method="global")
            pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
            pattern.minimize_space()
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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
    def test_pauli_measurment_opt_gate_transpiler(self, use_rustworkx: bool, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 3
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng, use_rzz=True)
            pattern = circuit.standardize_and_transpile(opt=True).pattern
            pattern.standardize(method="global")
            pattern.shift_signals(method="global")
            pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
            pattern.minimize_space()
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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
    def test_pauli_measurment_opt_gate_transpiler_without_signalshift(
        self, use_rustworkx: bool, fx_rng: Generator,
    ) -> None:
        nqubits = 3
        depth = 3
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng, use_rzz=True)
            pattern = circuit.standardize_and_transpile(opt=True).pattern
            pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
            pattern.minimize_space()
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

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

        np.testing.assert_equal(isolated_nodes, isolated_nodes_ref)

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

        np.testing.assert_equal(isolated_nodes, isolated_nodes_ref)

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
        np.testing.assert_equal(meas_plane, ref_meas_plane)


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
    def test_assert_equal_edge(self) -> None:
        test_case = [
            [(0, 1), (0, 1), True],
            [(1, 0), (0, 1), True],
            [(0, 1), (2, 0), False],
            [(0, 1), (2, 3), False],
            [(1, 3), (4, 1), False],
        ]
        for test in test_case:
            np.testing.assert_equal(assert_equal_edge(test[0], test[1]), test[2])

    def test_no_gate(self) -> None:
        n = 3
        circuit = Circuit(n)
        pattern = circuit.transpile().pattern
        localpattern = pattern.get_local_pattern()
        for node in localpattern.nodes.values():
            np.testing.assert_equal(node.seq, [])

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
        np.testing.assert_equal(nodes_check1, True)
        np.testing.assert_equal(nodes_check2, True)

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
        np.testing.assert_equal(edges_check1, True)
        np.testing.assert_equal(edges_check2, True)

    def test_standardize(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            localpattern = pattern.get_local_pattern()
            localpattern.standardize()
            pattern = localpattern.get_pattern()
            np.testing.assert_equal(pattern.is_standard(), True)
            pattern.minimize_space()
            state_p = pattern.simulate_pattern()
            state_ref = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)

    def test_shift_signals(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            localpattern = pattern.get_local_pattern()
            localpattern.standardize()
            localpattern.shift_signals()
            pattern = localpattern.get_pattern()
            np.testing.assert_equal(pattern.is_standard(), True)
            pattern.minimize_space()
            state_p = pattern.simulate_pattern()
            state_ref = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)

    def test_standardize_and_shift_signals(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize_and_shift_signals()
            np.testing.assert_equal(pattern.is_standard(), True)
            pattern.minimize_space()
            state_p = pattern.simulate_pattern()
            state_ref = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)

    def test_mixed_pattern_operations(self, fx_rng: Generator) -> None:
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
        for _ in range(3):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            state_ref = circuit.simulate_statevector().statevec
            for process in processes:
                pattern = circuit.transpile().pattern
                for operation in process:
                    if operation[0] == "standardize":
                        pattern.standardize(method=operation[1])
                    elif operation[0] == "signal":
                        pattern.shift_signals(method=operation[1])
                np.testing.assert_equal(pattern.is_standard(), True)
                pattern.minimize_space()
                state_p = pattern.simulate_pattern()
                np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)

    def test_opt_transpile_standardize(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile(opt=True).pattern
            pattern.standardize(method="local")
            np.testing.assert_equal(pattern.is_standard(), True)
            pattern.minimize_space()
            state_p = pattern.simulate_pattern()
            state_ref = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)

    def test_opt_transpile_shift_signals(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            pattern = circuit.transpile(opt=True).pattern
            pattern.standardize(method="local")
            pattern.shift_signals(method="local")
            np.testing.assert_equal(pattern.is_standard(), True)
            pattern.minimize_space()
            state_p = pattern.simulate_pattern()
            state_ref = circuit.simulate_statevector().statevec
            np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)

    def test_node_is_standardized(self) -> None:
        ref_sequence = [
            [[1, 2, 3, -1], True],
            [[1, 2, 3, -2, -3, -2, -4], True],
            [[1, -4, 2, -3, -1, 3], False],
            [[1, 2, 3, -1, -4, 2], False],
        ]
        for [seq, ref] in ref_sequence:
            node = CommandNode(0, seq, [], [], False, [], [])
            result = node.is_standard()
            np.testing.assert_equal(result, ref)

    def test_localpattern_is_standard(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, fx_rng)
            localpattern = circuit.transpile().pattern.get_local_pattern()
            result1 = localpattern.is_standard()
            localpattern.standardize()
            result2 = localpattern.is_standard()
            np.testing.assert_equal(result1, False)
            np.testing.assert_equal(result2, True)


def assert_equal_edge(edge: Sequence[int], ref: Sequence[int]) -> bool:
    ans = True
    for ei, ri in zip(edge, ref):
        ans &= ei == ri
    ansr = True
    for ei, ri in zip(edge, reversed(ref)):
        ansr &= ei == ri
    return ans or ansr

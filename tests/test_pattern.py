import unittest
import numpy as np
import tests.random_circuit as rc
from graphix.transpiler import Circuit
from graphix.pattern import Pattern


class TestPattern(unittest.TestCase):
    def test_standardize(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        for method in {"local", "global"}:
            pattern.standardize(method=method)
            np.testing.assert_equal(pattern.is_standard(), True)
            state = circuit.simulate_statevector()
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_minimize_space(self):
        nqubits = 5
        depth = 5
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize(method="global")
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_minimize_space_with_gflow(self):
        nqubits = 5
        depth = 5
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_minimize_space_graph_maxspace_with_flow(self):
        max_qubits = 20
        for nqubits in range(2, max_qubits):
            depth = 5
            pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
            circuit = rc.generate_gate(nqubits, depth, pairs)
            pattern = circuit.transpile()
            pattern.standardize(method="global")
            pattern.minimize_space()
            np.testing.assert_equal(pattern.max_space(), nqubits + 1)

    def test_parallelize_pattern(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize(method="global")
        pattern.parallelize_pattern()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_shift_signals(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        for method in {"local", "global"}:
            pattern = circuit.transpile()
            pattern.standardize(method=method)
            pattern.shift_signals(method=method)
            np.testing.assert_equal(pattern.is_standard(), True)
            state = circuit.simulate_statevector()
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_pauli_measurment(self):
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_pauli_measurment_opt_gate(self):
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.transpile(opt=True)
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_pauli_measurment_opt_gate_transpiler(self):
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True)
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_pauli_measurment_opt_gate_transpiler_without_signalshift(self):
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True)
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_pauli_measurement(self):
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

        pattern = circuit.transpile()
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements()

        isolated_nodes = pattern.get_isolated_nodes()
        # 48-node is the isolated and output node.
        isolated_nodes_ref = {48}

        np.testing.assert_equal(isolated_nodes, isolated_nodes_ref)

    def test_get_meas_plane(self):
        preset_meas_plane = ["XY", "XY", "XY", "YZ", "YZ", "YZ", "XZ", "XZ", "XZ"]
        vop_list = [0, 5, 6]  # [identity, S gate, H gate]
        pattern = Pattern(len(preset_meas_plane))
        pattern.set_output_nodes([i for i in range(len(preset_meas_plane))])
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


def cp(circuit, theta, control, target):
    """Controlled rotation gate, decomposed"""
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit, a, b):
    """swap gate, decomposed"""
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


class TestLocalPattern(unittest.TestCase):
    def test_assert_equal_edge(self):
        test_case = [
            [(0, 1), (0, 1), True],
            [(1, 0), (0, 1), True],
            [(0, 1), (2, 0), False],
            [(0, 1), (2, 3), False],
            [(1, 3), (4, 1), False],
        ]
        for test in test_case:
            np.testing.assert_equal(assert_equal_edge(test[0], test[1]), test[2])

    def test_no_gate(self):
        n = 3
        circuit = Circuit(n)
        pattern = circuit.transpile()
        localpattern = pattern.get_local_pattern()
        for node in localpattern.nodes.values():
            np.testing.assert_equal(node.seq, [])

    def test_get_graph(self):
        nqubits = 5
        depth = 4
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
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

    def test_standardize(self):
        nqubits = 5
        depth = 4
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        localpattern = pattern.get_local_pattern()
        localpattern.standardize()
        pattern = localpattern.get_pattern()
        np.testing.assert_equal(pattern.is_standard(), True)
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector()
        np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)

    def test_shift_signals(self):
        nqubits = 5
        depth = 4
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        localpattern = pattern.get_local_pattern()
        localpattern.standardize()
        localpattern.shift_signals()
        pattern = localpattern.get_pattern()
        np.testing.assert_equal(pattern.is_standard(), True)
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector()
        np.testing.assert_almost_equal(np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())), 1)


def assert_equal_edge(edge, ref):
    if (edge[0] == ref[0]) and (edge[1] == ref[1]):
        return True
    elif (edge[0] == ref[1]) and (edge[1] == ref[0]):
        return True
    else:
        return False


if __name__ == "__main__":
    unittest.main()

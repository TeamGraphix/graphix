import unittest
import numpy as np
import tests.random_circuit as rc
from graphix.transpiler import Circuit


class TestPattern(unittest.TestCase):
    def test_standardize(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_minimize_space(self):
        nqubits = 5
        depth = 5
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_parallelize_pattern(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.parallelize_pattern()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_shift_signals(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.shift_signals()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_pauli_measurment(self):
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
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
        pattern.standardize()
        pattern.shift_signals()
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
        pattern.standardize()
        pattern.shift_signals()
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
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()

        isolated_nodes = pattern.get_isolated_nodes()
        # 48-node is the isolated and output node.
        isolated_nodes_ref = {48}

        np.testing.assert_equal(isolated_nodes, isolated_nodes_ref)


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


if __name__ == "__main__":
    unittest.main()

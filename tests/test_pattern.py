import unittest
import numpy as np
from graphix.transpiler import Circuit
import tests.random_circuit as rc


class TestPattern_UnitGates(unittest.TestCase):

    def test_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_hadamard(self):
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_s(self):
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_x(self):
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_y(self):
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_z(self):
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_rx(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_ry(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_rz(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_i(self):
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)




class TestPattern(unittest.TestCase):

    def test_standardize(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i+1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_minimize_space(self):
        nqubits = 5
        depth = 5
        pairs = [(i, np.mod(i+1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_parallelize_pattern(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i+1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.parallelize_pattern()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_shift_signals(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i+1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.shift_signals()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_pauli_measurment(self):
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i+1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        pattern.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)


if __name__ == '__main__':
    unittest.main()

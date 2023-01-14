import unittest
import numpy as np
from graphix.transpiler import Circuit
import tests.random_circuit as rc


class TestTranspiler_UnitGates(unittest.TestCase):
    def test_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_hadamard(self):
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_s(self):
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_x(self):
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_y(self):
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_z(self):
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_rx(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_ry(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_rz(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_i(self):
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)


class TestTranspiler_Opt(unittest.TestCase):
    def test_transpile_opt(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.transpile(opt=True)
        state = circuit.simulate_statevector()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_standardize_and_transpile(self):
        nqubits = 3
        depth = 2
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.standardize_and_transpile()
        state = circuit.simulate_statevector()
        pattern.minimize_space()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_standardize_and_transpile_opt(self):
        nqubits = 3
        depth = 2
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True)
        state = circuit.simulate_statevector()
        pattern.minimize_space()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import tests.random_circuit as rc


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


if __name__ == "__main__":
    unittest.main()

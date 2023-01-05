import unittest
import itertools
import numpy as np
from graphix.transpiler import Circuit

import tests.random_circuit as rc


def random_op(sites, dtype=np.complex64, seed=0):
    np.random.seed(seed)
    size = 2**sites
    if dtype is np.complex64:
        return np.random.randn(size, size).astype(np.float32) + 1j * np.random.randn(size, size).astype(np.float32)
    if dtype is np.complex128:
        return np.random.randn(size, size).astype(np.float64) + 1j * np.random.randn(size, size).astype(np.float64)
    return np.random.randn(size, size).astype(dtype)


class TestMPS(unittest.TestCase):
    def test_expectation_value1(self):
        circuit = Circuit(1)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value2(self):
        circuit = Circuit(2)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op2 = random_op(2)
        input = [0, 1]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = mps_mbqc.expectation_value(random_op2, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value3(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op3 = random_op(3)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = mps_mbqc.expectation_value(random_op3, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value3_subspace1(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input, 1):
            value1 = state.expectation_value(random_op1, list(qargs))
            value2 = mps_mbqc.expectation_value(random_op1, list(qargs))
            np.testing.assert_almost_equal(value1, value2)
    
    def test_expectation_value3_subspace2(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op2 = random_op(2)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input, 2):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = mps_mbqc.expectation_value(random_op2, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value_ops3(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1_1 = random_op(1)
        random_op1_2 = random_op(1)
        random_op1_3 = random_op(1)
        random_op3 = np.kron(np.kron(random_op1_1, random_op1_2), random_op1_3)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = mps_mbqc.expectation_value_ops([random_op1_1, random_op1_2, random_op1_3], list(qargs))
            np.testing.assert_almost_equal(value1, value2, decimal=5)  # small accumulation exist, decimal set to 5

    def test_hadamard(self):
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile()
        pattern.standardize()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(
            value1,
            value2,
        )

    def test_s(self):
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_x(self):
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_y(self):
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_z(self):
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_rx(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_ry(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_rz(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_i(self):
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = mps_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        pattern.standardize()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op2 = random_op(2)
        value1 = state.expectation_value(random_op2, [0, 1])
        value2 = mps_mbqc.expectation_value(random_op2, [0, 1])
        np.testing.assert_almost_equal(value1, value2)

    def test_with_graphtrans(self):
        nqubits = 3
        depth = 6
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        state = circuit.simulate_statevector()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        random_op3 = random_op(3)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = mps_mbqc.expectation_value(random_op3, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_get_amplitude(self):
        nqubits = 3
        depth = 5
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        state = circuit.simulate_statevector().flatten()
        mps_mbqc = pattern.simulate_pattern(backend="mps")
        prob_circ = np.zeros(2**nqubits)
        prob_mbqc = np.zeros(2**nqubits)
        for i in range(2**nqubits):
            prob_circ[i] = abs(state[i]) ** 2
            prob_mbqc[i] = mps_mbqc.get_amplitude(i)
        np.testing.assert_almost_equal(prob_circ, prob_mbqc)


if __name__ == "__main__":
    unittest.main()

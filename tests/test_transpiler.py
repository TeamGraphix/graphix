import unittest

import numpy as np

import tests.random_circuit as rc
from graphix.transpiler import Circuit
import graphix.pauli
import graphix.simulator

SEED = 42
rc.set_seed(SEED)


class TestTranspiler_UnitGates(unittest.TestCase):
    def test_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_hadamard(self):
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_s(self):
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_x(self):
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_y(self):
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_z(self):
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_rx(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_ry(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_rz(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_i(self):
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_ccx(self):
        nqubits = 4
        depth = 6
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, use_ccx=True)
            pattern = circuit.transpile().pattern
            pattern.minimize_space()
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)


class TestTranspiler_Opt(unittest.TestCase):
    def test_ccx_opt(self):
        nqubits = 4
        depth = 6
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth, use_ccx=True)
            circuit.ccx(0, 1, 2)
            pattern = circuit.transpile(opt=True).pattern
            pattern.minimize_space()
            state = circuit.simulate_statevector().statevec
            state_mbqc = pattern.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_transpile_opt(self):
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.transpile(opt=True).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_standardize_and_transpile(self):
        nqubits = 3
        depth = 2
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.standardize_and_transpile().pattern
        state = circuit.simulate_statevector().statevec
        pattern.minimize_space()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_standardize_and_transpile_opt(self):
        nqubits = 3
        depth = 2
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True).pattern
        state = circuit.simulate_statevector().statevec
        pattern.minimize_space()
        state_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_measure(self):
        circuit = Circuit(2)
        circuit.h(1)
        circuit.cnot(0, 1)
        circuit.m(0, graphix.pauli.Plane.XY, 0.5)
        transpile = circuit.transpile()

        def simulate_and_measure():
            circuit_simulate = circuit.simulate_statevector()
            assert circuit_simulate.classical_measures[0] == (circuit_simulate.statevec.psi[0][1].imag > 0)
            return circuit_simulate.classical_measures[0]

        nb_shots = 1000
        count = sum(1 for _ in range(nb_shots) if simulate_and_measure())
        assert abs(count - nb_shots / 2) < nb_shots / 20


if __name__ == "__main__":
    unittest.main()

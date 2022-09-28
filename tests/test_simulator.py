import unittest
import numpy as np
from graphix.transpiler import Circuit
from graphix.simulator import Simulator


class TestSimulator(unittest.TestCase):

    def test_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        circuit.sort_outputs()
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_hadamard(self):
        circuit = Circuit(1)
        circuit.h(0)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_s(self):
        circuit = Circuit(1)
        circuit.s(0)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_x(self):
        circuit = Circuit(1)
        circuit.x(0)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_y(self):
        circuit = Circuit(1)
        circuit.y(0)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_z(self):
        circuit = Circuit(1)
        circuit.z(0)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_rx(self):
        circuit = Circuit(1)
        circuit.rx(0, 0.1 * np.pi)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_ry(self):
        circuit = Circuit(1)
        circuit.ry(0, 0.1 * np.pi)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_rz(self):
        circuit = Circuit(1)
        circuit.rz(0, 0.1 * np.pi)
        sim = Simulator(circuit)
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_circuit(self):
        circuit = Circuit(3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.rx(2, 0.2)
        circuit.cnot(0, 2)
        circuit.rx(0, np.pi / 3)
        circuit.sort_outputs()
        sim = Simulator(circuit)
        sim.measure_pauli()
        state = circuit.simulate_statevector()
        state_mbqc = sim.simulate_mbqc()
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)


if __name__ == '__main__':
    unittest.main()

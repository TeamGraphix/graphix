import unittest
import numpy as np
from graphix.transpiler import Circuit
from graphix.ops import Ops
import qiskit.quantum_info as qi


def run_mbqc(circuit, input_state):
    """runs the MBQC pattern.

    Parameters
    ----------
    circuit: graphq.transpiler.Circuit object

    input_state: qi.Statevector

    Returns
    ----------
    state : qi.Statevector
        output staet.
    """
    gstate = input_state.copy()
    n = circuit.Nnode - circuit.width
    gstate = gstate.expand(qi.Statevector(np.ones(2**n) / np.sqrt(2**n)))
    for i, j in circuit.edges:
        gstate = gstate.evolve(Ops.cz, [i, j])

    to_trace = []
    results = {}
    for i in circuit.measurement_order:
        if i in circuit.out:
            pass
        else:
            result = np.random.choice([0, 1])
            results[i] = result

            bpx, bpz = 0, 0
            if i in circuit.byproductx.keys():
                if circuit.byproductx[i]:
                    bpx = np.sum([results[j] for j in circuit.byproductx[i]])
            if i in circuit.byproductz.keys():
                if circuit.byproductz[i]:
                    bpz = np.sum([results[j] for j in circuit.byproductz[i]])

            signal_s, signal_t = 0, 0
            if circuit.domains[i][0]:
                signal_s = np.sum([results[j] for j in circuit.domains[i][0]])
            if circuit.domains[i][1]:
                signal_t = np.sum([results[j] for j in circuit.domains[i][1]])
            angle = circuit.angles[i] * np.pi * (-1)**(signal_s + bpx) \
                + np.pi * (signal_t + bpz)

            if not result:
                meas_op = qi.Statevector(np.array([1, np.exp(1j * angle)]) / np.sqrt(2)).to_operator()
            else:
                meas_op = qi.Statevector(np.array([1, -np.exp(1j * angle)]) / np.sqrt(2)).to_operator()
            gstate = gstate.evolve(meas_op, [i])
            to_trace.append(i)

    gstate = qi.Statevector(  # normalize
        gstate.data / np.sqrt(np.dot(gstate.data.conjugate(), gstate.data)))

    for i in circuit.out:
        if circuit.byproductx[i]:
            if np.mod(np.sum([results[j] for j in circuit.byproductx[i]]), 2):
                gstate = gstate.evolve(Ops.x, [i])
        if circuit.byproductz[i]:
            if np.mod(np.sum([results[j] for j in circuit.byproductz[i]]), 2):
                gstate = gstate.evolve(Ops.z, [i])

    return qi.partial_trace(gstate, to_trace).to_statevector()


class TestTranspiler(unittest.TestCase):

    def test_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        circuit.reorder_qubits()
        input_state = qi.random_statevector(4)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_hadamard(self):
        circuit = Circuit(1)
        circuit.h(0)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_s(self):
        circuit = Circuit(1)
        circuit.s(0)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_x(self):
        circuit = Circuit(1)
        circuit.x(0)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_y(self):
        circuit = Circuit(1)
        circuit.y(0)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_z(self):
        circuit = Circuit(1)
        circuit.z(0)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_rx(self):
        circuit = Circuit(1)
        circuit.rx(0, 0.1 * np.pi)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_ry(self):
        circuit = Circuit(1)
        circuit.ry(0, 0.1 * np.pi)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)

    def test_rz(self):
        circuit = Circuit(1)
        circuit.rz(0, 0.1 * np.pi)
        input_state = qi.random_statevector(2)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
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
        input_state = qi.random_statevector(2**3)
        state = circuit.simulate_statevector(
            input_state=input_state)
        state_mbqc = run_mbqc(circuit, input_state)
        np.testing.assert_almost_equal(
            np.abs(np.dot(state_mbqc.data.conjugate(), state.data)), 1)


if __name__ == '__main__':
    unittest.main()

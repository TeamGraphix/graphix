import numpy as np
import time
import unittest
from graphix.ops import States
from graphix.sim.statevec import Statevec, meas_op
import tests.random_circuit as rc


class StatevecTruncateOneQubitTest(unittest.TestCase):
    def test_truncate_one_qubit(self):
        n = 10
        k = 3

        sv = Statevec(nqubit=n)
        start = time.time()
        for i in range(n):
            sv.entangle([i, (i + 1) % n])
        m_op = meas_op(np.pi / 5)
        sv.evolve(m_op, [k])
        sv.ptrace([k])
        sv.normalize()
        end = time.time()
        print("time (ptrace): ", end - start)

        sv2 = Statevec(nqubit=n)
        start = time.time()
        for i in range(n):
            sv2.entangle([i, (i + 1) % n])
        sv2.evolve(m_op, [k])
        sv2.truncate_one_qubit(k)
        end = time.time()
        print("time (new method): ", end - start)

        np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1)

    def test_measurement_into_each_XYZ_basis(self):
        for state in States.vec:
            m_op = np.outer(state, state.T.conjugate())
            n = 3
            k = 0
            sv = Statevec(nqubit=n)
            sv.evolve(m_op, [k])
            sv.truncate_one_qubit(k)

            sv2 = Statevec(nqubit=n - 1)
            np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1)

    def test_with_rand_circuit_sim(self):
        n = 3
        depth = 3
        circuit = rc.get_rand_circuit(n, depth)
        state = circuit.simulate_statevector()

        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        sv_mbqc = pattern.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(sv_mbqc.psi.flatten().conjugate(), state.flatten())), 1)


if __name__ == "__main__":
    unittest.main()

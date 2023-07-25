import numpy as np
import time
import unittest
from graphix.sim.statevec import Statevec, meas_op


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

        np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1.0)


if __name__ == "__main__":
    unittest.main()

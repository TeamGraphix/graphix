import platform
import unittest
from copy import deepcopy

import numpy as np
from parameterized import parameterized_class

import graphix.sim
from graphix.ops import States
from graphix.sim.backends.backend_factory import _BACKENDS
from graphix.sim.statevec import Statevec, meas_op


@parameterized_class([{"backend": b} for b in _BACKENDS.keys()])
class TestStatevec(unittest.TestCase):
    def setUp(self):
        platform_name = platform.system()  # Calling sys.version_info throws Fatal Python error while using tox
        python_version = (
            platform.python_version_tuple()
        )  # Calling sys.version_info throws Fatal Python error while using tox
        if (
            self.backend == "jax"
            and platform_name == "Windows"
            and python_version[0] == "3"
            and python_version[1] == "8"
        ):
            self.skipTest("`jaxlib` does not support Windows with Python 3.8.")
        graphix.sim.set_backend(self.backend)

    def test_remove_one_qubit(self):
        n = 10
        k = 3

        sv = Statevec(nqubit=n)
        for i in range(n):
            sv.entangle([i, (i + 1) % n])
        m_op = meas_op(np.pi / 5)
        sv.evolve(m_op, [k])
        sv2 = deepcopy(sv)

        sv.remove_qubit(k)
        sv2.ptrace([k])
        sv2.normalize()

        np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1)

    def test_measurement_into_each_XYZ_basis(self):
        n = 3
        k = 0
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        for state in [States.plus, States.zero, States.one, States.iplus, States.iminus]:
            m_op = np.outer(state, state.T.conjugate())
            sv = Statevec(nqubit=n)
            sv.evolve(m_op, [k])
            sv.remove_qubit(k)

            sv2 = Statevec(nqubit=n - 1)
            np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1)

    def test_measurement_into_minus_state(self):
        n = 3
        k = 0
        m_op = np.outer(States.minus, States.minus.T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        with self.assertRaises(AssertionError):
            sv.remove_qubit(k)


if __name__ == "__main__":
    unittest.main()

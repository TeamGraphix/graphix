import unittest
from copy import deepcopy

import numpy as np
from graphix import Circuit
from graphix.states import BasicStates, PlanarState
from graphix.sim.statevec import Statevec, meas_op, StatevectorBackend
import graphix.pauli

class TestStatevec(unittest.TestCase):
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

    #TODO This is a weird test!
    def test_measurement_into_each_XYZ_basis(self):
        n = 3
        k = 0
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        # NOTE weird choice (MINUS is orthogonal to PLUS so zero)
        for state in [BasicStates.PLUS, BasicStates.ZERO, BasicStates.ONE, BasicStates.PLUS_I, BasicStates.MINUS_I]:
            m_op = np.outer(state.get_statevector(), state.get_statevector().T.conjugate())
            # print(m_op)
            sv = Statevec(nqubit=n)
            # print(sv)
            sv.evolve(m_op, [k])
            sv.remove_qubit(k)

            sv2 = Statevec(nqubit=n - 1)
            np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1)

    def test_measurement_into_minus_state(self):
        n = 3
        k = 0
        m_op = np.outer(BasicStates.MINUS.get_statevector(), BasicStates.MINUS.get_statevector().T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        with self.assertRaises(AssertionError):
            sv.remove_qubit(k)

class TestStatevecNew(unittest.TestCase):
    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()  # seed=422

        circ = Circuit(1)
        circ.h(0)
        self.hadamardpattern = circ.transpile()

    # test initialization only
    def test_init_success(self):

        # plus state (default)
        backend = StatevectorBackend(self.hadamardpattern)
        vec = Statevec(nqubit=1)
        np.testing.assert_allclose(vec.psi, backend.state.psi)
        # assert backend.state.Nqubit == 1
        assert len(backend.state.dims()) == 1

        # minus state 
        backend = StatevectorBackend(self.hadamardpattern, input_state = BasicStates.MINUS)
        vec = Statevec(nqubit=1, state = BasicStates.MINUS)
        np.testing.assert_allclose(vec.psi, backend.state.psi)
        # assert backend.state.Nqubit == 1
        assert len(backend.state.dims()) == 1

        # random planar state
        rand_angle = self.rng.random() * 2 * np.pi
        rand_plane = self.rng.choice(np.array([i for i in graphix.pauli.Plane]))
        state = PlanarState(plane = rand_plane, angle = rand_angle)
        backend = StatevectorBackend(self.hadamardpattern, input_state = state)
        vec = Statevec(nqubit=1, state = state)
        np.testing.assert_allclose(vec.psi, backend.state.psi)
        # assert backend.state.Nqubit == 1
        assert len(backend.state.dims()) == 1

        # incorrect number of dimensions
        # only one input node,, two states provided
        # doesn't fail! just takes the first qubit!
        # Discard second qubit so can be whatever

        rand_angle = self.rng.random(2) * 2 * np.pi
        rand_plane = self.rng.choice(np.array([i for i in graphix.pauli.Plane]), 2)

        state = PlanarState(plane = rand_plane[0], angle = rand_angle[0])
        state2 = PlanarState(plane = rand_plane[1], angle = rand_angle[1])
        #with self.assertRaises(ValueError):
        backend = StatevectorBackend(self.hadamardpattern, input_state = [state, state2])
        vec = Statevec(nqubit=1, state = state)
        np.testing.assert_allclose(vec.psi, backend.state.psi)
        # assert backend.state.Nqubit == 1
        assert len(backend.state.dims()) == 1






if __name__ == "__main__":
    unittest.main()

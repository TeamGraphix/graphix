import unittest
import numpy as np

from graphix.states import BasicStates, PlanarState
import graphix.pauli
import graphix.random_objects as randobj
from graphix.sim.statevec import Statevec
import functools


class TestStatevec(unittest.TestCase):
    """Test for Statevec class. Particularly new constructor."""

    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()  # seed=422

    # Errors: types, size,

    # test injitializing one qubit in plus state
    def test_default_success(self):
        vec = Statevec(nqubit=1)
        np.testing.assert_allclose(vec.psi, np.array([1, 1] / np.sqrt(2)))
        assert vec.Nqubit == 1

    def test_basicstates_success(self):
        # minus
        vec = Statevec(nqubit=1, state=BasicStates.MINUS)
        np.testing.assert_allclose(vec.psi, np.array([1, -1] / np.sqrt(2)))
        assert vec.Nqubit == 1
        # zero
        vec = Statevec(nqubit=1, state=BasicStates.ZERO)
        np.testing.assert_allclose(vec.psi, np.array([1, 0]), rtol=0, atol=1e-15)
        assert vec.Nqubit == 1
        # one
        vec = Statevec(nqubit=1, state=BasicStates.ONE)
        np.testing.assert_allclose(vec.psi, np.array([0, 1]), rtol=0, atol=1e-15)
        assert vec.Nqubit == 1
        # plus_i
        vec = Statevec(nqubit=1, state=BasicStates.PLUS_I)
        np.testing.assert_allclose(vec.psi, np.array([1, 1j] / np.sqrt(2)))
        assert vec.Nqubit == 1
        # minus_i
        vec = Statevec(nqubit=1, state=BasicStates.MINUS_I)
        np.testing.assert_allclose(vec.psi, np.array([1, -1j] / np.sqrt(2)))
        assert vec.Nqubit == 1

    # even more tests?
    def test_default_tensor_success(self):
        nqb = self.rng.integers(2, 5)
        # print(f"nqb is {nqb}")
        vec = Statevec(nqubit=nqb)
        np.testing.assert_allclose(vec.psi, np.ones(((2,) * nqb)) / (np.sqrt(2)) ** nqb)
        assert vec.Nqubit == nqb

        vec = Statevec(nqubit=nqb, state=BasicStates.MINUS_I)
        sv_list = [BasicStates.MINUS_I.get_statevector() for _ in range(nqb)]
        sv = functools.reduce(np.kron, sv_list)
        np.testing.assert_allclose(vec.psi, sv.reshape((2,) * nqb))
        assert vec.Nqubit == nqb

        rand_angle = self.rng.random() * 2 * np.pi
        rand_plane = self.rng.choice(np.array([i for i in graphix.pauli.Plane]))
        state = PlanarState(plane=rand_plane, angle=rand_angle)
        vec = Statevec(nqubit=nqb, state=state)
        sv_list = [state.get_statevector() for _ in range(nqb)]
        sv = functools.reduce(np.kron, sv_list)
        np.testing.assert_allclose(vec.psi, sv.reshape((2,) * nqb))
        assert vec.Nqubit == nqb

    def test_data_success(self):
        nqb = self.rng.integers(2, 5)
        l = 2 ** nqb
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        vec = Statevec(state=rand_vec)
        np.testing.assert_allclose(vec.psi, rand_vec.reshape((2,) * nqb))
        assert vec.Nqubit == nqb

    # fail: incorrect len
    def test_data_dim_fail(self):
        l = 5
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with self.assertRaises(ValueError):
            vec = Statevec(state=rand_vec)

    # fail: not normalized
    def test_data_norm_fail(self):
        nqb = self.rng.integers(2, 5)
        l = 2 ** nqb
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        with self.assertRaises(ValueError):
            vec = Statevec(state=rand_vec)

    # fail: no nqubit provided
    def test_default_fail(self):
        with self.assertRaises(ValueError):
            vec = Statevec()

    def test_copy(self):
        nqb = self.rng.integers(2, 5)
        l = 2 ** nqb
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        test_vec = Statevec(state=rand_vec)
        vec = Statevec(state=test_vec)

        np.testing.assert_allclose(vec.psi, test_vec.psi)
        assert vec.Nqubit == test_vec.Nqubit

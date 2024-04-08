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
        # assert vec.Nqubit == 1
        assert len(vec.dims()) == 1

    def test_basicstates_success(self):
        # minus
        vec = Statevec(nqubit=1, data=BasicStates.MINUS)
        np.testing.assert_allclose(vec.psi, np.array([1, -1] / np.sqrt(2)))
        # assert vec.Nqubit == 1
        assert len(vec.dims()) == 1

        # zero
        vec = Statevec(nqubit=1, data=BasicStates.ZERO)
        np.testing.assert_allclose(vec.psi, np.array([1, 0]), rtol=0, atol=1e-15)
        # assert vec.Nqubit == 1
        assert len(vec.dims()) == 1

        # one
        vec = Statevec(nqubit=1, data=BasicStates.ONE)
        np.testing.assert_allclose(vec.psi, np.array([0, 1]), rtol=0, atol=1e-15)
        # assert vec.Nqubit == 1
        assert len(vec.dims()) == 1

        # plus_i
        vec = Statevec(nqubit=1, data=BasicStates.PLUS_I)
        np.testing.assert_allclose(vec.psi, np.array([1, 1j] / np.sqrt(2)))
        # assert vec.Nqubit == 1
        assert len(vec.dims()) == 1

        # minus_i
        vec = Statevec(nqubit=1, data=BasicStates.MINUS_I)
        np.testing.assert_allclose(vec.psi, np.array([1, -1j] / np.sqrt(2)))
        #assert vec.Nqubit == 1
        assert len(vec.dims()) == 1

    # even more tests?
    def test_default_tensor_success(self):
        nqb = self.rng.integers(2, 5)
        vec = Statevec(nqubit=nqb)
        np.testing.assert_allclose(vec.psi, np.ones(((2,) * nqb)) / (np.sqrt(2)) ** nqb)
        # assert vec.Nqubit == nqb
        assert len(vec.dims()) == nqb

        vec = Statevec(nqubit=nqb, data=BasicStates.MINUS_I)
        sv_list = [BasicStates.MINUS_I.get_statevector() for _ in range(nqb)]
        sv = functools.reduce(np.kron, sv_list)
        np.testing.assert_allclose(vec.psi, sv.reshape((2,) * nqb))
        # assert vec.Nqubit == nqb
        assert len(vec.dims()) == nqb

        # tensor of same state
        rand_angle = self.rng.random() * 2 * np.pi
        rand_plane = self.rng.choice(np.array([i for i in graphix.pauli.Plane]))
        state = PlanarState(plane=rand_plane, angle=rand_angle)
        vec = Statevec(nqubit=nqb, data=state)
        sv_list = [state.get_statevector() for _ in range(nqb)]
        sv = functools.reduce(np.kron, sv_list)
        np.testing.assert_allclose(vec.psi, sv.reshape((2,) * nqb))
        # assert vec.Nqubit == nqb
        assert len(vec.dims()) == nqb

        # tensor of different states
        rand_angles = self.rng.random(nqb) * 2 * np.pi
        rand_planes = self.rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        states = [PlanarState(plane = i, angle = j) for i, j in zip(rand_planes, rand_angles)]
        vec = Statevec(nqubit=nqb, data=states)
        sv_list = [state.get_statevector() for state in states]
        sv = functools.reduce(np.kron, sv_list)
        np.testing.assert_allclose(vec.psi, sv.reshape((2,) * nqb))
        # assert vec.Nqubit == nqb
        assert len(vec.dims()) == nqb

    def test_data_success(self):
        nqb = self.rng.integers(2, 5)
        l = 2 ** nqb
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        vec = Statevec(data=rand_vec)
        np.testing.assert_allclose(vec.psi, rand_vec.reshape((2,) * nqb))
        # assert vec.Nqubit == nqb
        assert len(vec.dims()) == nqb
        

    # fail: incorrect len
    def test_data_dim_fail(self):
        l = 5
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with self.assertRaises(ValueError):
            vec = Statevec(data=rand_vec)

    # with less qubit than number of qubits inferred from a correct state vect
    # returns a truncated statevec that is hence not normalized
    # NOTE weird behaviour??
    def test_data_dim_fail_mismatch(self):
        nqb = 3
        rand_vec = self.rng.random(2 ** nqb) + 1j * self.rng.random(2 ** nqb)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with self.assertRaises(ValueError):
            vec = Statevec(nqubit = 2, data=rand_vec)

    # fail: not normalized
    def test_data_norm_fail(self):
        nqb = self.rng.integers(2, 5)
        l = 2 ** nqb
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        with self.assertRaises(ValueError):
            vec = Statevec(data=rand_vec)

    def test_defaults_to_one(self):
        vec = Statevec()
        assert len(vec.dims()) == 1

    # try copying Statevec input
    def test_copy_success(self):
        nqb = self.rng.integers(2, 5)
        l = 2 ** nqb
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        test_vec = Statevec(data=rand_vec)
        # try to copy it
        vec = Statevec(data=test_vec)

        np.testing.assert_allclose(vec.psi, test_vec.psi)
        # assert vec.Nqubit == test_vec.Nqubit
        assert len(vec.dims()) == len(test_vec.dims()) 

    # try calling with incorrect number of qubits compared to inferred one
    def test_copy_fail(self):
        nqb = self.rng.integers(2, 5)
        l = 2 ** nqb
        rand_vec = self.rng.random(l) + 1j * self.rng.random(l)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        test_vec = Statevec(data=rand_vec)

        with self.assertRaises(ValueError):
            vec = Statevec(nqubit=l - 1, data=test_vec)

import functools

import numpy as np
import pytest

import graphix.pauli
from graphix.sim.statevec import Statevec
from graphix.states import BasicStates, PlanarState


class TestStatevec:
    """Test for Statevec class. Particularly new constructor."""

    # test injitializing one qubit in plus state
    def test_default_success(self) -> None:
        vec = Statevec(nqubit=1)
        assert np.allclose(vec.psi, np.array([1, 1] / np.sqrt(2)))
        assert len(vec.dims()) == 1

    def test_basicstates_success(self) -> None:
        # minus
        vec = Statevec(nqubit=1, data=BasicStates.MINUS)
        assert np.allclose(vec.psi, np.array([1, -1] / np.sqrt(2)))
        assert len(vec.dims()) == 1

        # zero
        vec = Statevec(nqubit=1, data=BasicStates.ZERO)
        assert np.allclose(vec.psi, np.array([1, 0]), rtol=0, atol=1e-15)
        assert len(vec.dims()) == 1

        # one
        vec = Statevec(nqubit=1, data=BasicStates.ONE)
        assert np.allclose(vec.psi, np.array([0, 1]), rtol=0, atol=1e-15)
        assert len(vec.dims()) == 1

        # plus_i
        vec = Statevec(nqubit=1, data=BasicStates.PLUS_I)
        assert np.allclose(vec.psi, np.array([1, 1j] / np.sqrt(2)))
        assert len(vec.dims()) == 1

        # minus_i
        vec = Statevec(nqubit=1, data=BasicStates.MINUS_I)
        assert np.allclose(vec.psi, np.array([1, -1j] / np.sqrt(2)))
        assert len(vec.dims()) == 1

    # even more tests?
    def test_default_tensor_success(self, fx_rng: np.random.Generator) -> None:
        nqb = fx_rng.integers(2, 5)
        print(f"nqb is {nqb}")
        vec = Statevec(nqubit=nqb)
        assert np.allclose(vec.psi, np.ones(((2,) * nqb)) / (np.sqrt(2)) ** nqb)
        assert len(vec.dims()) == nqb

        vec = Statevec(nqubit=nqb, data=BasicStates.MINUS_I)
        sv_list = [BasicStates.MINUS_I.get_statevector() for _ in range(nqb)]
        sv = functools.reduce(np.kron, sv_list)
        assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

        # tensor of same state
        rand_angle = fx_rng.random() * 2 * np.pi
        rand_plane = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]))
        state = PlanarState(plane=rand_plane, angle=rand_angle)
        vec = Statevec(nqubit=nqb, data=state)
        sv_list = [state.get_statevector() for _ in range(nqb)]
        sv = functools.reduce(np.kron, sv_list)
        assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

        # tensor of different states
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        states = [PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        vec = Statevec(nqubit=nqb, data=states)
        sv_list = [state.get_statevector() for state in states]
        sv = functools.reduce(np.kron, sv_list)
        assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

    def test_data_success(self, fx_rng: np.random.Generator) -> None:
        nqb = fx_rng.integers(2, 5)
        length = 2**nqb
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        vec = Statevec(data=rand_vec)
        assert np.allclose(vec.psi, rand_vec.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

    # fail: incorrect len
    def test_data_dim_fail(self, fx_rng: np.random.Generator) -> None:
        length = 5
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with pytest.raises(ValueError):
            _vec = Statevec(data=rand_vec)

    # fail: with less qubit than number of qubits inferred from a correct state vect
    def test_data_dim_fail_mismatch(self, fx_rng: np.random.Generator) -> None:
        nqb = 3
        rand_vec = fx_rng.random(2**nqb) + 1j * fx_rng.random(2**nqb)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with pytest.raises(ValueError):
            _vec = Statevec(nqubit=2, data=rand_vec)

    # fail: not normalized
    def test_data_norm_fail(self, fx_rng: np.random.Generator) -> None:
        nqb = fx_rng.integers(2, 5)
        length = 2**nqb
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        with pytest.raises(ValueError):
            _vec = Statevec(data=rand_vec)

    def test_defaults_to_one(self) -> None:
        vec = Statevec()
        assert len(vec.dims()) == 1

    # try copying Statevec input
    def test_copy_success(self, fx_rng: np.random.Generator) -> None:
        nqb = fx_rng.integers(2, 5)
        length = 2**nqb
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        test_vec = Statevec(data=rand_vec)
        # try to copy it
        vec = Statevec(data=test_vec)

        assert np.allclose(vec.psi, test_vec.psi)
        assert len(vec.dims()) == len(test_vec.dims())

    # try calling with incorrect number of qubits compared to inferred one
    def test_copy_fail(self, fx_rng: np.random.Generator) -> None:
        nqb = fx_rng.integers(2, 5)
        length = 2**nqb
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        test_vec = Statevec(data=rand_vec)

        with pytest.raises(ValueError):
            _vec = Statevec(nqubit=length - 1, data=test_vec)

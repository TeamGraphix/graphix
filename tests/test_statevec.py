from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix.fundamentals import ANGLE_PI, Plane
from graphix.sim.statevec import Statevec, _norm_numeric
from graphix.states import BasicStates, PlanarState

if TYPE_CHECKING:
    from numpy.random import Generator


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
    def test_default_tensor_success(self, fx_rng: Generator) -> None:
        nqb = int(fx_rng.integers(2, 5))
        print(f"nqb is {nqb}")
        vec = Statevec(nqubit=nqb)
        assert np.allclose(vec.psi, np.ones((2,) * nqb) / (np.sqrt(2)) ** nqb)
        assert len(vec.dims()) == nqb

        vec = Statevec(nqubit=nqb, data=BasicStates.MINUS_I)
        sv_list = [BasicStates.MINUS_I.to_statevector() for _ in range(nqb)]
        sv = functools.reduce(lambda a, b: np.kron(a, b).astype(np.complex128, copy=False), sv_list)
        assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

        # tensor of same state
        rand_angle = fx_rng.random() * 2 * ANGLE_PI
        rand_plane = fx_rng.choice(np.array(Plane))
        state = PlanarState(rand_plane, rand_angle)
        vec = Statevec(nqubit=nqb, data=state)
        sv_list = [state.to_statevector() for _ in range(nqb)]
        sv = functools.reduce(lambda a, b: np.kron(a, b).astype(np.complex128, copy=False), sv_list)
        assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

        # tensor of different states
        rand_angles = fx_rng.random(nqb) * 2 * ANGLE_PI
        rand_planes = fx_rng.choice(np.array(Plane), nqb)
        states = [PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles, strict=True)]
        vec = Statevec(nqubit=nqb, data=states)
        sv_list = [state.to_statevector() for state in states]
        sv = functools.reduce(lambda a, b: np.kron(a, b).astype(np.complex128, copy=False), sv_list)
        assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

    def test_data_success(self, fx_rng: Generator) -> None:
        nqb = fx_rng.integers(2, 5)
        length = 2**nqb
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        vec = Statevec(data=rand_vec)
        assert np.allclose(vec.psi, rand_vec.reshape((2,) * nqb))
        assert len(vec.dims()) == nqb

    # fail: incorrect len
    def test_data_dim_fail(self, fx_rng: Generator) -> None:
        length = 5
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with pytest.raises(ValueError):
            _vec = Statevec(data=rand_vec)

    # fail: with less qubit than number of qubits inferred from a correct state vect
    def test_data_dim_fail_mismatch(self, fx_rng: Generator) -> None:
        nqb = 3
        rand_vec = fx_rng.random(2**nqb) + 1j * fx_rng.random(2**nqb)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with pytest.raises(ValueError):
            _vec = Statevec(nqubit=2, data=rand_vec)

    # fail: not normalized
    def test_data_norm_fail(self, fx_rng: Generator) -> None:
        nqb = fx_rng.integers(2, 5)
        length = 2**nqb
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        with pytest.raises(ValueError):
            _vec = Statevec(data=rand_vec)

    def test_defaults_to_one(self) -> None:
        vec = Statevec()
        assert len(vec.dims()) == 1

    # try copying Statevec input
    def test_copy_success(self, fx_rng: Generator) -> None:
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
    def test_copy_fail(self, fx_rng: Generator) -> None:
        nqb = int(fx_rng.integers(2, 5))
        length = 1 << nqb
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        test_vec = Statevec(data=rand_vec)

        with pytest.raises(ValueError):
            _vec = Statevec(nqubit=length - 1, data=test_vec)


class TestFidelityIsclose:
    def test_fidelity_same_state(self) -> None:
        state = Statevec(data=BasicStates.PLUS)
        assert state.fidelity(state) == pytest.approx(1)

    def test_fidelity_orthogonal(self) -> None:
        zero = Statevec(data=BasicStates.ZERO)
        one = Statevec(data=BasicStates.ONE)
        assert zero.fidelity(one) == pytest.approx(0)

    def test_fidelity_known_value(self) -> None:
        # F(|0>, |+>) = 0.5
        zero = Statevec(data=BasicStates.ZERO)
        plus = Statevec(data=BasicStates.PLUS)
        assert zero.fidelity(plus) == pytest.approx(0.5)

    def test_fidelity_global_phase(self) -> None:
        plus = Statevec(data=BasicStates.PLUS)
        plus_rotated = Statevec(data=np.array([1, 1]) / np.sqrt(2) * 1j)
        assert plus.fidelity(plus_rotated) == pytest.approx(1)

    def test_fidelity_symmetry(self, fx_rng: Generator) -> None:
        length = 4
        vec_a = fx_rng.random(length) + 1j * fx_rng.random(length)
        vec_a /= np.sqrt(np.sum(np.abs(vec_a) ** 2))
        vec_b = fx_rng.random(length) + 1j * fx_rng.random(length)
        vec_b /= np.sqrt(np.sum(np.abs(vec_b) ** 2))
        a = Statevec(data=vec_a)
        b = Statevec(data=vec_b)
        assert a.fidelity(b) == pytest.approx(b.fidelity(a))

    def test_isclose_same_state(self) -> None:
        state = Statevec(data=BasicStates.PLUS)
        assert state.isclose(state)

    def test_isclose_orthogonal(self) -> None:
        zero = Statevec(data=BasicStates.ZERO)
        one = Statevec(data=BasicStates.ONE)
        assert not zero.isclose(one)

    def test_isclose_global_phase(self) -> None:
        plus = Statevec(data=BasicStates.PLUS)
        rotated = Statevec(data=np.array([1, 1]) / np.sqrt(2) * np.exp(1j * 0.7))
        assert plus.isclose(rotated)

    def test_isclose_tolerance(self) -> None:
        zero = Statevec(data=BasicStates.ZERO)
        almost = Statevec(data=np.array([np.sqrt(1 - 1e-8), np.sqrt(1e-8)]))
        assert not zero.isclose(almost)
        assert zero.isclose(almost, atol=1e-6)


def test_normalize() -> None:
    statevec = Statevec(nqubit=1, data=BasicStates.PLUS)
    statevec.remove_qubit(0)
    assert _norm_numeric(statevec.psi.astype(np.complex128, copy=False)) == 1

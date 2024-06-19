from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

import graphix.pauli
from graphix.sim.statevec import Statevec, StatevectorBackend, meas_op
from graphix.states import BasicStates, PlanarState

if TYPE_CHECKING:
    from numpy.random import Generator


class TestStatevec:
    def test_remove_one_qubit(self) -> None:
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

        assert np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())) == pytest.approx(1)

    @pytest.mark.parametrize(
        "state", [BasicStates.PLUS, BasicStates.ZERO, BasicStates.ONE, BasicStates.PLUS_I, BasicStates.MINUS_I]
    )
    def test_measurement_into_each_XYZ_basis(self, state: BasicStates) -> None:
        n = 3
        k = 0
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        statevector = state.get_statevector()
        m_op = np.outer(statevector, statevector.T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        sv.remove_qubit(k)

        sv2 = Statevec(nqubit=n - 1)
        assert np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())) == pytest.approx(1)

    def test_measurement_into_minus_state(self) -> None:
        n = 3
        k = 0
        m_op = np.outer(BasicStates.MINUS.get_statevector(), BasicStates.MINUS.get_statevector().T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        with pytest.raises(AssertionError):
            sv.remove_qubit(k)


class TestStatevecNew:
    # more tests not really needed since redundant with Statevec constructor tests

    # test initialization only
    def test_init_success(self, hadamardpattern, fx_rng: Generator) -> None:
        # plus state (default)
        backend = StatevectorBackend(hadamardpattern)
        vec = Statevec(nqubit=1)
        assert np.allclose(vec.psi, backend.state.psi)
        assert len(backend.state.dims()) == 1

        # minus state
        backend = StatevectorBackend(hadamardpattern, input_state=BasicStates.MINUS)
        vec = Statevec(nqubit=1, data=BasicStates.MINUS)
        assert np.allclose(vec.psi, backend.state.psi)
        assert len(backend.state.dims()) == 1

        # random planar state
        rand_angle = fx_rng.random() * 2 * np.pi
        rand_plane = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]))
        state = PlanarState(plane=rand_plane, angle=rand_angle)
        backend = StatevectorBackend(hadamardpattern, input_state=state)
        vec = Statevec(nqubit=1, data=state)
        assert np.allclose(vec.psi, backend.state.psi)
        # assert backend.state.Nqubit == 1
        assert len(backend.state.dims()) == 1

        # data input and Statevec input

    def test_init_fail(self, hadamardpattern, fx_rng: Generator) -> None:
        rand_angle = fx_rng.random(2) * 2 * np.pi
        rand_plane = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), 2)

        state = PlanarState(plane=rand_plane[0], angle=rand_angle[0])
        state2 = PlanarState(plane=rand_plane[1], angle=rand_angle[1])
        with pytest.raises(ValueError):
            StatevectorBackend(hadamardpattern, input_state=[state, state2])

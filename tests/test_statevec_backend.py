from __future__ import annotations

from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pytest

from graphix.ops import States
from graphix.sim.statevec import Statevec, meas_op


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

    @pytest.mark.parametrize("state", [States.plus, States.zero, States.one, States.iplus, States.iminus])
    def test_measurement_into_each_xyz_basis(self, state: npt.NDArray) -> None:
        n = 3
        k = 0
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        m_op = np.outer(state, state.T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        sv.remove_qubit(k)

        sv2 = Statevec(nqubit=n - 1)
        assert np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())) == pytest.approx(1)

    def test_measurement_into_minus_state(self) -> None:
        n = 3
        k = 0
        m_op = np.outer(States.minus, States.minus.T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        with pytest.raises(AssertionError):
            sv.remove_qubit(k)

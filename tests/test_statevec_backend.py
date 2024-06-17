from __future__ import annotations

from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pytest

from graphix.ops import States
from graphix.sim.statevec import Statevec, _initial_state, _validate_max_qubit_num, meas_op


class TestStatevectorBackend:
    @pytest.mark.parametrize("max_qubit_num, max_space", [(1.0, 2), (1.1, 2), (0, 2), (1, 2), ("2", 1)])
    def test_validate_max_qubit_num_fail(self, max_qubit_num: float | int, max_space: int):
        with pytest.raises(ValueError):
            _validate_max_qubit_num(max_qubit_num, max_space)

    @pytest.mark.parametrize(
        "max_qubit_num, max_space, expect", [(None, 2, None), (None, 3, None), (1, 1, 1), (10, 1, 10), (10, 2, 10)]
    )
    def test_validate_max_qubit_num_pass(self, max_qubit_num: int | None, max_space: int, expect: int | None):
        assert _validate_max_qubit_num(max_qubit_num, max_space) == expect


class TestStatevec:
    @pytest.mark.parametrize(
        "nqubit, error",
        [
            (1.0, TypeError),
            (-1, ValueError),
        ],
    )
    def test_init_fail(self, nqubit: float | int, error: Exception):
        with pytest.raises(error):
            _initial_state(nqubit=nqubit)

    @pytest.mark.parametrize(
        "nqubit, psi, plus_states, expect",
        [
            (1, States.zero, False, States.zero),
            (1, States.zero, True, States.zero),
            (10, States.zero, False, States.zero),
            (2, None, True, np.tensordot(States.plus, States.plus, 0)),
        ],
    )
    def test_init_pass(self, nqubit: int, psi: npt.NDArray | None, plus_states: bool, expect: npt.NDArray):
        np.testing.assert_allclose(_initial_state(nqubit=nqubit, psi=psi, plus_states=plus_states), expect)

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

from __future__ import annotations

import itertools

import numpy as np
import pytest

from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
)
from graphix.clifford import Clifford


class TestCliffordDB:
    @pytest.mark.parametrize(("i", "j"), itertools.product(range(24), range(3)))
    def test_measure(self, i: int, j: int) -> None:
        pauli = CLIFFORD[j + 1]
        arr = CLIFFORD[i].conjugate().T @ pauli @ CLIFFORD[i]
        sym, sgn = CLIFFORD_MEASURE[i][j]
        arr_ = complex(sgn) * sym.matrix
        assert np.allclose(arr, arr_)

    @pytest.mark.parametrize(("i", "j"), itertools.product(range(24), range(24)))
    def test_multiplication(self, i: int, j: int) -> None:
        op = CLIFFORD[i] @ CLIFFORD[j]
        assert Clifford.try_from_matrix(op) == Clifford(CLIFFORD_MUL[i][j])

    @pytest.mark.parametrize("i", range(24))
    def test_conjugation(self, i: int) -> None:
        op = CLIFFORD[i].conjugate().T
        assert Clifford.try_from_matrix(op) == Clifford(CLIFFORD_CONJ[i])

    @pytest.mark.parametrize("i", range(24))
    def test_decomposition(self, i: int) -> None:
        op = np.eye(2, dtype=np.complex128)
        for j in CLIFFORD_HSZ_DECOMPOSITION[i]:
            op = op @ CLIFFORD[j]
        assert Clifford.try_from_matrix(op) == Clifford(i)

    @pytest.mark.parametrize("i", range(24))
    def test_safety(self, i: int) -> None:
        with pytest.raises(TypeError):
            # Cannot replace
            CLIFFORD[i] = np.eye(2)  # type: ignore[index]
        m = CLIFFORD[i]
        with pytest.raises(ValueError):
            # Cannot modify
            m[0, 0] = 42
        with pytest.raises(ValueError):
            # Cannot make it writeable
            m.flags.writeable = True
        v = m.view()
        with pytest.raises(ValueError):
            # Cannot create writeable view
            v.flags.writeable = True

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
    _CliffordMeasure,
)


class TestCliffordDB:
    @staticmethod
    def classify_pauli(arr: npt.NDArray[np.complex128]) -> _CliffordMeasure:
        """Compare the gate arr with Pauli gates and return the tuple of (Pauli string, sign).

        Parameters
        ----------
            arr: np.array
                2x2 matrix.

        Returns
        -------
            ind : _CliffordMeasure
        """
        if np.allclose(CLIFFORD[1], arr):
            return _CliffordMeasure("X", +1)
        if np.allclose(-1 * CLIFFORD[1], arr):
            return _CliffordMeasure("X", -1)
        if np.allclose(CLIFFORD[2], arr):
            return _CliffordMeasure("Y", +1)
        if np.allclose(-1 * CLIFFORD[2], arr):
            return _CliffordMeasure("Y", -1)
        if np.allclose(CLIFFORD[3], arr):
            return _CliffordMeasure("Z", +1)
        if np.allclose(-1 * CLIFFORD[3], arr):
            return _CliffordMeasure("Z", -1)
        msg = "No Pauli found"
        raise ValueError(msg)

    @staticmethod
    def clifford_index(g: npt.NDArray[np.complex128]) -> int:
        """Return the index of Clifford for a given 2x2 matrix.

        Compare the gate g with all Clifford gates (up to global phase) and return the matching index.

        Parameters
        ----------
            g : 2x2 numpy array.

        Returns
        -------
            i : index of Clifford gate
        """
        for i in range(24):
            ci = CLIFFORD[i]
            # normalise global phase
            norm = g[0, 1] / ci[0, 1] if ci[0, 0] == 0 else g[0, 0] / ci[0, 0]
            # compare
            if np.allclose(ci * norm, g):
                return i
        msg = "No Clifford found"
        raise ValueError(msg)

    @pytest.mark.parametrize(("i", "j"), itertools.product(range(24), range(3)))
    def test_measure(self, i: int, j: int) -> None:
        conj = CLIFFORD[i].conjugate().T
        pauli = CLIFFORD[j + 1]
        arr = np.matmul(np.matmul(conj, pauli), CLIFFORD[i])
        res = self.classify_pauli(arr)
        assert res == CLIFFORD_MEASURE[i][j]

    @pytest.mark.parametrize(("i", "j"), itertools.product(range(24), range(24)))
    def test_multiplication(self, i: int, j: int) -> None:
        arr = np.matmul(CLIFFORD[i], CLIFFORD[j])
        assert CLIFFORD_MUL[i][j] == self.clifford_index(arr)

    @pytest.mark.parametrize("i", range(24))
    def test_conjugation(self, i: int) -> None:
        arr = CLIFFORD[i].conjugate().T
        assert CLIFFORD_CONJ[i] == self.clifford_index(arr)

    @pytest.mark.parametrize("i", range(1, 24))
    def test_decomposition(self, i: int) -> None:
        op = np.eye(2, dtype=np.complex128)
        for j in CLIFFORD_HSZ_DECOMPOSITION[i]:
            op = op @ CLIFFORD[j]
        assert i == self.clifford_index(op)


class TestDB:
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

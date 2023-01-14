import unittest
import numpy as np
from graphix.clifford import CLIFFORD, CLIFFORD_CONJ, CLIFFORD_MEASURE, CLIFFORD_MUL, CLIFFORD_HSZ_DECOMPOSITION


class TestClifford(unittest.TestCase):
    @staticmethod
    def classify_pauli(arr):
        """returns the index of Pauli gate with sign for a given 2x2 matrix.

        Compare the gate arr with Pauli gates
        and return the tuple of (matching index, sign).

        Parameters
        ----------
            arr: np.array
                2x2 matrix.
        Returns
        ----------
            ind : tuple
                tuple containing (pauli index, sign index)
        """

        if np.allclose(CLIFFORD[1], arr):
            return (0, 0)
        elif np.allclose(-1 * CLIFFORD[1], arr):
            return (0, 1)
        elif np.allclose(CLIFFORD[2], arr):
            return (1, 0)
        elif np.allclose(-1 * CLIFFORD[2], arr):
            return (1, 1)
        elif np.allclose(CLIFFORD[3], arr):
            return (2, 0)
        elif np.allclose(-1 * CLIFFORD[3], arr):
            return (2, 1)
        else:
            raise ValueError("No Pauli found")

    @staticmethod
    def clifford_index(g):
        """returns the index of Clifford for a given 2x2 matrix.

        Compare the gate g with all Clifford gates (up to global phase)
        and return the matching index.

        Parameters
        ----------
            g : 2x2 numpy array.

        Returns
        ----------
            i : index of Clifford gate
        """

        for i in range(24):
            # normalise global phase
            if CLIFFORD[i][0, 0] == 0:
                norm = g[0, 1] / CLIFFORD[i][0, 1]
            else:
                norm = g[0, 0] / CLIFFORD[i][0, 0]
            # compare
            if np.allclose(CLIFFORD[i] * norm, g):
                return i
        raise ValueError("No Clifford found")

    def test_measure(self):
        for i in range(24):
            for j in range(3):
                conj = CLIFFORD[i].conjugate().T
                pauli = CLIFFORD[j + 1]
                arr = np.matmul(np.matmul(conj, pauli), CLIFFORD[i])
                res = self.classify_pauli(arr)
                assert res == CLIFFORD_MEASURE[i][j]

    def test_multiplication(self):
        for i in range(24):
            for j in range(24):
                arr = np.matmul(CLIFFORD[i], CLIFFORD[j])
                assert CLIFFORD_MUL[i, j] == self.clifford_index(arr)

    def test_conjugation(self):
        for i in range(24):
            arr = CLIFFORD[i].conjugate().T
            assert CLIFFORD_CONJ[i] == self.clifford_index(arr)

    def test_decomposition(self):
        for i in range(1, 24):
            op = np.eye(2)
            for j in CLIFFORD_HSZ_DECOMPOSITION[i]:
                op = op @ CLIFFORD[j]
            assert i == self.clifford_index(op)


if __name__ == "__main__":
    unittest.main()

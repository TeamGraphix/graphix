import unittest
import numpy as np
import graphix.clifford
import graphix.pauli


class TestPauli(unittest.TestCase):
    def test_unit_mul(self):
        for u in graphix.pauli.UNITS:
            for p in graphix.pauli.LIST:
                assert np.allclose((u * p).matrix, u.complex * p.matrix)

    def test_matmul(self):
        for a in graphix.pauli.LIST:
            for b in graphix.pauli.LIST:
                assert np.allclose((a @ b).matrix, a.matrix @ b.matrix)

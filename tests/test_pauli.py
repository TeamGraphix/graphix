import unittest

import graphix.clifford
import graphix.pauli
import numpy as np


class TestPauli(unittest.TestCase):
    def test_unit_mul(self):
        for u in graphix.pauli.UNITS:
            for p in graphix.pauli.LIST:
                assert np.allclose((u * p).matrix, u.complex * p.matrix)

    def test_matmul(self):
        for a in graphix.pauli.LIST:
            for b in graphix.pauli.LIST:
                assert np.allclose((a @ b).matrix, a.matrix @ b.matrix)

    def test_measure_update(self):
        for plane in graphix.pauli.Plane:
            for s in (False, True):
                for t in (False, True):
                    for clifford in graphix.clifford.TABLE:
                        for angle in (0, np.pi):
                            for choice in (False, True):
                                vop = clifford.index
                                if s:
                                    vop = graphix.clifford.CLIFFORD_MUL[1, vop]
                                if t:
                                    vop = graphix.clifford.CLIFFORD_MUL[3, vop]
                                vec = plane.polar(angle)
                                op_mat_ref = np.eye(2, dtype=np.complex128) / 2
                                for i in range(3):
                                    op_mat_ref += (-1) ** (choice) * vec[i] * graphix.clifford.CLIFFORD[i + 1] / 2
                                clifford_mat = graphix.clifford.CLIFFORD[vop]
                                op_mat_ref = clifford_mat.conj().T @ op_mat_ref @ clifford_mat
                                measure_update = graphix.pauli.MeasureUpdate.compute(plane, s, t, clifford)
                                new_angle = angle * measure_update.coeff + measure_update.add_term
                                vec = measure_update.new_plane.polar(new_angle)
                                op_mat = np.eye(2, dtype=np.complex128) / 2
                                for i in range(3):
                                    op_mat += (-1) ** (choice) * vec[i] * graphix.clifford.CLIFFORD[i + 1] / 2
                                assert np.allclose(op_mat, op_mat_ref) or np.allclose(op_mat, -op_mat_ref)

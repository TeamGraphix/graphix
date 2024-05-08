from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

import graphix.clifford
import graphix.pauli

if TYPE_CHECKING:
    from graphix.clifford import Clifford
    from graphix.pauli import ComplexUnit, Pauli, Plane


class TestPauli:
    @pytest.mark.parametrize(
        ("u", "p"),
        itertools.product(
            graphix.pauli.UNITS,
            graphix.pauli.LIST,
        ),
    )
    def test_unit_mul(self, u: ComplexUnit, p: Pauli) -> None:
        assert np.allclose((u * p).matrix, u.complex * p.matrix)

    @pytest.mark.parametrize(
        ("a", "b"),
        itertools.product(
            graphix.pauli.LIST,
            graphix.pauli.LIST,
        ),
    )
    def test_matmul(self, a: Pauli, b: Pauli) -> None:
        assert np.allclose((a @ b).matrix, a.matrix @ b.matrix)

    @pytest.mark.parametrize(
        ("plane", "s", "t", "clifford", "angle", "choice"),
        itertools.product(
            graphix.pauli.Plane,
            (False, True),
            (False, True),
            graphix.clifford.TABLE,
            (0, np.pi),
            (False, True),
        ),
    )
    def test_measure_update(
        self,
        plane: Plane,
        s: bool,
        t: bool,
        clifford: Clifford,
        angle: float,
        choice: bool,
    ) -> None:
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

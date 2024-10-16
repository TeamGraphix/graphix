from __future__ import annotations

import itertools

import numpy as np
import pytest

from graphix import pauli
from graphix.clifford import Clifford
from graphix.command import MeasureUpdate
from graphix.pauli import UNITS, ComplexUnit, Pauli, Plane


class TestPauli:
    @pytest.mark.parametrize(
        ("u", "p"),
        itertools.product(
            UNITS,
            pauli.LIST,
        ),
    )
    def test_unit_mul(self, u: ComplexUnit, p: Pauli) -> None:
        assert np.allclose((u * p).matrix, complex(u) * p.matrix)

    @pytest.mark.parametrize(
        ("a", "b"),
        itertools.product(
            pauli.LIST,
            pauli.LIST,
        ),
    )
    def test_matmul(self, a: Pauli, b: Pauli) -> None:
        assert np.allclose((a @ b).matrix, a.matrix @ b.matrix)

    @pytest.mark.parametrize(
        ("plane", "s", "t", "clifford", "angle", "choice"),
        itertools.product(
            Plane,
            (False, True),
            (False, True),
            Clifford,
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
        measure_update = MeasureUpdate.compute(plane, s, t, clifford)
        new_angle = angle * measure_update.coeff + measure_update.add_term
        vec = measure_update.new_plane.polar(new_angle)
        op_mat = np.eye(2, dtype=np.complex128) / 2
        for i in range(3):
            op_mat += (-1) ** (choice) * vec[i] * Clifford(i + 1).matrix / 2

        if s:
            clifford = Clifford.X @ clifford
        if t:
            clifford = Clifford.Z @ clifford
        vec = plane.polar(angle)
        op_mat_ref = np.eye(2, dtype=np.complex128) / 2
        for i in range(3):
            op_mat_ref += (-1) ** (choice) * vec[i] * Clifford(i + 1).matrix / 2
        clifford_mat = clifford.matrix
        op_mat_ref = clifford_mat.conj().T @ op_mat_ref @ clifford_mat

        assert np.allclose(op_mat, op_mat_ref) or np.allclose(op_mat, -op_mat_ref)

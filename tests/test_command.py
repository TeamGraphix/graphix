from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from graphix.clifford import Clifford
from graphix.command import MeasureUpdate
from graphix.fundamentals import Plane


@pytest.mark.parametrize(
    ("plane", "s", "t", "clifford", "angle", "choice"),
    itertools.product(
        Plane,
        (False, True),
        (False, True),
        Clifford,
        (0, math.pi),
        (False, True),
    ),
)
def test_measure_update(
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

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix.gf2_linalg import GF2Solver

if TYPE_CHECKING:
    from numpy.random import Generator

SIZES = (1, 2, 5, 10)


@pytest.mark.parametrize(
    ("rows", "cols", "neqs", "p"),
    itertools.product(SIZES, SIZES, SIZES, (0.0, 0.1, 0.5, 0.9, 1.0)),
)
def test_random(fx_rng: Generator, rows: int, cols: int, neqs: int, p: float) -> None:
    lhs = fx_rng.uniform(size=(rows, cols)) < p
    rhs = fx_rng.uniform(size=(rows, neqs)) < p
    sol = GF2Solver(lhs, rhs)
    sol._eliminate()
    sol._check()
    for i in range(neqs):
        x = sol.solve(i)
        if x is None:
            continue
        cmp = (lhs.astype(np.int64) @ x.astype(np.int64)) % 2
        assert all(cmp == rhs[:, i].astype(np.int64))

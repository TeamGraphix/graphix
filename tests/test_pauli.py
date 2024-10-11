from __future__ import annotations

import itertools

import numpy as np
import pytest

from graphix.pauli import ComplexUnit, Pauli


class TestPauli:
    @pytest.mark.parametrize(
        ("u", "p"),
        itertools.product(ComplexUnit, Pauli.iterate()),
    )
    def test_unit_mul(self, u: ComplexUnit, p: Pauli) -> None:
        assert np.allclose((u * p).matrix, complex(u) * p.matrix)

    @pytest.mark.parametrize(
        ("a", "b"),
        itertools.product(Pauli.iterate(), Pauli.iterate()),
    )
    def test_matmul(self, a: Pauli, b: Pauli) -> None:
        assert np.allclose((a @ b).matrix, a.matrix @ b.matrix)

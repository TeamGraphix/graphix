from __future__ import annotations

import itertools

import numpy as np
import pytest

from graphix.clifford import Clifford
from graphix.pauli import IXYZ, ComplexUnit, Pauli, Sign


class TestClifford:
    def test_named(self) -> None:
        assert hasattr(Clifford, "I")
        assert hasattr(Clifford, "X")
        assert hasattr(Clifford, "Y")
        assert hasattr(Clifford, "Z")
        assert hasattr(Clifford, "S")
        assert hasattr(Clifford, "H")

    def test_iteration(self) -> None:
        """Test that Clifford iteration does not take (I, X, Y, Z, S, H) into account."""
        assert len(Clifford) == 24
        assert len(frozenset(Clifford)) == 24

    def test_index_type(self) -> None:
        for c in Clifford:
            assert isinstance(c.index, int)

    @pytest.mark.parametrize("c", Clifford)
    def test_repr(self, c: Clifford) -> None:
        for term in repr(c).split(" @ "):
            assert term in [
                "graphix.clifford.I",
                "graphix.clifford.H",
                "graphix.clifford.S",
                "graphix.clifford.Z",
            ]

    @pytest.mark.parametrize(
        ("c", "p"),
        itertools.product(
            Clifford,
            (
                Pauli(sym, u)
                for sym in IXYZ
                for u in (
                    ComplexUnit(Sign.Plus, False),
                    ComplexUnit(Sign.Minus, False),
                    ComplexUnit(Sign.Plus, True),
                    ComplexUnit(Sign.Minus, True),
                )
            ),
        ),
    )
    def test_measure(self, c: Clifford, p: Pauli) -> None:
        cm = c.matrix
        pm = p.matrix
        cpc = c.measure(p)
        if c == Clifford.I:
            # Prevent aliasing
            assert cpc is not p
        assert np.allclose(cpc.matrix, cm.conj().T @ pm @ cm)

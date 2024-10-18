from __future__ import annotations

import functools
import itertools
import operator
from typing import Final

import numpy as np
import pytest

from graphix.clifford import Clifford
from graphix.fundamentals import IXYZ, ComplexUnit, Sign
from graphix.pauli import Pauli

_QASM3_DB: Final = {
    "id": Clifford.I,
    "x": Clifford.X,
    "y": Clifford.Y,
    "z": Clifford.Z,
    "s": Clifford.S,
    "sdg": Clifford.SDG,
    "h": Clifford.H,
}


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

    @pytest.mark.parametrize("c", Clifford)
    def test_repr(self, c: Clifford) -> None:
        for term in repr(c).split(" @ "):
            assert term in [
                "Clifford.I",
                "Clifford.H",
                "Clifford.S",
                "Clifford.Z",
            ]

    @pytest.mark.parametrize(
        ("c", "p"),
        itertools.product(
            Clifford,
            (
                Pauli(sym, u)
                for sym in IXYZ
                for u in (
                    ComplexUnit.from_properties(sign=Sign.PLUS, is_imag=False),
                    ComplexUnit.from_properties(sign=Sign.MINUS, is_imag=False),
                    ComplexUnit.from_properties(sign=Sign.PLUS, is_imag=True),
                    ComplexUnit.from_properties(sign=Sign.MINUS, is_imag=True),
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

    @pytest.mark.parametrize("c", Clifford)
    def test_qasm3(self, c: Clifford) -> None:
        cmul: Clifford = functools.reduce(operator.matmul, (_QASM3_DB[term] for term in reversed(c.qasm3)))
        assert cmul == c

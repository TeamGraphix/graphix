from __future__ import annotations

import itertools

import numpy as np
import pytest

from graphix.fundamentals import Axis, ComplexUnit, Sign
from graphix.pauli import Pauli


class TestPauli:
    def test_from_axis(self) -> None:
        assert Pauli.from_axis(Axis.X) == Pauli.X
        assert Pauli.from_axis(Axis.Y) == Pauli.Y
        assert Pauli.from_axis(Axis.Z) == Pauli.Z

    def test_axis(self) -> None:
        with pytest.raises(ValueError):
            _ = Pauli.I.axis
        assert Pauli.X.axis == Axis.X
        assert Pauli.Y.axis == Axis.Y
        assert Pauli.Z.axis == Axis.Z

    @pytest.mark.parametrize(
        ("u", "p"),
        itertools.product(ComplexUnit, Pauli),
    )
    def test_unit_mul(self, u: ComplexUnit, p: Pauli) -> None:
        assert np.allclose((u * p).matrix, complex(u) * p.matrix)

    @pytest.mark.parametrize(
        ("a", "b"),
        itertools.product(Pauli, Pauli),
    )
    def test_matmul(self, a: Pauli, b: Pauli) -> None:
        assert np.allclose((a @ b).matrix, a.matrix @ b.matrix)

    @pytest.mark.parametrize("p", Pauli.iterate(symbol_only=True))
    def test_repr(self, p: Pauli) -> None:
        pstr = f"Pauli.{p.symbol.name}"
        assert repr(p) == pstr
        assert repr(1 * p) == pstr
        assert repr(1j * p) == f"1j * {pstr}"
        assert repr(-1 * p) == f"-{pstr}"
        assert repr(-1j * p) == f"-1j * {pstr}"

    @pytest.mark.parametrize("p", Pauli.iterate(symbol_only=True))
    def test_str(self, p: Pauli) -> None:
        pstr = p.symbol.name
        assert str(p) == pstr
        assert str(1 * p) == pstr
        assert str(1j * p) == f"1j * {pstr}"
        assert str(-1 * p) == f"-{pstr}"
        assert str(-1j * p) == f"-1j * {pstr}"

    @pytest.mark.parametrize("p", Pauli)
    def test_neg(self, p: Pauli) -> None:
        pneg = -p
        assert pneg == -p

    def test_iterate_true(self) -> None:
        cmp = list(Pauli.iterate(symbol_only=True))
        assert len(cmp) == 4
        assert cmp[0] == Pauli.I
        assert cmp[1] == Pauli.X
        assert cmp[2] == Pauli.Y
        assert cmp[3] == Pauli.Z

    def test_iterate_false(self) -> None:
        cmp = list(Pauli.iterate(symbol_only=False))
        assert len(cmp) == 16
        assert cmp[0] == Pauli.I
        assert cmp[1] == 1j * Pauli.I
        assert cmp[2] == -1 * Pauli.I
        assert cmp[3] == -1j * Pauli.I
        assert cmp[4] == Pauli.X
        assert cmp[5] == 1j * Pauli.X
        assert cmp[6] == -1 * Pauli.X
        assert cmp[7] == -1j * Pauli.X
        assert cmp[8] == Pauli.Y
        assert cmp[9] == 1j * Pauli.Y
        assert cmp[10] == -1 * Pauli.Y
        assert cmp[11] == -1j * Pauli.Y
        assert cmp[12] == Pauli.Z
        assert cmp[13] == 1j * Pauli.Z
        assert cmp[14] == -1 * Pauli.Z
        assert cmp[15] == -1j * Pauli.Z

    def test_iter_meta(self) -> None:
        it = Pauli.iterate(symbol_only=False)
        it_ = iter(Pauli)
        for p, p_ in zip(it, it_):
            assert p == p_
        assert all(False for _ in it)
        assert all(False for _ in it_)

    @pytest.mark.parametrize(("p", "b"), itertools.product(Pauli.iterate(symbol_only=True), [0, 1]))
    def test_eigenstate(self, p: Pauli, b: int) -> None:
        ev = float(Sign.plus_if(b == 0)) if p != Pauli.I else 1
        evec = p.eigenstate(b).get_statevector()
        assert np.allclose(p.matrix @ evec, ev * evec)

    def test_eigenstate_invalid(self) -> None:
        with pytest.raises(ValueError):
            _ = Pauli.I.eigenstate(2)

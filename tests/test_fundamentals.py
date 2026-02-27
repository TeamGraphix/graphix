from __future__ import annotations

import itertools
import math
from math import pi

import pytest

from graphix.fundamentals import ANGLE_PI, Axis, ComplexUnit, Plane, Sign


class TestSign:
    def test_str(self) -> None:
        assert str(Sign.PLUS) == "+"
        assert str(Sign.MINUS) == "-"

    def test_plus_if(self) -> None:
        assert Sign.plus_if(True) == Sign.PLUS
        assert Sign.plus_if(False) == Sign.MINUS

    def test_minus_if(self) -> None:
        assert Sign.minus_if(True) == Sign.MINUS
        assert Sign.minus_if(False) == Sign.PLUS

    def test_neg(self) -> None:
        assert -Sign.PLUS == Sign.MINUS
        assert -Sign.MINUS == Sign.PLUS

    def test_mul_sign(self) -> None:
        assert Sign.PLUS * Sign.PLUS == Sign.PLUS
        assert Sign.PLUS * Sign.MINUS == Sign.MINUS
        assert Sign.MINUS * Sign.PLUS == Sign.MINUS
        assert Sign.MINUS * Sign.MINUS == Sign.PLUS

    def test_mul_int(self) -> None:
        left = Sign.PLUS * 1
        assert isinstance(left, int)
        assert left == int(Sign.PLUS)
        right = 1 * Sign.PLUS
        assert isinstance(right, int)
        assert right == int(Sign.PLUS)

        left = Sign.MINUS * 1
        assert isinstance(left, int)
        assert left == int(Sign.MINUS)
        right = 1 * Sign.MINUS
        assert isinstance(right, int)
        assert right == int(Sign.MINUS)

    def test_mul_float(self) -> None:
        left = Sign.PLUS * 1.0
        assert isinstance(left, float)
        assert left == float(Sign.PLUS)  # noqa: RUF069
        right = 1.0 * Sign.PLUS
        assert isinstance(right, float)
        assert right == float(Sign.PLUS)  # noqa: RUF069

        left = Sign.MINUS * 1.0
        assert isinstance(left, float)
        assert left == float(Sign.MINUS)  # noqa: RUF069
        right = 1.0 * Sign.MINUS
        assert isinstance(right, float)
        assert right == float(Sign.MINUS)  # noqa: RUF069

    def test_mul_complex(self) -> None:
        left = Sign.PLUS * complex(1)
        assert isinstance(left, complex)
        assert left == complex(Sign.PLUS)  # noqa: RUF069
        right = complex(1) * Sign.PLUS
        assert isinstance(right, complex)
        assert right == complex(Sign.PLUS)  # noqa: RUF069

        left = Sign.MINUS * complex(1)
        assert isinstance(left, complex)
        assert left == complex(Sign.MINUS)  # noqa: RUF069
        right = complex(1) * Sign.MINUS
        assert isinstance(right, complex)
        assert right == complex(Sign.MINUS)  # noqa: RUF069

    def test_int(self) -> None:
        # Necessary to justify `type: ignore`
        assert isinstance(int(Sign.PLUS), int)
        assert isinstance(int(Sign.MINUS), int)


class TestComplexUnit:
    def test_try_from(self) -> None:
        assert ComplexUnit.try_from(ComplexUnit.ONE) == ComplexUnit.ONE
        assert ComplexUnit.try_from(1) == ComplexUnit.ONE
        assert ComplexUnit.try_from(1.0) == ComplexUnit.ONE
        assert ComplexUnit.try_from(1.0 + 0.0j) == ComplexUnit.ONE
        assert ComplexUnit.try_from(3) is None

    def test_from_properties(self) -> None:
        assert ComplexUnit.from_properties() == ComplexUnit.ONE
        assert ComplexUnit.from_properties(is_imag=True) == ComplexUnit.J
        assert ComplexUnit.from_properties(sign=Sign.MINUS) == ComplexUnit.MINUS_ONE
        assert ComplexUnit.from_properties(sign=Sign.MINUS, is_imag=True) == ComplexUnit.MINUS_J

    @pytest.mark.parametrize(("sign", "is_imag"), itertools.product([Sign.PLUS, Sign.MINUS], [True, False]))
    def test_properties(self, sign: Sign, is_imag: bool) -> None:
        assert ComplexUnit.from_properties(sign=sign, is_imag=is_imag).sign == sign
        assert ComplexUnit.from_properties(sign=sign, is_imag=is_imag).is_imag == is_imag

    def test_complex(self) -> None:
        assert complex(ComplexUnit.ONE) == 1  # noqa: RUF069
        assert complex(ComplexUnit.J) == 1j  # noqa: RUF069
        assert complex(ComplexUnit.MINUS_ONE) == -1  # noqa: RUF069
        assert complex(ComplexUnit.MINUS_J) == -1j  # noqa: RUF069

    def test_str(self) -> None:
        assert str(ComplexUnit.ONE) == "1"
        assert str(ComplexUnit.J) == "1j"
        assert str(ComplexUnit.MINUS_ONE) == "-1"
        assert str(ComplexUnit.MINUS_J) == "-1j"

    @pytest.mark.parametrize(("lhs", "rhs"), itertools.product(ComplexUnit, ComplexUnit))
    def test_mul_self(self, lhs: ComplexUnit, rhs: ComplexUnit) -> None:
        assert complex(lhs * rhs) == complex(lhs) * complex(rhs)  # noqa: RUF069

    def test_mul_number(self) -> None:
        assert ComplexUnit.ONE * 1 == ComplexUnit.ONE
        assert 1 * ComplexUnit.ONE == ComplexUnit.ONE
        assert ComplexUnit.ONE * 1.0 == ComplexUnit.ONE  # noqa: RUF069
        assert 1.0 * ComplexUnit.ONE == ComplexUnit.ONE  # noqa: RUF069
        assert ComplexUnit.ONE * complex(1) == ComplexUnit.ONE  # noqa: RUF069
        assert complex(1) * ComplexUnit.ONE == ComplexUnit.ONE  # noqa: RUF069

    def test_neg(self) -> None:
        assert -ComplexUnit.ONE == ComplexUnit.MINUS_ONE
        assert -ComplexUnit.J == ComplexUnit.MINUS_J
        assert -ComplexUnit.MINUS_ONE == ComplexUnit.ONE
        assert -ComplexUnit.MINUS_J == ComplexUnit.J


_PLANE_INDEX = {Axis.X: 0, Axis.Y: 1, Axis.Z: 2}


class TestPlane:
    @pytest.mark.parametrize("p", Plane)
    def test_polar_consistency(self, p: Plane) -> None:
        icos = _PLANE_INDEX[p.cos]
        isin = _PLANE_INDEX[p.sin]
        irest = 3 - icos - isin
        po = p.polar(ANGLE_PI / 4)
        assert po[icos] == pytest.approx(math.cos(pi / 4))
        assert po[isin] == pytest.approx(math.sin(pi / 4))
        assert po[irest] == 0

    def test_from_axes(self) -> None:
        assert Plane.from_axes(Axis.X, Axis.Y) == Plane.XY
        assert Plane.from_axes(Axis.Y, Axis.Z) == Plane.YZ
        assert Plane.from_axes(Axis.X, Axis.Z) == Plane.XZ
        assert Plane.from_axes(Axis.Y, Axis.X) == Plane.XY
        assert Plane.from_axes(Axis.Z, Axis.Y) == Plane.YZ
        assert Plane.from_axes(Axis.Z, Axis.X) == Plane.XZ

    def test_from_axes_ng(self) -> None:
        with pytest.raises(ValueError):
            Plane.from_axes(Axis.X, Axis.X)
        with pytest.raises(ValueError):
            Plane.from_axes(Axis.Y, Axis.Y)
        with pytest.raises(ValueError):
            Plane.from_axes(Axis.Z, Axis.Z)

    def test_isclose(self) -> None:
        for p1, p2 in itertools.combinations(Plane, 2):
            assert not p1.isclose(p2)

        for a1, a2 in itertools.combinations(Axis, 2):
            assert not a1.isclose(a2)

        for p in Plane:
            assert p.isclose(p)
            for a in Axis:
                assert not p.isclose(a)

        for a in Axis:
            assert a.isclose(a)

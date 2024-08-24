"""
Pauli gates ± {1,j} × {I, X, Y, Z}
"""  # noqa: RUF002

from __future__ import annotations

import enum
import sys
import typing
from numbers import Number

import numpy as np
import pydantic

import graphix.clifford


class IXYZ(enum.Enum):
    I = -1
    X = 0
    Y = 1
    Z = 2


class Sign(enum.Enum):
    Plus = 1
    Minus = -1

    def __str__(self) -> str:
        if self == Sign.Plus:
            return "+"
        return "-"

    @staticmethod
    def plus_if(b: bool) -> Sign:
        if b:
            return Sign.Plus
        return Sign.Minus

    @staticmethod
    def minus_if(b: bool) -> Sign:
        if b:
            return Sign.Minus
        return Sign.Plus

    def __neg__(self) -> Sign:
        return Sign.minus_if(self == Sign.Plus)

    def __mul__(self, other: SignOrNumber) -> SignOrNumber:
        if isinstance(other, Sign):
            return Sign.plus_if(self == other)
        if isinstance(other, Number):
            return self.value * other
        return NotImplemented

    def __rmul__(self, other) -> Number:
        if isinstance(other, Number):
            return self.value * other
        return NotImplemented

    def __int__(self) -> int:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __complex__(self) -> complex:
        return complex(self.value)


if sys.version_info >= (3, 10):
    SignOrNumber = typing.TypeVar("SignOrNumber", bound=Sign | Number)
else:
    SignOrNumber = typing.TypeVar("SignOrNumber", bound=typing.Union[Sign, Number])


class ComplexUnit:
    """
    Complex unit: 1, -1, j, -j.

    Complex units can be multiplied with other complex units,
    with Python constants 1, -1, 1j, -1j, and can be negated.
    """

    def __init__(self, sign: Sign, is_imag: bool):
        self.__sign = sign
        self.__is_imag = is_imag

    @property
    def sign(self):
        return self.__sign

    @property
    def is_imag(self) -> bool:
        return self.__is_imag

    def __complex__(self) -> complex:
        """
        Return the unit as complex number
        """
        result: complex = complex(self.__sign)
        if self.__is_imag:
            result *= 1j
        return result

    def __str__(self) -> str:
        if self.__is_imag:
            result = "1j"
        else:
            result = "1"
        if self.__sign == Sign.Minus:
            result = "-" + result
        return result

    def prefix(self, s: str) -> str:
        """
        Prefix the given string by the complex unit as coefficient,
        1 leaving the string unchanged.
        """
        if self.__is_imag:
            result = "1j*" + s
        else:
            result = s
        if self.__sign == Sign.Minus:
            result = "-" + result
        return result

    def __mul__(self, other):
        if isinstance(other, ComplexUnit):
            is_imag = self.__is_imag != other.__is_imag
            sign = self.__sign * other.__sign * Sign.minus_if(self.__is_imag and other.__is_imag)
            return COMPLEX_UNITS[sign == Sign.Minus][is_imag]
        return NotImplemented

    def __rmul__(self, other):
        if other == 1:
            return self
        elif other == -1:
            return COMPLEX_UNITS[self.__sign == Sign.Plus][self.__is_imag]
        elif other == 1j:
            return COMPLEX_UNITS[self.__sign == Sign.plus_if(self.__is_imag)][not self.__is_imag]
        elif other == -1j:
            return COMPLEX_UNITS[self.__sign == Sign.minus_if(self.__is_imag)][not self.__is_imag]

    def __neg__(self):
        return COMPLEX_UNITS[self.__sign == Sign.Plus][self.__is_imag]


COMPLEX_UNITS = tuple(
    tuple(ComplexUnit(sign, is_imag) for is_imag in (False, True)) for sign in (Sign.Plus, Sign.Minus)
)


UNIT = COMPLEX_UNITS[False][False]


UNITS = (UNIT, -UNIT, 1j * UNIT, -1j * UNIT)


class Axis(enum.Enum):
    X = 0
    Y = 1
    Z = 2


class Plane(enum.Enum):
    XY = 0
    YZ = 1
    XZ = 2

    @property
    def axes(self) -> list[Axis]:
        # match self:
        #     case Plane.XY:
        #         return [Axis.X, Axis.Y]
        #     case Plane.YZ:
        #         return [Axis.Y, Axis.Z]
        #     case Plane.XZ:
        #         return [Axis.X, Axis.Z]
        if self == Plane.XY:
            return [Axis.X, Axis.Y]
        elif self == Plane.YZ:
            return [Axis.Y, Axis.Z]
        elif self == Plane.XZ:
            return [Axis.X, Axis.Z]

    @property
    def cos(self) -> Axis:
        # match self:
        #     case Plane.XY:
        #         return Axis.X
        #     case Plane.YZ:
        #         return Axis.Z  # former convention was Y
        #     case Plane.XZ:
        #         return Axis.Z  # former convention was X
        if self == Plane.XY:
            return Axis.X
        elif self == Plane.YZ:
            return Axis.Z  # former convention was Y
        elif self == Plane.XZ:
            return Axis.Z  # former convention was X

    @property
    def sin(self) -> Axis:
        # match self:
        #     case Plane.XY:
        #         return Axis.Y
        #     case Plane.YZ:
        #         return Axis.Y  # former convention was Z
        #     case Plane.XZ:
        #         return Axis.X  # former convention was Z
        if self == Plane.XY:
            return Axis.Y
        elif self == Plane.YZ:
            return Axis.Y  # former convention was Z
        elif self == Plane.XZ:
            return Axis.X  # former convention was Z

    def polar(self, angle: float) -> tuple[float, float, float]:
        result = [0, 0, 0]
        result[self.cos.value] = np.cos(angle)
        result[self.sin.value] = np.sin(angle)
        return tuple(result)

    @staticmethod
    def from_axes(a: Axis, b: Axis) -> Plane:
        if b.value < a.value:
            a, b = b, a
        # match a, b:
        #     case Axis.X, Axis.Y:
        #         return Plane.XY
        #     case Axis.Y, Axis.Z:
        #         return Plane.YZ
        #     case Axis.X, Axis.Z:
        #         return Plane.XZ
        if a == Axis.X and b == Axis.Y:
            return Plane.XY
        elif a == Axis.Y and b == Axis.Z:
            return Plane.YZ
        elif a == Axis.X and b == Axis.Z:
            return Plane.XZ
        assert a == b
        raise ValueError(f"Cannot make a plane giving the same axis {a} twice.")


class Pauli:
    """
    Pauli gate: u * {I, X, Y, Z} where u is a complex unit

    Pauli gates can be multiplied with other Pauli gates (with @),
    with complex units and unit constants (with *),
    and can be negated.
    """

    def __init__(self, symbol: IXYZ, unit: ComplexUnit):
        self.__symbol = symbol
        self.__unit = unit

    @staticmethod
    def from_axis(axis: Axis) -> Pauli:
        return Pauli(IXYZ[axis.name], UNIT)

    @property
    def axis(self) -> Axis:
        if self.__symbol == IXYZ.I:
            raise ValueError("I is not an axis.")
        return Axis[self.__symbol.name]

    @property
    def symbol(self):
        return self.__symbol

    @property
    def unit(self):
        return self.__unit

    @property
    def matrix(self) -> np.ndarray:
        """
        Return the matrix of the Pauli gate.
        """
        return complex(self.__unit) * graphix.clifford.CLIFFORD[self.__symbol.value + 1]

    def __repr__(self):
        return self.__unit.prefix(self.__symbol.name)

    def __matmul__(self, other):
        if isinstance(other, Pauli):
            if self.__symbol == IXYZ.I:
                symbol = other.__symbol
                unit = 1
            elif other.__symbol == IXYZ.I:
                symbol = self.__symbol
                unit = 1
            elif self.__symbol == other.__symbol:
                symbol = IXYZ.I
                unit = 1
            elif (self.__symbol.value + 1) % 3 == other.__symbol.value:
                symbol = IXYZ((self.__symbol.value + 2) % 3)
                unit = 1j
            else:
                symbol = IXYZ((self.__symbol.value + 1) % 3)
                unit = -1j
            return get(symbol, unit * self.__unit * other.__unit)
        return NotImplemented

    def __rmul__(self, other):
        return get(self.__symbol, other * self.__unit)

    def __neg__(self):
        return get(self.__symbol, -self.__unit)


TABLE = tuple(
    tuple(tuple(Pauli(symbol, COMPLEX_UNITS[sign][is_imag]) for is_imag in (False, True)) for sign in (False, True))
    for symbol in (IXYZ.I, IXYZ.X, IXYZ.Y, IXYZ.Z)
)


LIST = tuple(pauli for sign_im_list in TABLE for im_list in sign_im_list for pauli in im_list)


def get(symbol: IXYZ, unit: ComplexUnit) -> Pauli:
    """Return the Pauli gate with given symbol and unit."""
    return TABLE[symbol.value + 1][unit.sign == Sign.Minus][unit.is_imag]


I = get(IXYZ.I, UNIT)
X = get(IXYZ.X, UNIT)
Y = get(IXYZ.Y, UNIT)
Z = get(IXYZ.Z, UNIT)


def parse(name: str) -> Pauli:
    """
    Return the Pauli gate with the given name (limited to "I", "X", "Y" and "Z").
    """
    return get(IXYZ[name], UNIT)


def is_int(value: Number) -> bool:
    return value == int(value)


class PauliMeasurement(typing.NamedTuple):
    """
    Pauli measurement.
    """

    axis: Axis
    sign: Sign

    @staticmethod
    def try_from(plane: Plane, angle: float) -> PauliMeasurement | None:
        angle_double = 2 * angle
        if not is_int(angle_double):
            return None
        angle_double_mod_4 = int(angle_double) % 4
        if angle_double_mod_4 % 2 == 0:
            axis = plane.cos
        else:
            axis = plane.sin
        sign = Sign.minus_if(angle_double_mod_4 >= 2)
        return PauliMeasurement(axis, sign)


class MeasureUpdate(pydantic.BaseModel):
    new_plane: Plane
    coeff: int
    add_term: float

    @staticmethod
    def compute(plane: Plane, s: bool, t: bool, clifford: graphix.clifford.Clifford) -> MeasureUpdate:
        gates = list(map(Pauli.from_axis, plane.axes))
        if s:
            clifford = graphix.clifford.X @ clifford
        if t:
            clifford = graphix.clifford.Z @ clifford
        gates = list(map(clifford.measure, gates))
        new_plane = Plane.from_axes(*(gate.axis for gate in gates))
        cos_pauli = clifford.measure(Pauli.from_axis(plane.cos))
        sin_pauli = clifford.measure(Pauli.from_axis(plane.sin))
        exchange = cos_pauli.axis != new_plane.cos
        if exchange == (cos_pauli.unit.sign == sin_pauli.unit.sign):
            coeff = -1
        else:
            coeff = 1
        add_term = 0
        if cos_pauli.unit.sign == Sign.Minus:
            add_term += np.pi
        if exchange:
            add_term = np.pi / 2 - add_term
        return MeasureUpdate(new_plane=new_plane, coeff=coeff, add_term=add_term)

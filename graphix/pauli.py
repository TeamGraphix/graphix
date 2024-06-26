"""
Pauli gates ± {1,j} × {I, X, Y, Z}
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pydantic

import graphix.clifford
import graphix.ops

if TYPE_CHECKING:
    from graphix.states import BasicStates


class IXYZ(enum.Enum):
    I = -1
    X = 0
    Y = 1
    Z = 2


class ComplexUnit:
    """
    Complex unit: 1, -1, j, -j.

    Complex units can be multiplied with other complex units,
    with Python constants 1, -1, 1j, -1j, and can be negated.
    """

    def __init__(self, sign: bool, im: bool):
        self.__sign = sign
        self.__im = im

    @property
    def sign(self):
        return self.__sign

    @property
    def im(self):
        return self.__im

    @property
    def complex(self) -> complex:
        """
        Return the unit as complex number
        """
        result: complex = 1
        if self.__sign:
            result *= -1
        if self.__im:
            result *= 1j
        return result

    def __repr__(self):
        if self.__im:
            result = "1j"
        else:
            result = "1"
        if self.__sign:
            result = "-" + result
        return result

    def prefix(self, s: str) -> str:
        """
        Prefix the given string by the complex unit as coefficient,
        1 leaving the string unchanged.
        """
        if self.__im:
            result = "1j*" + s
        else:
            result = s
        if self.__sign:
            result = "-" + result
        return result

    def __mul__(self, other):
        if isinstance(other, ComplexUnit):
            im = self.__im != other.__im
            sign = (self.__sign != other.__sign) != (self.__im and other.__im)
            return COMPLEX_UNITS[sign][im]
        return NotImplemented

    def __rmul__(self, other):
        if other == 1:
            return self
        elif other == -1:
            return COMPLEX_UNITS[not self.__sign][self.__im]
        elif other == 1j:
            return COMPLEX_UNITS[self.__sign != self.__im][not self.__im]
        elif other == -1j:
            return COMPLEX_UNITS[self.__sign == self.__im][not self.__im]

    def __neg__(self):
        return COMPLEX_UNITS[not self.__sign][self.__im]


COMPLEX_UNITS = [[ComplexUnit(sign, im) for im in (False, True)] for sign in (False, True)]


UNIT = COMPLEX_UNITS[False][False]


UNITS = [UNIT, -UNIT, 1j * UNIT, -1j * UNIT]


class Axis(enum.Enum):
    X = 0
    Y = 1
    Z = 2

    @property
    def op(self) -> npt.NDArray:
        if self == Axis.X:
            return graphix.ops.Ops.x
        elif self == Axis.Y:
            return graphix.ops.Ops.y
        elif self == Axis.Z:
            return graphix.ops.Ops.z


class Plane(enum.Enum):
    XY = 0
    YZ = 1
    XZ = 2

    @property
    def axes(self) -> list[Axis]:
        """Return the pair of axes that carry the plane."""
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
    def orth(self) -> Axis:
        """Return the axis orthogonal to the plane."""
        if self == Plane.XY:
            return Axis.Z
        elif self == Plane.YZ:
            return Axis.X
        elif self == Plane.XZ:
            return Axis.Y

    @property
    def cos(self) -> Axis:
        """Return the axis of the plane that conventionally carries the cos."""
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
        """Return the axis of the plane that conventionally carries the sin."""
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
        """Return the Cartesian coordinates of the point of module 1 at the
        given angle, following the conventional orientation for cos and sin."""
        result = [0, 0, 0]
        result[self.cos.value] = np.cos(angle)
        result[self.sin.value] = np.sin(angle)
        return tuple(result)

    @staticmethod
    def from_axes(a: Axis, b: Axis) -> Plane:
        """Return the plane carried by the given axes."""
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
    def symbol(self) -> IXYZ:
        return self.__symbol

    @property
    def unit(self) -> ComplexUnit:
        return self.__unit

    @property
    def matrix(self) -> np.ndarray:
        """
        Return the matrix of the Pauli gate.
        """
        return self.__unit.complex * graphix.clifford.CLIFFORD[self.__symbol.value + 1]

    def get_eigenstate(self, eigenvalue=0) -> BasicStates:
        from graphix.states import BasicStates

        if self.symbol == IXYZ.X:
            return BasicStates.PLUS if eigenvalue == 0 else BasicStates.MINUS
        elif self.symbol == IXYZ.Y:
            return BasicStates.PLUS_I if eigenvalue == 0 else BasicStates.MINUS_I
        elif self.symbol == IXYZ.Z:
            return BasicStates.ZERO if eigenvalue == 0 else BasicStates.ONE
        # Any state is eigenstate of the identity
        elif self.symbol == IXYZ.I:
            return BasicStates.PLUS

    def __repr__(self) -> str:
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

    def __rmul__(self, other) -> Pauli:
        if isinstance(other, ComplexUnit):
            return get(self.__symbol, other * self.__unit)
        return NotImplemented

    def __neg__(self) -> Pauli:
        return get(self.__symbol, -self.__unit)


TABLE = [
    [[Pauli(symbol, COMPLEX_UNITS[sign][im]) for im in (False, True)] for sign in (False, True)]
    for symbol in (IXYZ.I, IXYZ.X, IXYZ.Y, IXYZ.Z)
]


LIST = [pauli for sign_im_list in TABLE for im_list in sign_im_list for pauli in im_list]


def get(symbol: IXYZ, unit: ComplexUnit) -> Pauli:
    """Return the Pauli gate with given symbol and unit."""
    return TABLE[symbol.value + 1][unit.sign][unit.im]


I = get(IXYZ.I, UNIT)
X = get(IXYZ.X, UNIT)
Y = get(IXYZ.Y, UNIT)
Z = get(IXYZ.Z, UNIT)


def parse(name: str) -> Pauli:
    """
    Return the Pauli gate with the given name (limited to "I", "X", "Y" and "Z").
    """
    return get(IXYZ[name], UNIT)


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
        if cos_pauli.unit.sign:
            add_term += np.pi
        if exchange:
            add_term = np.pi / 2 - add_term
        return MeasureUpdate(new_plane=new_plane, coeff=coeff, add_term=add_term)

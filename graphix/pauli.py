"""
Pauli gates ± {1,j} × {I, X, Y, Z}
"""

import enum
import typing

import numpy as np
import pydantic

import graphix.clifford


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


class Plane(enum.Enum):
    XY = 0
    YZ = 1
    XZ = 2

    @property
    def axes(self) -> typing.List[Axis]:
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

    def polar(self, angle: float) -> typing.Tuple[float, float, float]:
        result = [0, 0, 0]
        result[self.cos.value] = np.cos(angle)
        result[self.sin.value] = np.sin(angle)
        return tuple(result)

    @staticmethod
    def from_axes(a: Axis, b: Axis) -> "Plane":
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
    def from_axis(axis: Axis) -> "Pauli":
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
        return self.__unit.complex * graphix.clifford.CLIFFORD[self.__symbol.value + 1]

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
    def compute(plane: Plane, s: bool, t: bool, clifford: "graphix.clifford.Clifford") -> "MeasureUpdate":
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

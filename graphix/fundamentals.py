"""Fundamental components related to quantum mechanics."""

from __future__ import annotations

import enum
import sys
import typing
from enum import Enum
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, SupportsIndex, overload

import typing_extensions

from graphix.ops import Ops
from graphix.parameter import cos_sin

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from graphix.parameter import Expression, ExpressionOrFloat


if sys.version_info >= (3, 10):
    SupportsComplexCtor = SupportsComplex | SupportsFloat | SupportsIndex | complex
else:  # pragma: no cover
    from typing import Union

    SupportsComplexCtor = Union[SupportsComplex, SupportsFloat, SupportsIndex, complex]


class Sign(Enum):
    """Sign, plus or minus."""

    PLUS = 1
    MINUS = -1

    def __str__(self) -> str:
        """Return `+` or `-`."""
        if self == Sign.PLUS:
            return "+"
        return "-"

    @staticmethod
    def plus_if(b: bool) -> Sign:
        """Return `+` if `b` is `True`, `-` otherwise."""
        if b:
            return Sign.PLUS
        return Sign.MINUS

    @staticmethod
    def minus_if(b: bool) -> Sign:
        """Return `-` if `b` is `True`, `+` otherwise."""
        if b:
            return Sign.MINUS
        return Sign.PLUS

    def __neg__(self) -> Sign:
        """Swap the sign."""
        return Sign.minus_if(self == Sign.PLUS)

    @typing.overload
    def __mul__(self, other: Sign) -> Sign: ...

    @typing.overload
    def __mul__(self, other: int) -> int: ...

    @typing.overload
    def __mul__(self, other: float) -> float: ...

    @typing.overload
    def __mul__(self, other: complex) -> complex: ...

    def __mul__(self, other: Sign | complex) -> Sign | int | float | complex:
        """Multiply the sign with another sign or a number."""
        if isinstance(other, Sign):
            return Sign.plus_if(self == other)
        if isinstance(other, int):
            return int(self) * other
        if isinstance(other, float):
            return float(self) * other
        if isinstance(other, complex):
            return complex(self) * other
        return NotImplemented

    @typing.overload
    def __rmul__(self, other: int) -> int: ...

    @typing.overload
    def __rmul__(self, other: float) -> float: ...

    @typing.overload
    def __rmul__(self, other: complex) -> complex: ...

    def __rmul__(self, other: complex) -> int | float | complex:
        """Multiply the sign with a number."""
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __int__(self) -> int:
        """Return `1` for `+` and `-1` for `-`."""
        # mypy does not infer the return type correctly
        return self.value  # type: ignore[no-any-return]

    def __float__(self) -> float:
        """Return `1.0` for `+` and `-1.0` for `-`."""
        return float(self.value)

    def __complex__(self) -> complex:
        """Return `1.0 + 0j` for `+` and `-1.0 + 0j` for `-`."""
        return complex(self.value)


class ComplexUnit(Enum):
    """
    Complex unit: 1, -1, j, -j.

    Complex units can be multiplied with other complex units,
    with Python constants 1, -1, 1j, -1j, and can be negated.
    """

    # HACK: complex(u) == (1j) ** u.value for all u in ComplexUnit.

    ONE = 0
    J = 1
    MINUS_ONE = 2
    MINUS_J = 3

    @staticmethod
    def try_from(value: ComplexUnit | SupportsComplexCtor) -> ComplexUnit | None:
        """Return the ComplexUnit instance if the value is compatible, None otherwise."""
        if isinstance(value, ComplexUnit):
            return value
        try:
            value = complex(value)
        except Exception:
            return None
        if value == 1:
            return ComplexUnit.ONE
        if value == -1:
            return ComplexUnit.MINUS_ONE
        if value == 1j:
            return ComplexUnit.J
        if value == -1j:
            return ComplexUnit.MINUS_J
        return None

    @staticmethod
    def from_properties(*, sign: Sign = Sign.PLUS, is_imag: bool = False) -> ComplexUnit:
        """Construct ComplexUnit from its properties."""
        osign = 0 if sign == Sign.PLUS else 2
        oimag = 1 if is_imag else 0
        return ComplexUnit(osign + oimag)

    @property
    def sign(self) -> Sign:
        """Return the sign."""
        return Sign.plus_if(self.value < 2)

    @property
    def is_imag(self) -> bool:
        """Return `True` if `j` or `-j`."""
        return bool(self.value % 2)

    def __complex__(self) -> complex:
        """Return the unit as complex number."""
        ret: complex = 1j**self.value
        return ret

    def __str__(self) -> str:
        """Return a string representation of the unit."""
        result = "1j" if self.is_imag else "1"
        if self.sign == Sign.MINUS:
            result = "-" + result
        return result

    def __mul__(self, other: ComplexUnit | SupportsComplexCtor) -> ComplexUnit:
        """Multiply the complex unit with a number."""
        if isinstance(other, ComplexUnit):
            return ComplexUnit((self.value + other.value) % 4)
        if other_ := ComplexUnit.try_from(other):
            return self.__mul__(other_)
        return NotImplemented

    def __rmul__(self, other: SupportsComplexCtor) -> ComplexUnit:
        """Multiply the complex unit with a number."""
        return self.__mul__(other)

    def __neg__(self) -> ComplexUnit:
        """Return the opposite of the complex unit."""
        return ComplexUnit((self.value + 2) % 4)


class IXYZ(Enum):
    """I, X, Y or Z."""

    I = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix representation."""
        if self == IXYZ.I:
            return Ops.I
        if self == IXYZ.X:
            return Ops.X
        if self == IXYZ.Y:
            return Ops.Y
        if self == IXYZ.Z:
            return Ops.Z
        typing_extensions.assert_never(self)


class Axis(Enum):
    """Axis: `X`, `Y` or `Z`."""

    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix representation."""
        if self == Axis.X:
            return Ops.X
        if self == Axis.Y:
            return Ops.Y
        if self == Axis.Z:
            return Ops.Z
        typing_extensions.assert_never(self)


class Plane(Enum):
    # TODO: Refactor using match
    """Plane: `XY`, `YZ` or `XZ`."""

    XY = enum.auto()
    YZ = enum.auto()
    XZ = enum.auto()

    @property
    def axes(self) -> tuple[Axis, Axis]:
        """Return the pair of axes that carry the plane."""
        if self == Plane.XY:
            return (Axis.X, Axis.Y)
        if self == Plane.YZ:
            return (Axis.Y, Axis.Z)
        if self == Plane.XZ:
            return (Axis.X, Axis.Z)
        typing_extensions.assert_never(self)

    @property
    def orth(self) -> Axis:
        """Return the axis orthogonal to the plane."""
        if self == Plane.XY:
            return Axis.Z
        if self == Plane.YZ:
            return Axis.X
        if self == Plane.XZ:
            return Axis.Y
        typing_extensions.assert_never(self)

    @property
    def cos(self) -> Axis:
        """Return the axis of the plane that conventionally carries the cos."""
        if self == Plane.XY:
            return Axis.X
        if self == Plane.YZ:
            return Axis.Z  # former convention was Y
        if self == Plane.XZ:
            return Axis.Z  # former convention was X
        typing_extensions.assert_never(self)

    @property
    def sin(self) -> Axis:
        """Return the axis of the plane that conventionally carries the sin."""
        if self == Plane.XY:
            return Axis.Y
        if self == Plane.YZ:
            return Axis.Y  # former convention was Z
        if self == Plane.XZ:
            return Axis.X  # former convention was Z
        typing_extensions.assert_never(self)

    @overload
    def polar(self, angle: float) -> tuple[float, float, float]: ...

    @overload
    def polar(self, angle: Expression) -> tuple[Expression, Expression, Expression]: ...

    def polar(
        self, angle: ExpressionOrFloat
    ) -> tuple[float, float, float] | tuple[ExpressionOrFloat, ExpressionOrFloat, ExpressionOrFloat]:
        """Return the Cartesian coordinates of the point of module 1 at the given angle, following the conventional orientation for cos and sin."""
        pp = (self.cos, self.sin)
        cos, sin = cos_sin(angle)
        if pp == (Axis.X, Axis.Y):
            return (cos, sin, 0)
        if pp == (Axis.Z, Axis.Y):
            return (0, sin, cos)
        if pp == (Axis.Z, Axis.X):
            return (sin, 0, cos)
        raise RuntimeError("Unreachable.")  # pragma: no cover

    @staticmethod
    def from_axes(a: Axis, b: Axis) -> Plane:
        """Return the plane carried by the given axes."""
        ab = {a, b}
        if ab == {Axis.X, Axis.Y}:
            return Plane.XY
        if ab == {Axis.Y, Axis.Z}:
            return Plane.YZ
        if ab == {Axis.X, Axis.Z}:
            return Plane.XZ
        assert a == b
        raise ValueError(f"Cannot make a plane giving the same axis {a} twice.")

"""Pauli gates ± {1,j} × {I, X, Y, Z}."""  # noqa: RUF002

from __future__ import annotations

import dataclasses
import enum
import functools
import math
import typing
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import typing_extensions

from graphix import type_utils
from graphix._db import WellKnownMatrix

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    from graphix.states import PlanarState


class IXYZ(Enum):
    """I, X, Y or Z."""

    I = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()


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
        if isinstance(other, int | float | complex):
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

    # HACK: Related to arg(z)
    PLUS = 0
    PLUS_J = 1
    MINUS = 2
    MINUS_J = 3

    @staticmethod
    def try_from_complex(value: complex) -> ComplexUnit | None:
        """Return the ComplexUnit instance if the value is compatible, None otherwise."""
        if value == 1:
            return ComplexUnit.PLUS
        if value == -1:
            return ComplexUnit.MINUS
        if value == 1j:
            return ComplexUnit.PLUS_J
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

    def __repr__(self) -> str:
        """Return a string representation of the unit."""
        result = "1j" if self.is_imag else "1"
        if self.sign == Sign.MINUS:
            result = "-" + result
        return result

    def __mul__(self, other: ComplexUnit | complex) -> ComplexUnit:
        """Multiply the complex unit with another complex unit."""
        if isinstance(other, ComplexUnit):
            return ComplexUnit((self.value + other.value) % 4)
        if other_ := ComplexUnit.try_from_complex(other):
            return self.__mul__(other_)
        return NotImplemented

    def __rmul__(self, other: complex) -> ComplexUnit:
        """Multiply the complex unit with a number."""
        if isinstance(other, complex):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> ComplexUnit:
        """Return the opposite of the complex unit."""
        return ComplexUnit((self.value + 2) % 4)


class Axis(Enum):
    """Axis: `X`, `Y` or `Z`."""

    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()

    @property
    def op(self) -> npt.NDArray[np.complex128]:
        """Return the single qubit operator associated to the axis."""
        if self == Axis.X:
            return WellKnownMatrix.X
        if self == Axis.Y:
            return WellKnownMatrix.Y
        if self == Axis.Z:
            return WellKnownMatrix.Z
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

    def polar(self, angle: float) -> tuple[float, float, float]:
        """Return the Cartesian coordinates of the point of module 1 at the given angle, following the conventional orientation for cos and sin."""
        pp = (self.cos, self.sin)
        if pp == (Axis.X, Axis.Y):
            return (math.cos(angle), math.sin(angle), 0)
        if pp == (Axis.Z, Axis.Y):
            return (0, math.sin(angle), math.cos(angle))
        if pp == (Axis.Z, Axis.X):
            return (math.sin(angle), 0, math.cos(angle))
        raise RuntimeError("Unreachable.")

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


@functools.lru_cache(maxsize=None)
def _matmul_impl(lhs: IXYZ, rhs: IXYZ) -> Pauli:
    """Return the product of two Paulis."""
    if lhs == IXYZ.I:
        return Pauli(rhs)
    if rhs == IXYZ.I:
        return Pauli(lhs)
    if lhs == rhs:
        return Pauli()
    lr = (lhs, rhs)
    if lr == (IXYZ.X, IXYZ.Y):
        return Pauli(IXYZ.Z, ComplexUnit.PLUS_J)
    if lr == (IXYZ.Y, IXYZ.X):
        return Pauli(IXYZ.Z, ComplexUnit.MINUS_J)
    if lr == (IXYZ.Y, IXYZ.Z):
        return Pauli(IXYZ.X, ComplexUnit.PLUS_J)
    if lr == (IXYZ.Z, IXYZ.Y):
        return Pauli(IXYZ.X, ComplexUnit.MINUS_J)
    if lr == (IXYZ.Z, IXYZ.X):
        return Pauli(IXYZ.Y, ComplexUnit.PLUS_J)
    if lr == (IXYZ.X, IXYZ.Z):
        return Pauli(IXYZ.Y, ComplexUnit.MINUS_J)
    raise RuntimeError("Unreachable.")


@dataclasses.dataclass(frozen=True)
class Pauli:
    """Pauli gate: `u * {I, X, Y, Z}` where u is a complex unit.

    Pauli gates can be multiplied with other Pauli gates (with `@`),
    with complex units and unit constants (with `*`),
    and can be negated.
    """

    symbol: IXYZ = IXYZ.I
    unit: ComplexUnit = ComplexUnit.PLUS
    I: ClassVar[Pauli]
    X: ClassVar[Pauli]
    Y: ClassVar[Pauli]
    Z: ClassVar[Pauli]

    @staticmethod
    def from_axis(axis: Axis) -> Pauli:
        """Return the Pauli associated to the given axis."""
        return Pauli(IXYZ[axis.name])

    @property
    def axis(self) -> Axis:
        """Return the axis associated to the Pauli.

        Fails if the Pauli is identity.
        """
        if self.symbol == IXYZ.I:
            raise ValueError("I is not an axis.")
        return Axis[self.symbol.name]

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix of the Pauli gate."""
        co = complex(self.unit)
        if self.symbol == IXYZ.I:
            return co * WellKnownMatrix.I
        if self.symbol == IXYZ.X:
            return co * WellKnownMatrix.X
        if self.symbol == IXYZ.Y:
            return co * WellKnownMatrix.Y
        if self.symbol == IXYZ.Z:
            return co * WellKnownMatrix.Z
        typing_extensions.assert_never(self.symbol)

    def get_eigenstate(self, eigenvalue: int | Sign = 0) -> PlanarState:
        """Return the eigenstate of the Pauli."""
        from graphix.states import BasicStates

        if isinstance(eigenvalue, Sign):
            # Normalize the eigenvalue
            eigenvalue = 0 if eigenvalue == Sign.PLUS else 1

        if self.symbol == IXYZ.X:
            return BasicStates.PLUS if eigenvalue == 0 else BasicStates.MINUS
        if self.symbol == IXYZ.Y:
            return BasicStates.PLUS_I if eigenvalue == 0 else BasicStates.MINUS_I
        if self.symbol == IXYZ.Z:
            return BasicStates.ZERO if eigenvalue == 0 else BasicStates.ONE
        # Any state is eigenstate of the identity
        if self.symbol == IXYZ.I:
            return BasicStates.PLUS
        typing_extensions.assert_never(self.symbol)

    def __str__(self) -> str:
        """Return a string representation of the Pauli (without module prefix)."""
        if self.unit == ComplexUnit.PLUS:
            return str(self.symbol)
        if self.unit == ComplexUnit.MINUS:
            return f"-{self.symbol}"
        if self.unit == ComplexUnit.PLUS_J:
            return f"1j * {self.symbol}"
        if self.unit == ComplexUnit.MINUS_J:
            return f"-1j * {self.symbol}"
        typing_extensions.assert_never(self.unit)

    def __matmul__(self, other: Pauli) -> Pauli:
        """Return the product of two Paulis."""
        if isinstance(other, Pauli):
            return _matmul_impl(self.symbol, other.symbol) * (self.unit * other.unit)
        return NotImplemented

    def __mul__(self, other: ComplexUnit) -> Pauli:
        """Return the product of two Paulis."""
        if isinstance(other, ComplexUnit):
            return dataclasses.replace(self, unit=self.unit * other)
        return NotImplemented

    def __rmul__(self, other: ComplexUnit) -> Pauli:
        """Return the product of two Paulis."""
        if isinstance(other, ComplexUnit):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> Pauli:
        """Return the opposite."""
        return dataclasses.replace(self, unit=-self.unit)

    @staticmethod
    def iterate(include_unit: bool = True) -> Iterator[Pauli]:
        """Iterate over all Pauli gates.

        Parameters
        ----------
            include_unit (bool, optional): Include the unit in the iteration. Defaults to True.
        """
        us = iter(ComplexUnit) if include_unit else (ComplexUnit.PLUS,)
        for unit in us:
            for symbol in IXYZ:
                yield Pauli(symbol, unit)


Pauli.I = Pauli(IXYZ.I)
Pauli.X = Pauli(IXYZ.X)
Pauli.Y = Pauli(IXYZ.Y)
Pauli.Z = Pauli(IXYZ.Z)

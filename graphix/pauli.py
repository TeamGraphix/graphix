"""Pauli gates ± {1,j} × {I, X, Y, Z}."""  # noqa: RUF002

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, ClassVar

import typing_extensions

from graphix._db import WellKnownMatrix
from graphix.fundamentals import IXYZ, Axis, ComplexUnit, Sign

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    from graphix.states import PlanarState


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

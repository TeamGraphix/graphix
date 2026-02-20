"""Pauli gates ± {1,j} × {I, X, Y, Z}."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import typing_extensions

from graphix.fundamentals import IXYZ_VALUES, Axis, ComplexUnit, I, SupportsComplexCtor
from graphix.ops import Ops
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    from graphix.fundamentals import IXYZ
    from graphix.states import PlanarState


class _PauliMeta(type):
    def __iter__(cls) -> Iterator[Pauli]:
        """Iterate over all Pauli gates, including the unit."""
        return Pauli.iterate()


@dataclasses.dataclass(frozen=True)
class Pauli(metaclass=_PauliMeta):
    r"""Pauli gate: ``u * {I, X, Y, Z}`` where u is a complex unit.

    Pauli gates can be multiplied with other Pauli gates (with ``@``),
    with complex units and unit constants (with ``*``),
    and can be negated.
    """

    symbol: IXYZ = I
    unit: ComplexUnit = ComplexUnit.ONE
    I: ClassVar[Pauli]
    X: ClassVar[Pauli]
    Y: ClassVar[Pauli]
    Z: ClassVar[Pauli]

    @staticmethod
    def from_axis(axis: Axis) -> Pauli:
        """Return the Pauli associated to the given axis."""
        return Pauli(axis)

    @property
    def axis(self) -> Axis:
        """Return the axis associated to the Pauli.

        Fails if the Pauli is identity.
        """
        if self.symbol == I:
            raise ValueError("I is not an axis.")
        return self.symbol

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix of the Pauli gate."""
        co = complex(self.unit)
        return co * Ops.from_ixyz(self.symbol)

    def eigenstate(self, binary: int = 0) -> PlanarState:
        """Return the eigenstate of the Pauli."""
        if binary not in {0, 1}:
            raise ValueError("b must be 0 or 1.")
        match self.symbol:
            case Axis.X:
                return BasicStates.PLUS if binary == 0 else BasicStates.MINUS
            case Axis.Y:
                return BasicStates.PLUS_I if binary == 0 else BasicStates.MINUS_I
            case Axis.Z:
                return BasicStates.ZERO if binary == 0 else BasicStates.ONE
            case _:
                # Any state is eigenstate of the identity
                if self.symbol == I:
                    return BasicStates.PLUS
                typing_extensions.assert_never(self.symbol)

    def _repr_impl(self, prefix: str | None) -> str:
        """Return ``repr`` string with an optional prefix."""
        sym = self.symbol.name
        if prefix is not None:
            sym = f"{prefix}.{sym}"
        match self.unit:
            case ComplexUnit.ONE:
                return sym
            case ComplexUnit.MINUS_ONE:
                return f"-{sym}"
            case ComplexUnit.J:
                return f"1j * {sym}"
            case ComplexUnit.MINUS_J:
                return f"-1j * {sym}"
            case _:
                typing_extensions.assert_never(self.unit)

    def __repr__(self) -> str:
        """Return a string representation of the Pauli."""
        return self._repr_impl(self.__class__.__name__)

    def __str__(self) -> str:
        """Return a simplified string representation of the Pauli."""
        return self._repr_impl(None)

    @staticmethod
    def _matmul_impl(lhs: IXYZ, rhs: IXYZ) -> Pauli:
        """Return the product of ``lhs`` and ``rhs`` ignoring units."""
        if lhs == I:
            return Pauli(rhs)
        if rhs == I:
            return Pauli(lhs)
        if lhs == rhs:
            return Pauli()
        lr = (lhs, rhs)
        match lr:
            case (Axis.X, Axis.Y):
                return Pauli(Axis.Z, ComplexUnit.J)
            case (Axis.Y, Axis.X):
                return Pauli(Axis.Z, ComplexUnit.MINUS_J)
            case (Axis.Y, Axis.Z):
                return Pauli(Axis.X, ComplexUnit.J)
            case (Axis.Z, Axis.Y):
                return Pauli(Axis.X, ComplexUnit.MINUS_J)
            case (Axis.Z, Axis.X):
                return Pauli(Axis.Y, ComplexUnit.J)
            case (Axis.X, Axis.Z):
                return Pauli(Axis.Y, ComplexUnit.MINUS_J)
            case _:
                raise RuntimeError("Unreachable.")  # pragma: no cover

    def __matmul__(self, other: Pauli) -> Pauli:
        """Return the product of two Paulis."""
        if isinstance(other, Pauli):
            return self._matmul_impl(self.symbol, other.symbol) * (self.unit * other.unit)
        return NotImplemented

    def __mul__(self, other: ComplexUnit | SupportsComplexCtor) -> Pauli:
        """Return the product of two Paulis."""
        if u := ComplexUnit.try_from(other):
            return dataclasses.replace(self, unit=self.unit * u)
        return NotImplemented

    def __rmul__(self, other: ComplexUnit | SupportsComplexCtor) -> Pauli:
        """Return the product of two Paulis."""
        return self.__mul__(other)

    def __neg__(self) -> Pauli:
        """Return the opposite."""
        return dataclasses.replace(self, unit=-self.unit)

    @staticmethod
    def iterate(symbol_only: bool = False) -> Iterator[Pauli]:
        """Iterate over all Pauli gates.

        Parameters
        ----------
            symbol_only (bool, optional): Exclude the unit in the iteration. Defaults to False.
        """
        us = (ComplexUnit.ONE,) if symbol_only else tuple(ComplexUnit)
        for symbol in IXYZ_VALUES:
            for unit in us:
                yield Pauli(symbol, unit)


Pauli.I = Pauli(I)
Pauli.X = Pauli(Axis.X)
Pauli.Y = Pauli(Axis.Y)
Pauli.Z = Pauli(Axis.Z)

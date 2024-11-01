"""Pauli gates ± {1,j} × {I, X, Y, Z}."""  # noqa: RUF002

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, ClassVar

import typing_extensions

from graphix.fundamentals import IXYZ, Axis, ComplexUnit, SupportsComplexCtor
from graphix.ops import Ops
from graphix.states import BasicStates

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
    raise RuntimeError("Unreachable.")  # pragma: no cover


class _PauliMeta(type):
    def __iter__(cls) -> Iterator[Pauli]:
        """Iterate over all Pauli gates, including the unit."""
        return Pauli.iterate()


@dataclasses.dataclass(frozen=True)
class Pauli(metaclass=_PauliMeta):
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
            return co * Ops.I
        if self.symbol == IXYZ.X:
            return co * Ops.X
        if self.symbol == IXYZ.Y:
            return co * Ops.Y
        if self.symbol == IXYZ.Z:
            return co * Ops.Z
        typing_extensions.assert_never(self.symbol)

    def eigenstate(self, b: int = 0) -> PlanarState:
        """Return the eigenstate of the Pauli."""
        if b not in {0, 1}:
            raise ValueError("b must be 0 or 1.")
        if self.symbol == IXYZ.X:
            return BasicStates.PLUS if b == 0 else BasicStates.MINUS
        if self.symbol == IXYZ.Y:
            return BasicStates.PLUS_I if b == 0 else BasicStates.MINUS_I
        if self.symbol == IXYZ.Z:
            return BasicStates.ZERO if b == 0 else BasicStates.ONE
        # Any state is eigenstate of the identity
        if self.symbol == IXYZ.I:
            return BasicStates.PLUS
        typing_extensions.assert_never(self.symbol)

    def _repr_impl(self, prefix: str | None) -> str:
        sym = self.symbol.name
        if prefix is not None:
            sym = f"{prefix}.{sym}"
        if self.unit == ComplexUnit.PLUS:
            return sym
        if self.unit == ComplexUnit.MINUS:
            return f"-{sym}"
        if self.unit == ComplexUnit.PLUS_J:
            return f"1j * {sym}"
        if self.unit == ComplexUnit.MINUS_J:
            return f"-1j * {sym}"
        typing_extensions.assert_never(self.unit)

    def __repr__(self) -> str:
        """Return a string representation of the Pauli."""
        return self._repr_impl(self.__class__.__name__)

    def __str__(self) -> str:
        """Return a simplified string representation of the Pauli."""
        return self._repr_impl(None)

    def __matmul__(self, other: Pauli) -> Pauli:
        """Return the product of two Paulis."""
        if isinstance(other, Pauli):
            return _matmul_impl(self.symbol, other.symbol) * (self.unit * other.unit)
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
    def iterate(include_unit: bool = True) -> Iterator[Pauli]:
        """Iterate over all Pauli gates.

        Parameters
        ----------
            include_unit (bool, optional): Include the unit in the iteration. Defaults to True.
        """
        us = tuple(ComplexUnit) if include_unit else (ComplexUnit.PLUS,)
        for symbol in IXYZ:
            for unit in us:
                yield Pauli(symbol, unit)


Pauli.I = Pauli(IXYZ.I)
Pauli.X = Pauli(IXYZ.X)
Pauli.Y = Pauli(IXYZ.Y)
Pauli.Z = Pauli(IXYZ.Z)

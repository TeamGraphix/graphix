"""Quantum states and operators."""

from __future__ import annotations

import abc
from abc import ABC
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pydantic.dataclasses
import typing_extensions

from graphix.fundamentals import Plane


# generic class State for all States
# FIXME: Name conflict
class State(ABC):
    """Abstract base class for single qubit states objects.

    Only requirement for concrete classes is to have
    a get_statevector() method that returns the statevector
    representation of the state
    """

    @abc.abstractmethod
    def get_statevector(self) -> npt.NDArray[np.complex128]:
        """Return the state vector."""

    def get_densitymatrix(self) -> npt.NDArray[np.complex128]:
        """Return the density matrix."""
        # return DM in 2**n x 2**n dim (2x2 here)
        return np.outer(self.get_statevector(), self.get_statevector().conj())


@pydantic.dataclasses.dataclass
class PlanarState(State):
    """Light object used to instantiate backends.

    doesn't cover all possible states but this is
    covered in :class:`graphix.sim.statevec.Statevec`
    and :class:`graphix.sim.densitymatrix.DensityMatrix`
    constructors.

    :param plane: One of the three planes (XY, XZ, YZ)
    :type plane: :class:`graphix.pauli.Plane`
    :param angle: angle IN RADIANS
    :type angle: complex
    :return: State
    :rtype: :class:`graphix.states.State` object
    """

    plane: Plane
    angle: float

    def __repr__(self) -> str:
        """Return a string representation of the planar state."""
        return f"graphix.states.PlanarState({self.plane}, {self.angle})"

    def __str__(self) -> str:
        """Return a string description of the planar state."""
        return f"PlanarState object defined in plane {self.plane} with angle {self.angle}."

    def get_statevector(self) -> npt.NDArray[np.complex128]:
        """Return the state vector."""
        if self.plane == Plane.XY:
            return np.asarray([1 / np.sqrt(2), np.exp(1j * self.angle) / np.sqrt(2)], dtype=np.complex128)

        if self.plane == Plane.YZ:
            return np.asarray([np.cos(self.angle / 2), 1j * np.sin(self.angle / 2)], dtype=np.complex128)

        if self.plane == Plane.XZ:
            return np.asarray([np.cos(self.angle / 2), np.sin(self.angle / 2)], dtype=np.complex128)
        # other case never happens since exhaustive
        typing_extensions.assert_never(self.plane)


# States namespace for input initialization.
class BasicStates:
    """Basic states."""

    ZERO: ClassVar[PlanarState] = PlanarState(Plane.XZ, 0)
    ONE: ClassVar[PlanarState] = PlanarState(Plane.XZ, np.pi)
    PLUS: ClassVar[PlanarState] = PlanarState(Plane.XY, 0)
    MINUS: ClassVar[PlanarState] = PlanarState(Plane.XY, np.pi)
    PLUS_I: ClassVar[PlanarState] = PlanarState(Plane.XY, np.pi / 2)
    MINUS_I: ClassVar[PlanarState] = PlanarState(Plane.XY, -np.pi / 2)
    # remove that in the end
    # need in TN backend
    VEC: ClassVar[list[PlanarState]] = [PLUS, MINUS, ZERO, ONE, PLUS_I, MINUS_I]

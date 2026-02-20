"""Quantum states and operators."""

from __future__ import annotations

import abc
import dataclasses
from abc import ABC
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import typing_extensions

from graphix.fundamentals import ANGLE_PI, Angle, Plane, angle_to_rad


# generic class State for all States
# FIXME: Name conflict
class State(ABC):
    """Abstract base class for single qubit states objects.

    Only requirement for concrete classes is to have
    a to_statevector() method that returns the statevector
    representation of the state
    """

    @abc.abstractmethod
    def to_statevector(self) -> npt.NDArray[np.complex128]:
        """Return the state vector."""

    def to_densitymatrix(self) -> npt.NDArray[np.complex128]:
        """Return the density matrix."""
        # return DM in 2**n x 2**n dim (2x2 here)
        return np.outer(self.to_statevector(), self.to_statevector().conj()).astype(np.complex128, copy=False)


@dataclasses.dataclass
class PlanarState(State):
    """Light object used to instantiate backends.

    doesn't cover all possible states but this is
    covered in :class:`graphix.sim.statevec.Statevec`
    and :class:`graphix.sim.densitymatrix.DensityMatrix`
    constructors.

    :param plane: One of the three planes (XY, XZ, YZ)
    :type plane: :class:`graphix.pauli.Plane`
    :param angle: Angle (in units of π)
    :type angle: float
    :return: State
    :rtype: :class:`graphix.states.State` object
    """

    plane: Plane
    angle: Angle

    def __repr__(self) -> str:
        """Return a string representation of the planar state."""
        return f"graphix.states.PlanarState({self.plane}, {self.angle})"

    def __str__(self) -> str:
        """Return a string description of the planar state."""
        return f"PlanarState object defined in plane {self.plane} with angle {self.angle}."

    def to_statevector(self) -> npt.NDArray[np.complex128]:
        """Return the state vector."""
        angle_rad = angle_to_rad(self.angle)
        match self.plane:
            case Plane.XY:
                return np.asarray([1 / np.sqrt(2), np.exp(1j * angle_rad) / np.sqrt(2)], dtype=np.complex128)
            case Plane.YZ:
                return np.asarray([np.cos(angle_rad / 2), 1j * np.sin(angle_rad / 2)], dtype=np.complex128)
            case Plane.XZ:
                return np.asarray([np.cos(angle_rad / 2), np.sin(angle_rad / 2)], dtype=np.complex128)
            case _:
                # other case never happens since exhaustive
                typing_extensions.assert_never(self.plane)


# States namespace for input initialization.
class BasicStates:
    """Basic states."""

    ZERO: ClassVar[PlanarState] = PlanarState(Plane.XZ, 0)
    ONE: ClassVar[PlanarState] = PlanarState(Plane.XZ, ANGLE_PI)
    PLUS: ClassVar[PlanarState] = PlanarState(Plane.XY, 0)
    MINUS: ClassVar[PlanarState] = PlanarState(Plane.XY, ANGLE_PI)
    PLUS_I: ClassVar[PlanarState] = PlanarState(Plane.XY, ANGLE_PI / 2)
    MINUS_I: ClassVar[PlanarState] = PlanarState(Plane.XY, -ANGLE_PI / 2)
    # remove that in the end
    # need in TN backend
    VEC: ClassVar[list[PlanarState]] = [PLUS, MINUS, ZERO, ONE, PLUS_I, MINUS_I]

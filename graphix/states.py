"""
quantum states and operators
"""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import typing_extensions

from graphix.pauli import Plane


# generic class State for all States
class State(abc.ABC):
    """Abstract base class for single qubit states objects.
    Only requirement for concrete classes is to have
    a statevector property that returns the statevector
    representation of the state
    """

    @functools.cached_property
    @abc.abstractmethod
    def statevector(self) -> npt.NDArray:
        pass

    @functools.cached_property
    def densitymatrix(self) -> npt.NDArray:
        # return DM in 2**n x 2**n dim (2x2 here)
        return np.outer(self.statevector, self.statevector.conj())


@dataclasses.dataclass(frozen=True)
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

    def __str__(self) -> str:
        return f"PlanarState object defined in plane {self.plane} with angle {self.angle}."

    @functools.cached_property
    @typing_extensions.override
    def statevector(self) -> npt.NDArray:
        if self.plane == Plane.XY:
            return np.array([1, np.exp(1j * self.angle)]) / np.sqrt(2)

        if self.plane == Plane.YZ:
            return np.array([np.cos(self.angle / 2), 1j * np.sin(self.angle / 2)])

        if self.plane == Plane.XZ:
            return np.array([np.cos(self.angle / 2), np.sin(self.angle / 2)])
        # other case never happens since exhaustive
        typing_extensions.assert_never(self.plane)


# States namespace for input initialization.
class BasicStates:
    ZERO: ClassVar = PlanarState(Plane.XZ, 0)
    ONE: ClassVar = PlanarState(Plane.XZ, np.pi)
    PLUS: ClassVar = PlanarState(Plane.XY, 0)
    MINUS: ClassVar = PlanarState(Plane.XY, np.pi)
    PLUS_I: ClassVar = PlanarState(Plane.XY, np.pi / 2)
    MINUS_I: ClassVar = PlanarState(Plane.XY, -np.pi / 2)
    # remove that in the end
    # need in TN backend
    VEC: ClassVar = [PLUS, MINUS, ZERO, ONE, PLUS_I, MINUS_I]

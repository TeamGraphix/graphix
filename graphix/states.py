"""Quantum states and operators."""

from __future__ import annotations

import abc
import typing
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core
import typing_extensions

from graphix.pauli import Plane


# generic class State for all States
class State(abc.ABC):
    """Abstract base class for single qubit states objects.

    Only requirement for concrete classes is to have
    a get_statevector() method that returns the statevector
    representation of the state
    """

    @abc.abstractmethod
    def get_statevector(self) -> npt.NDArray:
        """Return the state vector."""
        ...

    def get_densitymatrix(self) -> npt.NDArray:
        """Return the density matrix."""
        # return DM in 2**n x 2**n dim (2x2 here)
        return np.outer(self.get_statevector(), self.get_statevector().conj())

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        """Validate state."""

        def check_state(obj) -> State:
            if not isinstance(obj, State):
                raise ValueError("State expected")
            return obj

        return pydantic_core.core_schema.no_info_plain_validator_function(function=check_state)


class PlanarState(pydantic.BaseModel, State):
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
        return f"graphix.states.PlanarState(plane={self.plane}, angle={self.angle})"

    def __str__(self) -> str:
        """Return a string description of the planar state."""
        return f"PlanarState object defined in plane {self.plane} with angle {self.angle}."

    def get_statevector(self) -> npt.NDArray:
        """Return the state vector."""
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
    """Basic states."""

    ZERO = PlanarState(plane=Plane.XZ, angle=0)
    ONE = PlanarState(plane=Plane.XZ, angle=np.pi)
    PLUS = PlanarState(plane=Plane.XY, angle=0)
    MINUS = PlanarState(plane=Plane.XY, angle=np.pi)
    PLUS_I = PlanarState(plane=Plane.XY, angle=np.pi / 2)
    MINUS_I = PlanarState(plane=Plane.XY, angle=-np.pi / 2)
    # remove that in the end
    # need in TN backend
    VEC: ClassVar = [PLUS, MINUS, ZERO, ONE, PLUS_I, MINUS_I]

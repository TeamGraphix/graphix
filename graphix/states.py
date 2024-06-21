"""
quantum states and operators
"""

import abc

import numpy as np
import numpy.typing as npt
import pydantic

import graphix.pauli


# generic class State for all States
class State(abc.ABC):
    """Abstract base class for single qubit states objects.
    Only requirement for concrete classes is to have
    a get_statevector() method that returns the statevector
    representation of the state
    """

    @abc.abstractmethod
    def get_statevector(self) -> npt.NDArray:
        pass

    def get_densitymatrix(self) -> npt.NDArray:
        # return DM in 2**n x 2**n dim (2x2 here)
        return np.outer(self.get_statevector(), self.get_statevector().conj())


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

    plane: graphix.pauli.Plane
    angle: float

    def __repr__(self) -> str:
        return f"PlanarState object defined in plane {self.plane} with angle {self.angle}."

    def get_statevector(self) -> npt.NDArray:
        if self.plane == graphix.pauli.Plane.XY:
            return np.array([1, np.exp(1j * self.angle)]) / np.sqrt(2)

        if self.plane == graphix.pauli.Plane.YZ:
            return np.array([np.cos(self.angle / 2), 1j * np.sin(self.angle / 2)])

        if self.plane == graphix.pauli.Plane.XZ:
            return np.array([np.cos(self.angle / 2), np.sin(self.angle / 2)])
        # other case never happens since exhaustive
        assert False


# States namespace for input initialization.
class BasicStates:
    ZERO = PlanarState(plane=graphix.pauli.Plane.XZ, angle=0)
    ONE = PlanarState(plane=graphix.pauli.Plane.XZ, angle=np.pi)
    PLUS = PlanarState(plane=graphix.pauli.Plane.XY, angle=0)
    MINUS = PlanarState(plane=graphix.pauli.Plane.XY, angle=np.pi)
    PLUS_I = PlanarState(plane=graphix.pauli.Plane.XY, angle=np.pi / 2)
    MINUS_I = PlanarState(plane=graphix.pauli.Plane.XY, angle=-np.pi / 2)
    # remove that in the end
    # need in TN backend
    VEC = [PLUS, MINUS, ZERO, ONE, PLUS_I, MINUS_I]

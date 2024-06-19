"""
quantum states and operators
"""

import abc

import numpy as np
import pydantic

import graphix.pauli


# generic class State for all States
class State(abc.ABC):
    """Abstract base class for states objects.
    Only requirement for concrete classes is to have
    a get_statevector() method that returns the statevector
    representation of the state
    """

    @abc.abstractmethod
    def get_statevector(self) -> np.ndarray:
        pass


# don't turn it into Statevec here
# Weird not to allow all states?
# Made it inherit from more generic State class.
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

    def __repr__(self):
        return f"PlanarState object defined in plane {self.plane} with angle {self.angle}."

    def get_statevector(self) -> np.ndarray:
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


# Plane.cos.value Plane.cos is an Axis, Axis.value = 0,1,2 (enum)

# Everywhere this is called. use StateVec(State))
# class States:
#     plus = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])  # plus
#     minus = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])  # minus
#     zero = np.array([1.0, 0.0])  # zero
#     one = np.array([0.0, 1.0])  # one
#     iplus = np.array([1.0 / np.sqrt(2), 1.0j / np.sqrt(2)])  # +1 eigenstate of Pauli Y
#     iminus = np.array([1.0 / np.sqrt(2), -1.0j / np.sqrt(2)])  # -1 eigenstate of Pauli Y
#     vec = [plus, minus, zero, one, iplus, iminus]

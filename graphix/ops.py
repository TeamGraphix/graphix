"""Quantum states and operators."""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, ClassVar, overload

import numpy as np
import numpy.typing as npt

# assert_never added in Python 3.11
from typing_extensions import assert_never

from graphix import utils
from graphix.fundamentals import IXYZ, Axis, rad_of_angle
from graphix.parameter import Expression, cos_sin, exp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.fundamentals import Angle, ParameterizedAngle
    from graphix.parameter import ExpressionOrComplex, ExpressionOrFloat


class Ops:
    """Basic single- and two-qubits operators."""

    I: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, 1]]))
    X: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[0, 1], [1, 0]]))
    Y: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[0, -1j], [1j, 0]]))
    Z: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, -1]]))
    S: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, 1j]]))
    SDG: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 0], [0, -1j]]))
    H: ClassVar[npt.NDArray[np.complex128]] = utils.lock(np.asarray([[1, 1], [1, -1]]) / np.sqrt(2))
    CZ: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
        )
    )
    CNOT: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
        )
    )
    SWAP: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
        )
    )
    CCX: ClassVar[npt.NDArray[np.complex128]] = utils.lock(
        np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
        )
    )

    @overload
    @staticmethod
    def _cast_array(array: Iterable[Iterable[complex]], theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def _cast_array(
        array: Iterable[Iterable[ExpressionOrComplex]], theta: ExpressionOrFloat
    ) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]: ...

    @staticmethod
    def _cast_array(
        array: Iterable[Iterable[ExpressionOrComplex]], theta: ExpressionOrFloat
    ) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        if isinstance(theta, Expression):
            return np.asarray(array, dtype=np.object_)
        return np.asarray(array, dtype=np.complex128)

    @overload
    @staticmethod
    def rx(theta: Angle) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rx(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rx(theta: ParameterizedAngle) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """X rotation.

        Parameters
        ----------
        theta : float
            rotation angle in units of π

        Returns
        -------
        operator : 2*2 np.asarray
        """
        cos, sin = cos_sin(rad_of_angle(theta) / 2)
        return Ops._cast_array([[cos, -1j * sin], [-1j * sin, cos]], theta)

    @overload
    @staticmethod
    def ry(theta: Angle) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def ry(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def ry(theta: ParameterizedAngle) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """Y rotation.

        Parameters
        ----------
        theta : Angle
            rotation angle in units of π

        Returns
        -------
        operator : 2*2 np.asarray
        """
        cos, sin = cos_sin(rad_of_angle(theta) / 2)
        return Ops._cast_array([[cos, -sin], [sin, cos]], theta)

    @overload
    @staticmethod
    def rz(theta: Angle) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rz(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rz(theta: ParameterizedAngle) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """Z rotation.

        Parameters
        ----------
        theta : Angle
            rotation angle in units of π

        Returns
        -------
        operator : 2*2 np.asarray
        """
        return Ops._cast_array([[exp(-1j * rad_of_angle(theta) / 2), 0], [0, exp(1j * rad_of_angle(theta) / 2)]], theta)

    @overload
    @staticmethod
    def rzz(theta: Angle) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rzz(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rzz(theta: ParameterizedAngle) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """zz-rotation.

        Equivalent to the sequence
        cnot(control, target),
        rz(target, angle),
        cnot(control, target)

        Parameters
        ----------
        theta : float
            rotation angle in units of π

        Returns
        -------
        operator : 4*4 np.asarray
        """
        return Ops._cast_array(Ops.CNOT @ np.kron(Ops.I, Ops.rz(theta)) @ Ops.CNOT, theta)

    @staticmethod
    def build_tensor_pauli_ops(n_qubits: int) -> npt.NDArray[np.complex128]:
        r"""Build all the 4^n tensor Pauli operators {I, X, Y, Z}^{\otimes n}.

        :param n_qubits: number of copies (qubits) to consider
        :type n_qubits: int
        :return: the array of the 4^n operators of shape (2^n, 2^n)
        :rtype: np.ndarray
        """
        if isinstance(n_qubits, int):
            if not n_qubits >= 1:
                raise ValueError(f"The number of qubits must be an integer <= 1 and not {n_qubits}.")
        else:
            raise TypeError(f"The number of qubits must be an integer and not {n_qubits}.")

        def _reducer(lhs: npt.NDArray[np.complex128], rhs: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
            return np.kron(lhs, rhs).astype(np.complex128, copy=False)

        return np.array([reduce(_reducer, i) for i in product((Ops.I, Ops.X, Ops.Y, Ops.Z), repeat=n_qubits)])

    @staticmethod
    def from_axis(axis: Axis) -> npt.NDArray[np.complex128]:
        """Return the matrix representation of an AXIS."""
        if axis == Axis.X:
            return Ops.X
        if axis == Axis.Y:
            return Ops.Y
        if axis == Axis.Z:
            return Ops.Z
        assert_never(axis)

    @staticmethod
    def from_ixyz(ixyz: IXYZ) -> npt.NDArray[np.complex128]:
        """Return the matrix representation of an IXYZ."""
        if ixyz == IXYZ.I:
            return Ops.I
        if ixyz == IXYZ.X:
            return Ops.X
        if ixyz == IXYZ.Y:
            return Ops.Y
        if ixyz == IXYZ.Z:
            return Ops.Z
        assert_never(ixyz)

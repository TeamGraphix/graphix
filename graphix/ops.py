"""Quantum states and operators."""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, ClassVar, overload

import numpy as np
import numpy.typing as npt

from graphix import utils
from graphix.parameter import Expression, cos_sin, exp

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    def rx(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rx(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rx(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """X rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.asarray
        """
        cos, sin = cos_sin(theta / 2)
        return Ops._cast_array(
            [[cos, -1j * sin], [-1j * sin, cos]],
            theta,
        )

    @overload
    @staticmethod
    def ry(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def ry(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def ry(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """Y rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.asarray
        """
        cos, sin = cos_sin(theta / 2)
        return Ops._cast_array([[cos, -sin], [sin, cos]], theta)

    @overload
    @staticmethod
    def rz(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rz(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rz(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """Z rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.asarray
        """
        return Ops._cast_array([[exp(-1j * theta / 2), 0], [0, exp(1j * theta / 2)]], theta)

    @overload
    @staticmethod
    def rzz(theta: float) -> npt.NDArray[np.complex128]: ...

    @overload
    @staticmethod
    def rzz(theta: Expression) -> npt.NDArray[np.object_]: ...

    @staticmethod
    def rzz(theta: ExpressionOrFloat) -> npt.NDArray[np.complex128] | npt.NDArray[np.object_]:
        """zz-rotation.

        Equivalent to the sequence
        cnot(control, target),
        rz(target, angle),
        cnot(control, target)

        Parameters
        ----------
        theta : float
            rotation angle in radian

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

        # TODO: Refactor this
        return np.array([reduce(np.kron, i) for i in product((Ops.I, Ops.X, Ops.Y, Ops.Z), repeat=n_qubits)])

"""Quantum states and operators."""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from graphix import utils


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

    @staticmethod
    def rx(theta: float) -> npt.NDArray[np.complex128]:
        """X rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.asarray
        """
        return np.asarray(
            [[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]],
            dtype=np.complex128,
        )

    @staticmethod
    def ry(theta: float) -> npt.NDArray[np.complex128]:
        """Y rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.asarray
        """
        return np.asarray(
            [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128
        )

    @staticmethod
    def rz(theta: float) -> npt.NDArray[np.complex128]:
        """Z rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.asarray
        """
        return np.asarray([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)

    @staticmethod
    def rzz(theta: float) -> npt.NDArray[np.complex128]:
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
        return np.asarray(Ops.CNOT @ np.kron(Ops.I, Ops.rz(theta)) @ Ops.CNOT, dtype=np.complex128)

    @staticmethod
    def build_tensor_pauli_ops(n_qubits: int) -> npt.NDArray[np.complex128]:
        r"""Build all the 4^n tensor Pauli operators {I, X, Y, Z}^{\otimes n}.

        :param n_qubits: number of copies (qubits) to consider
        :type n_qubits: int
        :return: the array of the 4^n operators of shape (2^n, 2^n)
        :rtype: np.ndarray
        """
        if isinstance(n_qubits, int):
            if not 1 <= n_qubits:
                raise ValueError(f"The number of qubits must be an integer <= 1 and not {n_qubits}.")
        else:
            raise TypeError(f"The number of qubits must be an integer and not {n_qubits}.")

        # TODO: Refactor this
        return np.array([reduce(np.kron, i) for i in product((Ops.I, Ops.X, Ops.Y, Ops.Z), repeat=n_qubits)])

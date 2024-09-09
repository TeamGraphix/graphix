"""Quantum states and operators."""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import ClassVar

import numpy as np
import numpy.typing as npt


class Ops:
    """Basic single- and two-qubits operators."""

    I: ClassVar = np.eye(2)
    X: ClassVar = np.array([[0, 1], [1, 0]])
    Y: ClassVar = np.array([[0, -1j], [1j, 0]])
    Z: ClassVar = np.array([[1, 0], [0, -1]])
    S: ClassVar = np.array([[1, 0], [0, 1j]])
    H: ClassVar = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CZ: ClassVar = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    CNOT: ClassVar = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    SWAP: ClassVar = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    CCX: ClassVar = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )

    @staticmethod
    def rx(theta) -> npt.NDArray:
        """X rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.array
        """
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

    @staticmethod
    def ry(theta) -> npt.NDArray:
        """Y rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.array
        """
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])

    @staticmethod
    def rz(theta) -> npt.NDArray:
        """Z rotation.

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        -------
        operator : 2*2 np.array
        """
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])

    @staticmethod
    def rzz(theta) -> npt.NDArray:
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
        operator : 4*4 np.array
        """
        return Ops.CNOT @ np.kron(Ops.I, Ops.rz(theta)) @ Ops.CNOT

    @staticmethod
    def build_tensor_pauli_ops(n_qubits: int) -> npt.NDArray:
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

        tensor_pauli_ops = [
            reduce(lambda x, y: np.kron(x, y), i) for i in product((Ops.I, Ops.X, Ops.Y, Ops.Z), repeat=n_qubits)
        ]

        return np.array(tensor_pauli_ops)

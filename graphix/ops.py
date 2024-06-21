"""
quantum states and operators
"""

from functools import reduce
from itertools import product

import numpy as np


class Ops:
    """Basic single- and two-qubits operators"""

    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    s = np.array([[1, 0], [0, 1j]])
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    ccx = np.array(
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
    Pauli_ops = [np.eye(2), x, y, z]

    @staticmethod
    def Rx(theta):
        """x rotation

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        ----------
        operator : 2*2 np.array
        """

        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

    @staticmethod
    def Ry(theta):
        """y rotation

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        ----------
        operator : 2*2 np.array
        """
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])

    @staticmethod
    def Rz(theta):
        """z rotation

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        ----------
        operator : 2*2 np.array
        """
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])

    @staticmethod
    def Rzz(theta):
        """zz-rotation.
        Equivalent to the sequence
        CNOT(control, target),
        Rz(target, angle),
        CNOT(control, target)

        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        ----------
        operator : 4*4 np.array
        """
        return Ops.cnot @ np.kron(np.eye(2), Ops.Rz(theta)) @ Ops.cnot

    @staticmethod
    def build_tensor_Pauli_ops(n_qubits: int):
        """Method to build all the 4^n tensor Pauli operators {I, X, Y, Z}^{\otimes n}

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

        tensor_Pauli_ops = [reduce(lambda x, y: np.kron(x, y), i) for i in product(Ops.Pauli_ops, repeat=n_qubits)]

        return np.array(tensor_Pauli_ops)

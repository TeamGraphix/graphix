"""
quantum states and operators
"""

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

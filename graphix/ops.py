"""
quantum states and operators
"""

import qiskit.quantum_info as qi
import numpy as np


class States:
    """ Pauli basis states implemented with qiskit.quantum_info
    """
    xplus_state = qi.Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])
    yplus_state = qi.Statevector([1 / np.sqrt(2), 1j / np.sqrt(2)])
    zplus_state = qi.Statevector([1, 0])  # |0>
    xminus_state = qi.Statevector([1 / np.sqrt(2), -1 / np.sqrt(2)])
    yminus_state = qi.Statevector([1 / np.sqrt(2), -1j / np.sqrt(2)])
    zminus_state = qi.Statevector([0, 1])  # |1>


class Ops:
    """ Basic operatos implemented with qiskit.quantum_info
    """
    x = qi.Operator(np.array([[0, 1], [1, 0]]))
    y = qi.Operator(np.array([[0, -1j], [1j, 0]]))
    z = qi.Operator(np.array([[1, 0], [0, -1]]))
    s = qi.Operator(np.array([[1, 0], [0, 1j]]))
    h = qi.Operator(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    cz = qi.Operator(np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1]]))
    cnot = qi.Operator(np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]]))
    swap = qi.Operator(np.array([[1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]]))

    @staticmethod
    def Rx(theta):
        """ x rotation
        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        ----------
        operator : qiskit.quantum_info.Opearator
        """
        return qi.Operator(np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]))

    @staticmethod
    def Ry(theta):
        """ y rotation
        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        ----------
        operator : qiskit.quantum_info.Opearator
        """
        return qi.Operator(np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ]))

    @staticmethod
    def Rz(theta):
        """ z rotation
        Parameters
        ----------
        theta : float
            rotation angle in radian

        Returns
        ----------
        operator : qiskit.quantum_info.Opearator
        """
        return qi.Operator(np.array(
            [
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ]))

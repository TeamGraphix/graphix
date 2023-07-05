import numpy as np
from graphix.sim.density_matrix import DensityMatrix
import abc


class NoiseModel(abc.ABC):
    """No noise"""

    def __init__(self):
        pass
    
    # NOTE is this useful?
    def assign_simulator(self, simulator):
        self.simulator = simulator

    @abc.abstractmethod
    def prepare_qubit(self, cmd):
        """return qubit to be added with preparation errors.
        """
        pass
    
    @abc.abstractmethod
    def entangle(self, cmd):
        """apply noise to qubits that happens in the CZ gate process"""
        pass

    @abc.abstractmethod
    def measure(self, cmd):
        """apply noise to qubits that happens in the measurement process"""
        pass

    @abc.abstractmethod
    def confuse_result(self, cmd):
        """assign wrong measurement result"""
        pass

    @abc.abstractmethod
    def byproduct_x(self, cmd):
        """apply noise to qubits that happens in the X gate process"""
        pass

    @abc.abstractmethod
    def byproduct_z(self, cmd):
        """apply noise to qubits that happens in the Z gate process"""
        pass

    @abc.abstractmethod
    def clifford(self, cmd):
        """apply noise to qubits that happens in the Clifford gate process"""
        # NOTE might be different depending on the gate.
        pass

    @abc.abstractmethod
    def tick_clock(self):
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
        pass

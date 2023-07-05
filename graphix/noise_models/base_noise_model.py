import numpy as np
from graphix.sim.density_matrix import DensityMatrix
from graphix.noise_models.noise_model import NoiseModel


class BaseNoiseModel(NoiseModel):
    """No noise"""

    def prepare_qubit(self, cmd):
        """return qubit to be added.
        in this base model, we return clean qubit in |+> state"""
        return DensityMatrix(nqubit=1)

    def entangle(self, cmd):
        """apply noise to qubits that happens in the CZ gate process"""
        pass

    def measure(self, cmd):
        """apply noise to qubits that happens in the measurement process"""
        pass

    def confuse_result(self, cmd):
        """assign wrong measurement result"""
        p = 0.0
        if np.random.rand() < p:
            self.simulator.result[cmd[1]] = 1 - self.simulator.result[cmd[1]]

    def byproduct_x(self, cmd):
        """apply noise to qubits that happens in the X gate process"""
        pass

    def byproduct_z(self, cmd):
        """apply noise to qubits that happens in the Z gate process"""
        pass

    def clifford(self, cmd):
        """apply noise to qubits that happens in the Clifford gate process"""
        pass

    def tick_clock(self):
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
        pass

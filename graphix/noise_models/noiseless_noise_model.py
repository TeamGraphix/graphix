import numpy as np

from graphix.channels import KrausChannel
from graphix.noise_models.noise_model import NoiseModel


class NoiselessNoiseModel(NoiseModel):
    """Noiseless noise model for testing.
    Only return the identity channel.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(self):
        pass

    def prepare_qubit(self):
        """return the channel to apply after clean single-qubit preparation. Here just identity."""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def entangle(self):
        """return noise model to qubits that happens after the CZ gats"""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(4)}])

    def measure(self):
        """apply noise to qubit to be measured."""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def confuse_result(self, cmd):
        """assign wrong measurement result"""
        p = 0.0
        if np.random.rand() < p:
            self.simulator.results[cmd[1]] = 1 - self.simulator.results[cmd[1]]

    def byproduct_x(self):
        """apply noise to qubits after X gate correction"""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def byproduct_z(self):
        """apply noise to qubits after Z gate correction"""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def clifford(self):
        """apply noise to qubits that happens in the Clifford gate process"""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def tick_clock(self):
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
        pass

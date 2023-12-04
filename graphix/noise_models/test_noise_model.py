import numpy as np
from graphix.noise_models.noise_model import NoiseModel
from graphix.channels import (
    Channel,
    create_dephasing_channel,
    create_depolarising_channel,
    create_2_qubit_depolarising_channel,
)
from graphix.ops import Ops

import tests.random_objects as randobj


class TestNoiseModel(NoiseModel):
    """Test noise model for testing.
    Only return the identity channel.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(self, x_error_prob=0.0, entanglement_error_prob=0.0, measure_channel_prob=0.0, measure_error_prob=0.0):
        self.x_error_prob = x_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob

    def prepare_qubit(self):
        """return the channel to apply after clean single-qubit preparation. Here just identity."""
        return Channel([{"parameter": 1.0, "operator": np.eye(2)}])

    def entangle(self):
        """return noise model to qubits that happens after the CZ gate"""
        return create_2_qubit_depolarising_channel(self.entanglement_error_prob)
        # randobj.rand_channel_kraus(dim=4)#

    def measure(self):
        """apply noise to qubit to be measured."""
        return create_depolarising_channel(self.measure_channel_prob)

    # randobj.rand_channel_kraus(dim=2, rank=3)

    # randobj.rand_channel_kraus(dim=2)
    # Random Pauli channel
    # Channel([{"parameter": np.sqrt(0.3), "operator": np.eye(2)},{"parameter": np.sqrt(0.005), "operator": Ops.x},{"parameter": np.sqrt(0.1), "operator": Ops.y},{"parameter": np.sqrt(1.-0.3-0.005-0.1), "operator": Ops.z}])
    # create_depolarising_channel(self.measure_channel_prob)

    # Channel([{"parameter": 1.0, "operator": np.eye(2)}])

    def confuse_result(self, cmd):
        """assign wrong measurement result
        cmd = "M"
        """
        # NOTE put self.measure_error_prob as argument of the method? Nope! Called in simulator

        print("before", self.simulator.results[cmd[1]])
        if np.random.rand() < self.measure_error_prob:
            self.simulator.results[cmd[1]] = 1 - self.simulator.results[cmd[1]]
        print("after", self.simulator.results[cmd[1]])

    def byproduct_x(self):
        """apply noise to qubits after X gate correction"""
        return create_depolarising_channel(self.x_error_prob)  # create_dephasing_channel(self.x_error_prob)

    def byproduct_z(self):
        """apply noise to qubits after Z gate correction"""
        return Channel([{"parameter": 1.0, "operator": np.eye(2)}])

    def clifford(self):
        """apply noise to qubits that happens in the Clifford gate process"""
        # TODO list separate different Cliffords to allow customization
        return Channel([{"parameter": 1.0, "operator": np.eye(2)}])

    def tick_clock(self):
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
        pass

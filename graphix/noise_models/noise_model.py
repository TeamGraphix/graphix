import abc


class NoiseModel(abc.ABC):
    """Abstract base class for all noise models."""

    # define the noise model parameters in there
    @abc.abstractmethod
    def __init__(self):
        pass

    # shared by all objects of the child class.
    def assign_simulator(self, simulator):
        self.simulator = simulator

    @abc.abstractmethod
    def prepare_qubit(self):
        """return qubit to be added with preparation errors."""
        pass

    @abc.abstractmethod
    def entangle(self):
        """apply noise to qubits that happens in the CZ gate process"""
        pass

    @abc.abstractmethod
    def measure(self):
        """apply noise to qubits that happens in the measurement process"""
        pass

    @abc.abstractmethod
    def confuse_result(self):
        """assign wrong measurement result"""
        pass

    @abc.abstractmethod
    def byproduct_x(self):
        """apply noise to qubits that happens in the X gate process"""
        pass

    @abc.abstractmethod
    def byproduct_z(self):
        """apply noise to qubits that happens in the Z gate process"""
        pass

    @abc.abstractmethod
    def clifford(self):
        """apply noise to qubits that happens in the Clifford gate process"""
        # NOTE might be different depending on the gate.
        pass

    @abc.abstractmethod
    def tick_clock(self):
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
        pass

"""Abstract base class for all noise models."""

from __future__ import annotations

import abc


class NoiseModel(abc.ABC):
    """Abstract base class for all noise models."""

    # shared by all objects of the child class.
    def assign_simulator(self, simulator) -> None:
        """Assign a simulator to the noise model."""
        self.simulator = simulator

    @abc.abstractmethod
    def prepare_qubit(self) -> None:
        """Return qubit to be added with preparation errors."""
        ...

    @abc.abstractmethod
    def entangle(self) -> None:
        """Apply noise to qubits that happens in the CZ gate process."""
        ...

    @abc.abstractmethod
    def measure(self) -> None:
        """Apply noise to qubits that happens in the measurement process."""
        ...

    @abc.abstractmethod
    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""

    @abc.abstractmethod
    def byproduct_x(self) -> None:
        """Apply noise to qubits that happens in the X gate process."""
        ...

    @abc.abstractmethod
    def byproduct_z(self) -> None:
        """Apply noise to qubits that happens in the Z gate process."""
        ...

    @abc.abstractmethod
    def clifford(self) -> None:
        """Apply noise to qubits that happens in the Clifford gate process."""
        # NOTE might be different depending on the gate.
        ...

    @abc.abstractmethod
    def tick_clock(self) -> None:
        """Notion of time in real devices - this is where we apply effect of T1 and T2.

        We assume commands that lie between 'T' commands run simultaneously on the device.
        """
        ...

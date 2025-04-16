"""Abstract base class for all noise models."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphix.channels import KrausChannel
    from graphix.simulator import PatternSimulator


class NoiseModel(abc.ABC):
    """Abstract base class for all noise models."""

    data: PatternSimulator

    # shared by all objects of the child class.
    def assign_simulator(self, simulator: PatternSimulator) -> None:
        """Assign a simulator to the noise model."""
        self.simulator = simulator

    @abc.abstractmethod
    def prepare_qubit(self) -> KrausChannel:
        """Return qubit to be added with preparation errors."""
        ...

    @abc.abstractmethod
    def entangle(self) -> KrausChannel:
        """Apply noise to qubits that happens in the CZ gate process."""
        ...

    @abc.abstractmethod
    def measure(self) -> KrausChannel:
        """Apply noise to qubits that happens in the measurement process."""
        ...

    @abc.abstractmethod
    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""

    @abc.abstractmethod
    def byproduct_x(self) -> KrausChannel:
        """Apply noise to qubits that happens in the X gate process."""
        ...

    @abc.abstractmethod
    def byproduct_z(self) -> KrausChannel:
        """Apply noise to qubits that happens in the Z gate process."""
        ...

    @abc.abstractmethod
    def clifford(self) -> KrausChannel:
        """Apply noise to qubits that happens in the Clifford gate process."""
        # NOTE might be different depending on the gate.
        ...

    @abc.abstractmethod
    def tick_clock(self) -> None:
        """Notion of time in real devices - this is where we apply effect of T1 and T2.

        We assume commands that lie between 'T' commands run simultaneously on the device.
        """
        ...

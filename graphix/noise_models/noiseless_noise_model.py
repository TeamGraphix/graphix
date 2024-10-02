"""Noiseless noise model for testing."""

from __future__ import annotations

import numpy as np

from graphix.channels import KrausChannel, KrausData
from graphix.noise_models.noise_model import NoiseModel


class NoiselessNoiseModel(NoiseModel):
    """Noiseless noise model for testing.

    Only return the identity channel.
    """

    def prepare_qubit(self):
        """Return the channel to apply after clean single-qubit preparation. Here just identity."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    def entangle(self):
        """Return noise model to qubits that happens after the CZ gates."""
        return KrausChannel([KrausData(1.0, np.eye(4))])

    def measure(self):
        """Apply noise to qubit to be measured."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""
        return result

    def byproduct_x(self):
        """Apply noise to qubits after X gate correction."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    def byproduct_z(self):
        """Apply noise to qubits after Z gate correction."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    def clifford(self):
        """Apply noise to qubits that happens in the Clifford gate process."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    def tick_clock(self):
        """Notion of time in real devices - this is where we apply effect of T1 and T2.

        See :meth:`NoiseModel.tick_clock`.
        """
        pass

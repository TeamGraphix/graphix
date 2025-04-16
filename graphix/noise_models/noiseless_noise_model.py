"""Noiseless noise model for testing."""

from __future__ import annotations

import numpy as np
import typing_extensions

from graphix.channels import KrausChannel, KrausData
from graphix.noise_models.noise_model import NoiseModel


class NoiselessNoiseModel(NoiseModel):
    """Noiseless noise model for testing.

    Only return the identity channel.
    """

    @typing_extensions.override
    def prepare_qubit(self) -> KrausChannel:
        """Return the channel to apply after clean single-qubit preparation. Here just identity."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def entangle(self) -> KrausChannel:
        """Return noise model to qubits that happens after the CZ gates."""
        return KrausChannel([KrausData(1.0, np.eye(4))])

    @typing_extensions.override
    def measure(self) -> KrausChannel:
        """Apply noise to qubit to be measured."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result."""
        return result

    @typing_extensions.override
    def byproduct_x(self) -> KrausChannel:
        """Apply noise to qubits after X gate correction."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def byproduct_z(self) -> KrausChannel:
        """Apply noise to qubits after Z gate correction."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def clifford(self) -> KrausChannel:
        """Apply noise to qubits that happens in the Clifford gate process."""
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def tick_clock(self) -> None:
        """Notion of time in real devices - this is where we apply effect of T1 and T2.

        See :meth:`NoiseModel.tick_clock`.
        """

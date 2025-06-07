"""Noise model that introduces no errors.

This class is useful for unit tests or benchmarks where deterministic
behaviour is required. All methods simply return an identity
:class:`~graphix.channels.KrausChannel`.
"""

from __future__ import annotations

import numpy as np
import typing_extensions

from graphix.channels import KrausChannel, KrausData
from graphix.noise_models.noise_model import NoiseModel


class NoiselessNoiseModel(NoiseModel):
    """Noise model that performs no operation."""

    @typing_extensions.override
    def prepare_qubit(self) -> KrausChannel:
        """Return the identity preparation channel.

        Returns
        -------
        KrausChannel
            Identity channel :math:`I_2`.
        """
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def entangle(self) -> KrausChannel:
        """Return the identity channel for entangling operations.

        Returns
        -------
        KrausChannel
            Identity channel :math:`I_4`.
        """
        return KrausChannel([KrausData(1.0, np.eye(4))])

    @typing_extensions.override
    def measure(self) -> KrausChannel:
        """Return the identity channel for measurements.

        Returns
        -------
        KrausChannel
            Identity channel :math:`I_2`.
        """
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def confuse_result(self, result: bool) -> bool:
        """Return the unmodified measurement result.

        Parameters
        ----------
        result : bool
            Ideal measurement outcome.

        Returns
        -------
        bool
            Same as ``result``.
        """
        return result

    @typing_extensions.override
    def byproduct_x(self) -> KrausChannel:
        """Return the identity channel for X corrections.

        Returns
        -------
        KrausChannel
            Identity channel :math:`I_2`.
        """
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def byproduct_z(self) -> KrausChannel:
        """Return the identity channel for Z corrections.

        Returns
        -------
        KrausChannel
            Identity channel :math:`I_2`.
        """
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def clifford(self) -> KrausChannel:
        """Return the identity channel for Clifford gates.

        Returns
        -------
        KrausChannel
            Identity channel :math:`I_2`.
        """
        return KrausChannel([KrausData(1.0, np.eye(2))])

    @typing_extensions.override
    def tick_clock(self) -> None:
        """Advance the simulator clock without applying errors.

        Notes
        -----
        This method is present for API compatibility and does not modify the
        internal state. See
        :meth:`~graphix.noise_models.noise_model.NoiseModel.tick_clock`.
        """
        pass

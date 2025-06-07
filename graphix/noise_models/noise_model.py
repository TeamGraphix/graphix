"""Abstract interface for noise models.

This module defines :class:`NoiseModel`, the base class used by
:class:`graphix.simulator.PatternSimulator` when running noisy
simulations. Child classes implement concrete noise processes by
overriding the abstract methods defined here.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphix.channels import KrausChannel
    from graphix.simulator import PatternSimulator


class NoiseModel(abc.ABC):
    """Base class for all noise models."""

    data: PatternSimulator

    # shared by all objects of the child class.
    def assign_simulator(self, simulator: PatternSimulator) -> None:
        """Assign the running simulator.

        Parameters
        ----------
        simulator : :class:`~graphix.simulator.PatternSimulator`
            Simulator instance that will use this noise model.
        """
        self.simulator = simulator

    @abc.abstractmethod
    def prepare_qubit(self) -> KrausChannel:
        """Return the preparation channel.

        Returns
        -------
        KrausChannel
            Channel applied after single-qubit preparation.
        """
        ...

    @abc.abstractmethod
    def entangle(self) -> KrausChannel:
        """Return the channel applied after entanglement.

        Returns
        -------
        KrausChannel
            Channel modeling noise during the CZ gate.
        """
        ...

    @abc.abstractmethod
    def measure(self) -> KrausChannel:
        """Return the measurement channel.

        Returns
        -------
        KrausChannel
            Channel applied immediately before measurement.
        """
        ...

    @abc.abstractmethod
    def confuse_result(self, result: bool) -> bool:
        """Return a possibly flipped measurement outcome.

        Parameters
        ----------
        result : bool
            Ideal measurement result.

        Returns
        -------
        bool
            Possibly corrupted result.
        """

    @abc.abstractmethod
    def byproduct_x(self) -> KrausChannel:
        """Return the channel for X by-product corrections.

        Returns
        -------
        KrausChannel
            Channel applied after an X correction.
        """
        ...

    @abc.abstractmethod
    def byproduct_z(self) -> KrausChannel:
        """Return the channel for Z by-product corrections.

        Returns
        -------
        KrausChannel
            Channel applied after a Z correction.
        """
        ...

    @abc.abstractmethod
    def clifford(self) -> KrausChannel:
        """Return the channel for Clifford gates.

        Returns
        -------
        KrausChannel
            Channel modeling the noise of Clifford operations.
        """
        # NOTE might be different depending on the gate.
        ...

    @abc.abstractmethod
    def tick_clock(self) -> None:
        """Advance the simulator clock.

        This accounts for idle errors such as :math:`T_1` and :math:`T_2`. All
        commands between consecutive ``T`` instructions are considered
        simultaneous.
        """
        ...

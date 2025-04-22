"""Noiseless noise model for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.noise_models.noise_model import CommandOrNoise, NoiseCommands, NoiseModel

if TYPE_CHECKING:
    from graphix.command import BaseM


class NoiselessNoiseModel(NoiseModel):
    """Noiseless noise model for testing.

    Only return the identity channel.
    """

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return []

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        return [cmd]

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result."""
        return result

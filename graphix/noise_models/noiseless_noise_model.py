"""Noise model that introduces no errors.

This class is useful for unit tests or benchmarks where deterministic
behaviour is required. All methods simply return an identity
:class:`~graphix.channels.KrausChannel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# override introduced in Python 3.12
from typing_extensions import override

from graphix.noise_models.noise_model import CommandOrNoise, NoiseModel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.command import BaseM
    from graphix.measurements import Outcome


class NoiselessNoiseModel(NoiseModel):
    """Noise model that performs no operation."""

    @override
    def input_nodes(self, nodes: Iterable[int]) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return []

    @override
    def command(self, cmd: CommandOrNoise) -> list[CommandOrNoise]:
        """Return the noise to apply to the command ``cmd``."""
        return [cmd]

    @override
    def confuse_result(self, cmd: BaseM, result: Outcome) -> Outcome:
        """Assign wrong measurement result."""
        return result

"""Depolarising noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import typing_extensions

from graphix.channels import KrausChannel, depolarising_channel, two_qubit_depolarising_channel
from graphix.command import BaseM, CommandKind
from graphix.noise_models.noise_model import A, CommandOrNoise, Noise, NoiseCommands, NoiseModel
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass
class DepolarisingNoise(Noise):
    """One-qubit depolarising noise with probabibity `prob`."""

    prob: float

    @property
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 1

    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return depolarising_channel(self.prob)


@dataclass
class TwoQubitDepolarisingNoise(Noise):
    """Two-qubits depolarising noise with probabibity `prob`."""

    prob: float

    @property
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 2

    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return two_qubit_depolarising_channel(self.prob)


class DepolarisingNoiseModel(NoiseModel):
    """Depolarising noise model.

    Only return the identity channel.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
        rng: Generator | None = None,
    ) -> None:
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        self.rng = ensure_rng(rng)

    def input_nodes(self, nodes: Iterable[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return [A(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[node]) for node in nodes]

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        if cmd.kind == CommandKind.N:
            return [cmd, A(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[cmd.node])]
        if cmd.kind == CommandKind.E:
            return [cmd, A(noise=TwoQubitDepolarisingNoise(self.entanglement_error_prob), nodes=list(cmd.nodes))]
        if cmd.kind == CommandKind.M:
            return [A(noise=DepolarisingNoise(self.measure_channel_prob), nodes=[cmd.node]), cmd]
        if cmd.kind == CommandKind.X:
            return [cmd, A(noise=DepolarisingNoise(self.x_error_prob), nodes=[cmd.node])]
        if cmd.kind == CommandKind.Z:
            return [cmd, A(noise=DepolarisingNoise(self.z_error_prob), nodes=[cmd.node])]
        # Use of `==` here for mypy
        if cmd.kind == CommandKind.C or cmd.kind == CommandKind.T or cmd.kind == CommandKind.A:  # noqa: PLR1714
            return [cmd]
        if cmd.kind == CommandKind.S:
            raise ValueError("Unexpected signal!")
        typing_extensions.assert_never(cmd.kind)

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        return result

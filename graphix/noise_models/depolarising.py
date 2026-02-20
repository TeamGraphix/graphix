"""Depolarising noise model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typing_extensions

from graphix.channels import KrausChannel, depolarising_channel, two_qubit_depolarising_channel
from graphix.command import BaseM, CommandKind
from graphix.measurements import toggle_outcome
from graphix.noise_models.noise_model import ApplyNoise, CommandOrNoise, Noise, NoiseModel
from graphix.rng import ensure_rng
from graphix.utils import Probability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.random import Generator

    from graphix.measurements import Outcome


class DepolarisingNoise(Noise):
    """One-qubit depolarising noise with probabibity ``prob``."""

    prob = Probability()

    def __init__(self, prob: float) -> None:
        """Initialize one-qubit depolarizing noise.

        Parameters
        ----------
        prob : float
            Probability parameter of the noise, between 0 and 1.
        """
        self.prob = prob

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 1

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return depolarising_channel(self.prob)


class TwoQubitDepolarisingNoise(Noise):
    """Two-qubits depolarising noise with probabibity ``prob``."""

    prob = Probability()

    def __init__(self, prob: float) -> None:
        """Initialize two-qubit depolarizing noise.

        Parameters
        ----------
        prob : float
            Probability parameter of the noise, between 0 and 1.
        """
        self.prob = prob

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 2

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return two_qubit_depolarising_channel(self.prob)


class DepolarisingNoiseModel(NoiseModel):
    """Depolarising noise model.

    :param NoiseModel: Parent abstract class class:`NoiseModel`
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

    @typing_extensions.override
    def input_nodes(self, nodes: Iterable[int], rng: Generator | None = None) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return [ApplyNoise(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[node]) for node in nodes]

    @typing_extensions.override
    def command(self, cmd: CommandOrNoise, rng: Generator | None = None) -> list[CommandOrNoise]:
        """Return the noise to apply to the command ``cmd``."""
        match cmd.kind:
            case CommandKind.N:
                return [cmd, ApplyNoise(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[cmd.node])]
            case CommandKind.E:
                return [
                    cmd,
                    ApplyNoise(noise=TwoQubitDepolarisingNoise(self.entanglement_error_prob), nodes=list(cmd.nodes)),
                ]
            case CommandKind.M:
                return [ApplyNoise(noise=DepolarisingNoise(self.measure_channel_prob), nodes=[cmd.node]), cmd]
            case CommandKind.X:
                return [
                    cmd,
                    ApplyNoise(noise=DepolarisingNoise(self.x_error_prob), nodes=[cmd.node], domain=cmd.domain),
                ]
            case CommandKind.Z:
                return [
                    cmd,
                    ApplyNoise(noise=DepolarisingNoise(self.z_error_prob), nodes=[cmd.node], domain=cmd.domain),
                ]
            case CommandKind.C | CommandKind.T | CommandKind.ApplyNoise:
                return [cmd]
            case CommandKind.S:
                raise ValueError("Unexpected signal!")
            case _:
                typing_extensions.assert_never(cmd.kind)

    @typing_extensions.override
    def confuse_result(self, cmd: BaseM, result: Outcome, rng: Generator | None = None) -> Outcome:
        """Assign wrong measurement result cmd = "M"."""
        if self.rng.uniform() < self.measure_error_prob:
            return toggle_outcome(result)
        return result

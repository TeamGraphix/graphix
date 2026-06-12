"""Amplitude damping noise model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typing_extensions

from graphix.channels import (
    KrausChannel,
    amplitude_damping_channel,
    two_qubit_amplitude_damping_channel,
)
from graphix.command import BaseM, CommandKind
from graphix.noise_models.noise_model import ApplyNoise, Noise, NoiseModel
from graphix.utils import Probability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.random import Generator

    from graphix.measurements import Outcome
    from graphix.noise_models.noise_model import CommandOrNoise


class AmplitudeDampingNoise(Noise):
    """One-qubit amplitude damping noise with damping parameter ``prob``."""

    prob = Probability()

    def __init__(self, prob: float) -> None:
        r"""Initialize one-qubit amplitude damping noise.

        Parameters
        ----------
        prob : float
            Damping parameter :math:`\\gamma` of the noise, between 0 and 1.
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
        return amplitude_damping_channel(self.prob)


class TwoQubitAmplitudeDampingNoise(Noise):
    """Two-qubit amplitude damping noise with damping parameter ``prob``."""

    prob = Probability()

    def __init__(self, prob: float) -> None:
        r"""Initialize two-qubit amplitude damping noise.

        Parameters
        ----------
        prob : float
            Damping parameter :math:`\\gamma` of the noise, between 0 and 1.
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
        return two_qubit_amplitude_damping_channel(self.prob)


class AmplitudeDampingNoiseModel(NoiseModel):
    r"""Amplitude damping noise model.

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
    ) -> None:
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_channel_prob = measure_channel_prob

    @typing_extensions.override
    def input_nodes(
        self, nodes: Iterable[int], rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return [ApplyNoise(noise=AmplitudeDampingNoise(self.prepare_error_prob), nodes=[node]) for node in nodes]

    @typing_extensions.override
    def command(
        self, cmd: CommandOrNoise, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to the command ``cmd``."""
        match cmd.kind:
            case CommandKind.N:
                return [cmd, ApplyNoise(noise=AmplitudeDampingNoise(self.prepare_error_prob), nodes=[cmd.node])]
            case CommandKind.E:
                return [
                    cmd,
                    ApplyNoise(
                        noise=TwoQubitAmplitudeDampingNoise(self.entanglement_error_prob), nodes=list(cmd.nodes)
                    ),
                ]
            case CommandKind.M:
                return [ApplyNoise(noise=AmplitudeDampingNoise(self.measure_channel_prob), nodes=[cmd.node]), cmd]
            case CommandKind.X:
                return [
                    cmd,
                    ApplyNoise(noise=AmplitudeDampingNoise(self.x_error_prob), nodes=[cmd.node], domain=cmd.domain),
                ]
            case CommandKind.Z:
                return [
                    cmd,
                    ApplyNoise(noise=AmplitudeDampingNoise(self.z_error_prob), nodes=[cmd.node], domain=cmd.domain),
                ]
            case CommandKind.C | CommandKind.T | CommandKind.ApplyNoise:
                return [cmd]
            case CommandKind.S:
                raise ValueError("Unexpected signal!")
            case _:
                typing_extensions.assert_never(cmd.kind)

    @typing_extensions.override
    def confuse_result(
        self, cmd: BaseM, result: Outcome, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> Outcome:
        r"""Return the measurement result unchanged.

        Amplitude damping is a purely quantum channel and introduces no
        classical readout error. Compose this model with another noise
        model (e.g. via :class:`ComposeNoiseModel`) to add readout error.
        """
        return result

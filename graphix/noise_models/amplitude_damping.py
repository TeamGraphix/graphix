"""Amplitude damping noise model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typing_extensions

from graphix.channels import KrausChannel, amplitude_damping_channel, two_qubit_amplitude_damping_channel
from graphix.command import BaseM, CommandKind
from graphix.measurements import toggle_outcome
from graphix.noise_models.noise_model import ApplyNoise, Noise, NoiseModel
from graphix.rng import ensure_rng
from graphix.utils import Probability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.random import Generator

    from graphix.measurements import Outcome
    from graphix.noise_models.noise_model import CommandOrNoise


class AmplitudeDampingNoise(Noise):
    """One-qubit amplitude damping noise with damping parameter ``gamma``.

    Parameters
    ----------
    gamma : float
        Damping parameter between 0 and 1. Represents the probability of
        energy loss (|1> -> |0> transition).
    """

    gamma = Probability()

    def __init__(self, gamma: float) -> None:
        """Initialize one-qubit amplitude damping noise.

        Parameters
        ----------
        gamma : float
            Damping parameter, between 0 and 1.
        """
        self.gamma = gamma

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """Return the number of qubits targeted by the noise element."""
        return 1

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return amplitude_damping_channel(self.gamma)


class TwoQubitAmplitudeDampingNoise(Noise):
    """Two-qubit amplitude damping noise with damping parameter ``gamma``.

    The channel is the tensor product of two independent single-qubit
    amplitude damping channels with the same parameter ``gamma``.

    Parameters
    ----------
    gamma : float
        Damping parameter between 0 and 1.
    """

    gamma = Probability()

    def __init__(self, gamma: float) -> None:
        """Initialize two-qubit amplitude damping noise.

        Parameters
        ----------
        gamma : float
            Damping parameter, between 0 and 1.
        """
        self.gamma = gamma

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """Return the number of qubits targeted by the noise element."""
        return 2

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return two_qubit_amplitude_damping_channel(self.gamma)


class AmplitudeDampingNoiseModel(NoiseModel):
    """Amplitude damping noise model.

    Applies amplitude damping noise after each command in a pattern.
    Each noise source can be tuned individually via its own ``gamma`` parameter.

    Parameters
    ----------
    prepare_error_gamma : float
        Damping applied after node preparation (N command), default 0.
    x_error_gamma : float
        Damping applied after X byproduct correction, default 0.
    z_error_gamma : float
        Damping applied after Z byproduct correction, default 0.
    entanglement_error_gamma : float
        Two-qubit damping applied after CZ entanglement (E command), default 0.
    measure_channel_gamma : float
        Damping applied to a qubit *before* measurement (M command), default 0.
    measure_error_prob : float
        Classical bit-flip probability on the measurement outcome, default 0.

    :param NoiseModel: Parent abstract class :class:`NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        prepare_error_gamma: float = 0.0,
        x_error_gamma: float = 0.0,
        z_error_gamma: float = 0.0,
        entanglement_error_gamma: float = 0.0,
        measure_channel_gamma: float = 0.0,
        measure_error_prob: float = 0.0,
    ) -> None:
        self.prepare_error_gamma = prepare_error_gamma
        self.x_error_gamma = x_error_gamma
        self.z_error_gamma = z_error_gamma
        self.entanglement_error_gamma = entanglement_error_gamma
        self.measure_channel_gamma = measure_channel_gamma
        self.measure_error_prob = measure_error_prob

    @typing_extensions.override
    def input_nodes(
        self, nodes: Iterable[int], rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return amplitude damping noise to apply to input nodes."""
        return [ApplyNoise(noise=AmplitudeDampingNoise(self.prepare_error_gamma), nodes=[node]) for node in nodes]

    @typing_extensions.override
    def command(
        self, cmd: CommandOrNoise, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to the command ``cmd``."""
        match cmd.kind:
            case CommandKind.N:
                return [cmd, ApplyNoise(noise=AmplitudeDampingNoise(self.prepare_error_gamma), nodes=[cmd.node])]
            case CommandKind.E:
                return [
                    cmd,
                    ApplyNoise(
                        noise=TwoQubitAmplitudeDampingNoise(self.entanglement_error_gamma),
                        nodes=list(cmd.nodes),
                    ),
                ]
            case CommandKind.M:
                return [ApplyNoise(noise=AmplitudeDampingNoise(self.measure_channel_gamma), nodes=[cmd.node]), cmd]
            case CommandKind.X:
                return [
                    cmd,
                    ApplyNoise(noise=AmplitudeDampingNoise(self.x_error_gamma), nodes=[cmd.node], domain=cmd.domain),
                ]
            case CommandKind.Z:
                return [
                    cmd,
                    ApplyNoise(noise=AmplitudeDampingNoise(self.z_error_gamma), nodes=[cmd.node], domain=cmd.domain),
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
        """Assign wrong measurement result cmd = \"M\"."""
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        if rng.uniform() < self.measure_error_prob:
            return toggle_outcome(result)
        return result

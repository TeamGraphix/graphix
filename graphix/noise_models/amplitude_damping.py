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
    """One-qubit amplitude damping noise with damping parameter ``gamma``."""

    gamma = Probability()

    def __init__(self, gamma: float) -> None:
        """Initialize one-qubit amplitude damping noise.

        Parameters
        ----------
        gamma : float
            Damping parameter of the noise, between 0 and 1.
        """
        self.gamma = gamma

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 1

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return amplitude_damping_channel(self.gamma)


class TwoQubitAmplitudeDampingNoise(Noise):
    """Two-qubit amplitude damping noise with damping parameter ``gamma``."""

    gamma = Probability()

    def __init__(self, gamma: float) -> None:
        """Initialize two-qubit amplitude damping noise.

        Parameters
        ----------
        gamma : float
            Damping parameter of the noise, between 0 and 1.
        """
        self.gamma = gamma

    @property
    @typing_extensions.override
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 2

    @typing_extensions.override
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        return two_qubit_amplitude_damping_channel(self.gamma)


class AmplitudeDampingNoiseModel(NoiseModel):
    """Amplitude damping noise model.

    Mirrors the structure of
    :class:`graphix.noise_models.depolarising.DepolarisingNoiseModel`, applying an
    amplitude damping channel at each step of a pattern. Channel parameters are
    named with a ``_gamma`` suffix to reflect that they are damping parameters.

    Parameters
    ----------
    prepare_error_gamma : float
        Damping applied to each freshly prepared node.
    x_error_gamma : float
        Damping applied after an ``X`` correction (conditioned on its domain).
    z_error_gamma : float
        Damping applied after a ``Z`` correction (conditioned on its domain).
    entanglement_error_gamma : float
        Two-qubit damping applied after an entangling ``E`` command.
    measure_channel_gamma : float
        Damping applied to a node immediately before it is measured.
    measure_error_prob : float
        Probability of reporting a flipped (classical) measurement outcome.
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
        """Return the noise to apply to input nodes."""
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
                        noise=TwoQubitAmplitudeDampingNoise(self.entanglement_error_gamma), nodes=list(cmd.nodes)
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
        """Assign a possibly flipped measurement result for cmd = "M"."""
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        if rng.uniform() < self.measure_error_prob:
            return toggle_outcome(result)
        return result

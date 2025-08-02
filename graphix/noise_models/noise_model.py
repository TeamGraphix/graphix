"""Abstract interface for noise models.

This module defines :class:`NoiseModel`, the base class used by
:class:`graphix.simulator.PatternSimulator` when running noisy
simulations. Child classes implement concrete noise processes by
overriding the abstract methods defined here.
"""

from __future__ import annotations

import dataclasses
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

# override introduced in Python 3.12
from typing_extensions import override

from graphix.command import BaseM, Command, CommandKind, Node, _KindChecker

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.channels import KrausChannel
    from graphix.measurements import Outcome


class Noise(ABC):
    """Abstract base class for noise."""

    @property
    @abstractmethod
    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise."""

    @abstractmethod
    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise."""


@dataclass
class A(_KindChecker):
    """Apply noise command."""

    kind: ClassVar[Literal[CommandKind.A]] = dataclasses.field(default=CommandKind.A, init=False)
    noise: Noise
    nodes: list[Node]


if sys.version_info >= (3, 10):
    CommandOrNoise = Command | A
else:
    from typing import Union

    CommandOrNoise = Union[Command, A]


class NoiseModel(ABC):
    """Abstract base class for all noise models."""

    @abstractmethod
    def input_nodes(self, nodes: Iterable[int]) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""

    @abstractmethod
    def command(self, cmd: CommandOrNoise) -> list[CommandOrNoise]:
        """Return the noise to apply to the command `cmd`."""

    @abstractmethod
    def confuse_result(self, cmd: BaseM, result: Outcome) -> Outcome:
        """Return a possibly flipped measurement outcome.

        Parameters
        ----------
        result : Outcome
            Ideal measurement result.

        cmd : BaseM
            The measurement command that produced the given outcome.

        Returns
        -------
        Outcome
            Possibly corrupted result.
        """

    def transpile(self, sequence: Iterable[CommandOrNoise]) -> list[CommandOrNoise]:
        """Apply the noise to a sequence of commands and return the resulting sequence."""
        return [n_cmd for cmd in sequence for n_cmd in self.command(cmd)]


@dataclass(frozen=True)
class ComposeNoiseModel(NoiseModel):
    """Compose noise models."""

    models: list[NoiseModel]

    @override
    def input_nodes(self, nodes: Iterable[int]) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return [n_cmd for m in self.models for n_cmd in m.input_nodes(nodes)]

    @override
    def command(self, cmd: CommandOrNoise) -> list[CommandOrNoise]:
        """Return the noise to apply to the command `cmd`."""
        sequence = [cmd]
        for model in self.models:
            sequence = model.transpile(sequence)
        return sequence

    @override
    def confuse_result(self, cmd: BaseM, result: Outcome) -> Outcome:
        """Assign wrong measurement result."""
        for m in self.models:
            result = m.confuse_result(cmd, result)
        return result

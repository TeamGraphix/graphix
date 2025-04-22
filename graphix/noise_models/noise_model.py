"""Abstract base class for all noise models."""

from __future__ import annotations

import dataclasses
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from graphix.command import BaseM, Command, CommandKind, Node, _KindChecker

if TYPE_CHECKING:
    from graphix.channels import KrausChannel


class Noise(ABC):
    """Abstract base class for noise."""

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


NoiseCommands = list[CommandOrNoise]


class NoiseModel(ABC):
    """Abstract base class for all noise models."""

    @abstractmethod
    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""

    @abstractmethod
    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""

    @abstractmethod
    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result."""

    def transpile(self, sequence: NoiseCommands) -> NoiseCommands:
        """Apply the noise to a sequence of commands and return the resulting sequence."""
        return [n_cmd for cmd in sequence for n_cmd in self.command(cmd)]


@dataclass(frozen=True)
class ComposeNoiseModel(NoiseModel):
    """Compose noise models."""

    models: list[NoiseModel]

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return [n_cmd for m in self.models for n_cmd in m.input_nodes(nodes)]

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        sequence = [cmd]
        for model in self.models:
            sequence = model.transpile(sequence)
        return sequence

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result."""
        for m in self.l:
            result = m.confuse_result(cmd, result)
        return result

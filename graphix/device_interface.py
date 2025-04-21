"""Abstract base class for quantum device backends and jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from graphix.pattern import Pattern


class Job(ABC):
    """Abstract base class representing a quantum job handle."""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Check whether the job has completed.

        Returns
        -------
        bool
            True if the job is done, False otherwise.
        """

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the job."""

    @abstractmethod
    def retrieve_result(self) -> dict[str, int]:
        """Retrieve the result from a completed job.

        Returns
        -------
        Dict[str, int]
            Result of the job execution.
        """


class CompileOptions(ABC):
    """Abstract base class for specifying compilation options.

    To be extended by concrete implementations.
    """

    def __init__(self) -> None:
        """Define a dummy abstract method to satisfy ABC requirements."""


class DeviceBackend(ABC):
    """Abstract base class representing a quantum device backend (hardware or simulator)."""

    VALID_MODES: ClassVar[frozenset[str]] = frozenset({"hardware", "simulator"})

    def __init__(self) -> None:
        """Initialize the backend."""

    @abstractmethod
    def compile(self, options: CompileOptions | None = None) -> None:
        """Compile the pattern using given compile options.

        Parameters
        ----------
        options : CompileOptions, optional
            Options for compilation.
        """

    @abstractmethod
    def submit_job(self, shots: int = 1024) -> Job:
        """Submit a compiled job to the backend.

        Parameters
        ----------
        shots : int, optional
            Number of shots/samples to execute. Defaults to 1024.

        Returns
        -------
        Job
            Monitor or retrieve job result.
        """

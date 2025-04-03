from abc import ABC, abstractmethod
from typing import Dict, Optional

from graphix.pattern import Pattern


class JobHandler(ABC):
    """Abstract base class representing a quantum job handle."""

    @abstractmethod
    def get_id(self) -> str:
        """Return the unique ID of the job.

        Returns
        -------
        str
            Unique job ID.
        """

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


class CompileOptions(ABC):
    """Abstract base class for specifying compilation options.

    To be extended by concrete implementations.
    """


class DeviceBackend(ABC):
    """Abstract base class representing a quantum device backend (hardware or simulator)."""

    VALID_MODES: set[str] = {"hardware", "simulator"}

    def __init__(self) -> None:
        """Initialize the backend with no assigned pattern."""
        self.pattern: Optional[Pattern] = None

    def set_pattern(self, pattern: Pattern) -> None:
        """Assign a pattern to be compiled and executed on the backend.

        Parameters
        ----------
        pattern : Pattern
            The pattern to assign.
        """
        self.pattern = pattern

    @abstractmethod
    def compile(self, options: Optional[CompileOptions] = None) -> None:
        """Compile the pattern using given compile options.

        Parameters
        ----------
        options : CompileOptions, optional
            Options for compilation.
        """

    @abstractmethod
    def submit_job(self, shots: int = 1024) -> JobHandler:
        """Submit a compiled job to the backend.

        Parameters
        ----------
        shots : int, optional
            Number of shots/samples to execute. Defaults to 1024.

        Returns
        -------
        JobHandler
            Handle to monitor or retrieve job result.
        """

    @abstractmethod
    def retrieve_result(self, job_handle: JobHandler) -> Dict[str, int]:
        """Retrieve the result from a completed job.

        Parameters
        ----------
        job_handle : JobHandler
            The handle of the submitted job.

        Returns
        -------
        Any
            Result of the job execution.
        """

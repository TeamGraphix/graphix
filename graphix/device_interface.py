from abc import ABC, abstractmethod
from typing import Any, Optional
from graphix.pattern import Pattern


class JobHandle(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def cancel(self) -> None:
        pass


class CompileOptions(ABC):
    """Base class for compile options."""
    pass


class DeviceBackend(ABC):
    def __init__(self) -> None:
        self.pattern: Optional[Pattern] = None

    def set_pattern(self, pattern: Pattern) -> None:
        self.pattern = pattern

    @abstractmethod
    def compile(self, options: Optional[CompileOptions] = None) -> None:
        pass

    @abstractmethod
    def submit_job(self, shots: int) -> JobHandle:
        pass

    @abstractmethod
    def retrieve_result(self, job_handle: JobHandle) -> Any:
        pass

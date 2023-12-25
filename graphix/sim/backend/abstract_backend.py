from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Sequence

Tensor = Any


class AbstractBackend:
    """An abstract backend class for all backends to inherit from."""

    @property
    @abstractmethod
    def pi(self) -> float:
        """Return the value of pi."""
        raise NotImplementedError

    @abstractmethod
    def sin(self, x: Tensor) -> Tensor:
        """Return the elementwise sine of an array."""
        raise NotImplementedError

    @abstractmethod
    def cos(self, x: Tensor) -> Tensor:
        """Return the elementwise cosine of an array."""
        raise NotImplementedError

    @abstractmethod
    def sum(self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False) -> Tensor:
        """Return the sum of an array."""
        raise NotImplementedError

    @abstractmethod
    def mod(self, a: Tensor, b: Tensor) -> Tensor:
        """Return the elementwise modulus of two arrays."""
        raise NotImplementedError

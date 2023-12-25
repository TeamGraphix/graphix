from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Sequence

Array = Any


class AbstractBackend:
    """An abstract backend class for all backends to inherit from."""

    @abstractmethod
    def sin(self, x: Array) -> Array:
        """Return the elementwise sine of an array."""
        raise NotImplementedError

    @abstractmethod
    def cos(self, x: Array) -> Array:
        """Return the elementwise cosine of an array."""
        raise NotImplementedError

    @abstractmethod
    def sum(self, a: Array, axis: Optional[Sequence[int]] = None, keepdims: bool = False) -> Array:
        """Return the sum of an array."""
        raise NotImplementedError

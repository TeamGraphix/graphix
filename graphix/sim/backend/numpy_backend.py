from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .abstract_backend import AbstractBackend

Tensor = Any


class NumPyBackend(AbstractBackend):
    """A backend that uses NumPy for its computations."""

    def eye(self, N: int, dtype: Optional[str] = None, M: Optional[int] = None) -> Tensor:
        return np.eye(N, M=M, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        return np.ones(shape, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        return np.zeros(shape, dtype=dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.copy()

    def abs(self, a: Tensor) -> Tensor:
        return np.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return np.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return np.cos(a)

    def acos(self, a: Tensor) -> Tensor:
        return np.arccos(a)

    def acosh(self, a: Tensor) -> Tensor:
        return np.arccosh(a)

    def asin(self, a: Tensor) -> Tensor:
        return np.arcsin(a)

    def asinh(self, a: Tensor) -> Tensor:
        return np.arcsinh(a)

    def atan(self, a: Tensor) -> Tensor:
        return np.arctan(a)

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return np.arctan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return np.arctanh(a)

    def cosh(self, a: Tensor) -> Tensor:
        return np.cosh(a)

    def tan(self, a: Tensor) -> Tensor:
        return np.tan(a)

    def tanh(self, a: Tensor) -> Tensor:
        return np.tanh(a)

    def sinh(self, a: Tensor) -> Tensor:
        return np.sinh(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size

    def eigvalsh(self, a: Tensor) -> Tensor:
        return np.linalg.eigvalsh(a)

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return np.kron(a, b)

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__()

    def numpy(self, a: Tensor) -> Tensor:
        return a

    def det(self, a: Tensor) -> Tensor:
        return np.linalg.det(a)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def std(self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False) -> Tensor:
        return np.std(a, axis=axis, keepdims=keepdims)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return np.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return np.argmin(a, axis=axis)

    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.cumsum(a, axis)

    def real(self, a: Tensor) -> Tensor:
        return np.real(a)

    def imag(self, a: Tensor) -> Tensor:
        return np.imag(a)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return np.arange(start=0, stop=start, step=step)
        return np.arange(start=start, stop=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return np.mod(x, y)

    def set_random_state(self, seed: Optional[int] = None, get_only: bool = False) -> Any:
        g = np.random.default_rng(seed)
        if get_only is False:
            self.g = g
        return g

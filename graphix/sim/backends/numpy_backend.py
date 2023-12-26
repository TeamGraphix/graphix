from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .abstract_backend import AbstractBackend

default_dtype: str
Tensor = Any


class NumPyBackend(AbstractBackend):
    """A backend that uses NumPy for its computations."""

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def pi(self) -> float:
        return np.pi

    def eye(self, N: int, dtype: Optional[str] = None, M: Optional[int] = None) -> Tensor:
        if dtype is None:
            dtype = default_dtype
        return np.eye(N, M=M, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = default_dtype
        return np.ones(shape, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = default_dtype
        return np.zeros(shape, dtype=dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.copy()

    def abs(self, a: Tensor) -> Tensor:
        return np.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return np.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return np.cos(a)

    def tan(self, a: Tensor) -> Tensor:
        return np.tan(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return np.kron(a, b)

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__()

    def tensordot(self, a: Tensor, b: Tensor, axes: Union[int, Tuple[Sequence[int], Sequence[int]]]) -> Tensor:
        return np.tensordot(a, b, axes=axes)

    def moveaxis(self, a: Tensor, source: Sequence[int], destination: Sequence[int]) -> Tensor:
        return np.moveaxis(a, source, destination)

    def reshape(self, a: Tensor, shape: Sequence[int]) -> Tensor:
        return np.reshape(a, shape)

    def eig(self, a: Tensor) -> Tensor:
        return np.linalg.eig(a)

    def eigh(self, a: Tensor) -> Tensor:
        return np.linalg.eigh(a)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return np.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return np.argmin(a, axis=axis)

    def real(self, a: Tensor) -> Tensor:
        return np.real(a)

    def imag(self, a: Tensor) -> Tensor:
        return np.imag(a)

    def dot(self, a: Tensor, b: Tensor) -> Tensor:
        return np.dot(a, b)

    def sqrt(self, a: Tensor) -> Tensor:
        return np.sqrt(a)

    def sum(self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False) -> Tensor:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return np.arange(start=0, stop=start, step=step)
        return np.arange(start=start, stop=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return np.mod(x, y)

    def set_random_state(self, seed: Optional[int] = None, get_only: bool = False) -> Any:
        random_state = np.random.default_rng(seed)
        if get_only is False:
            self.random_state = random_state
        return random_state

    def jit(self, func: Callable[..., Tensor]) -> Callable[..., Tensor]:
        return func

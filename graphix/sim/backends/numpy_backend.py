from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import Generator

from graphix.sim.backends.abstract_backend import Tensor

from .abstract_backend import AbstractBackend
from .settings import default_dtype

Tensor = Any


class NumPyRandomState:
    def __init__(self, rng: Generator):
        self.rng = np.random.default_rng(rng)


class NumPyBackend(AbstractBackend):
    """A backend that uses NumPy for its computations."""

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def pi(self) -> float:
        return np.pi

    @property
    def nan(self) -> float:
        return np.nan

    def array(self, a: Any, dtype: Optional[str] = None) -> Tensor:
        """Create an array."""
        return np.array(a, dtype=dtype)

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

    def isclose(
        self, a: Tensor, b: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
    ) -> Tensor:
        return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def equal(self, a: Tensor, b: Tensor) -> Tensor:
        return np.equal(a, b)

    def where(self, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        return np.where(condition, x, y)

    def any(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return np.any(a, axis=axis)

    def set_random_state(self, seed: Optional[int] = None, get_only: bool = False) -> Any:
        random_state = np.random.default_rng(seed)
        if get_only is False:
            self.random_state = NumPyRandomState(random_state)
        return NumPyRandomState(random_state)

    def random_choice(
        self,
        a: Tensor,
        p: Optional[Tensor] = None,
        random_state: Optional[NumPyRandomState] = None,
    ) -> Tensor:
        if random_state is None and hasattr(self, "random_state") is False:
            return np.random.choice(a, p=p)
        elif random_state is None and hasattr(self, "random_state"):
            return self.random_state.rng.choice(a, p=p)
        if not isinstance(random_state, NumPyRandomState):
            raise TypeError("random_state must be of type NumPyRandomState")
        return random_state.rng.choice(a, p=p)

    def jit(
        self,
        func: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
    ) -> Callable[..., Any]:
        return func

    def cond(self, pred: bool, true_fn: Callable[..., Any], false_fn: Callable[..., Any]) -> Callable[..., Any]:
        return true_fn() if pred else false_fn()

    def fori_loop(
        self, lower: int, upper: int, body_fun: Callable[..., Any], init_val: Any, *args: Any, **kwargs: Any
    ) -> Any:
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val

    def set_element(self, a: Tensor, index: int, value: Any) -> None:
        a[index] = value

    def logical_and(self, a: Tensor, b: Tensor) -> Tensor:
        return np.logical_and(a, b)

    def logical_or(self, a: Tensor, b: Tensor) -> Tensor:
        return np.logical_or(a, b)

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

from .abstract_backend import AbstractBackend

default_dtype: str
Tensor = Any

jax = None
jnp = None
jsp = None


class JaxBackend(AbstractBackend):
    """A backend that uses Jax for its computations."""

    def __init__(self) -> None:
        global jax  # jax package
        global jnp  # jax.numpy module
        global jsp  # jax.scipy module
        try:
            import jax
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Jax is not installed. See https://jax.readthedocs.io/en/latest/installation.html for installation instructions."
            )

        jnp = jax.numpy
        jsp = jax.scipy

    @property
    def name(self) -> str:
        return "jax"

    @property
    def pi(self) -> float:
        return jnp.pi

    def eye(self, N: int, dtype: Optional[str] = None, M: Optional[int] = None) -> Tensor:
        if dtype is None:
            dtype = default_dtype
        return jnp.eye(N, M=M, dtype=dtype)

    def ones(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = default_dtype
        return jnp.ones(shape, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = default_dtype
        return jnp.zeros(shape, dtype=dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.copy()

    def abs(self, a: Tensor) -> Tensor:
        return jnp.abs(a)

    def sin(self, a: Tensor) -> Tensor:
        return jnp.sin(a)

    def cos(self, a: Tensor) -> Tensor:
        return jnp.cos(a)

    def tan(self, a: Tensor) -> Tensor:
        return jnp.tan(a)

    def size(self, a: Tensor) -> Tensor:
        return a.size

    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        return jnp.kron(a, b)

    def dtype(self, a: Tensor) -> str:
        return a.dtype.__str__()

    def tensordot(self, a: Tensor, b: Tensor, axes: Union[int, Tuple[Sequence[int], Sequence[int]]]) -> Tensor:
        return jnp.tensordot(a, b, axes=axes)

    def moveaxis(self, a: Tensor, source: Sequence[int], destination: Sequence[int]) -> Tensor:
        return jnp.moveaxis(a, source, destination)

    def reshape(self, a: Tensor, shape: Sequence[int]) -> Tensor:
        return jnp.reshape(a, shape)

    def eig(self, a: Tensor) -> Tensor:
        return jnp.linalg.eig(a)

    def eigh(self, a: Tensor) -> Tensor:
        return jnp.linalg.eigh(a)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        return jnp.mean(a, axis=axis, keepdims=keepdims)

    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.min(a, axis=axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.max(a, axis=axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return jnp.argmax(a, axis=axis)

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return jnp.argmin(a, axis=axis)

    def real(self, a: Tensor) -> Tensor:
        return jnp.real(a)

    def imag(self, a: Tensor) -> Tensor:
        return jnp.imag(a)

    def dot(self, a: Tensor, b: Tensor) -> Tensor:
        return jnp.dot(a, b)

    def sqrt(self, a: Tensor) -> Tensor:
        return jnp.sqrt(a)

    def sum(self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False) -> Tensor:
        return jnp.sum(a, axis=axis, keepdims=keepdims)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        if stop is None:
            return jnp.arange(start=0, stop=start, step=step)
        return jnp.arange(start=start, stop=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return jnp.mod(x, y)

    def isclose(
        self, a: Tensor, b: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
    ) -> Tensor:
        return jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def set_random_state(self, seed: Optional[int] = None, get_only: bool = False) -> Any:
        if seed is None:
            random_state = jax.random.PRNGKey(42)
        else:
            random_state = jax.random.PRNGKey(seed)
        if get_only is False:
            self.random_state = random_state
        return random_state

    def jit(
        self,
        func: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
    ) -> Callable[..., Any]:
        return jax.jit(func, static_argnums=static_argnums)

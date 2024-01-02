from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

from .abstract_backend import AbstractBackend
from .settings import default_dtype

Tensor = Any
Error = Any

jax = None
jnp = None
jsp = None
checkify = None
PRNGKeyArray = None


class JaxRandomState:
    def __init__(self, prngkey: PRNGKeyArray):
        self.prngkey = prngkey


class JaxBackend(AbstractBackend):
    """A backend that uses Jax for its computations."""

    def __init__(self) -> None:
        global jax  # jax package
        global jnp  # jax.numpy module
        global jsp  # jax.scipy module
        global checkify  # jax.experimental.checkify module
        global PRNGKeyArray  # jax.random.PRNGKeyArray class
        try:
            import jax
            from jax.experimental import checkify
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Jax is not installed. See https://jax.readthedocs.io/en/latest/installation.html for installation instructions."
            )

        jnp = jax.numpy
        jsp = jax.scipy
        PRNGKeyArray = jax.random.PRNGKeyArray

    @property
    def name(self) -> str:
        return "jax"

    @property
    def pi(self) -> float:
        return jnp.pi

    @property
    def nan(self) -> float:
        return jnp.nan

    def array(self, a: Any, dtype: Optional[str] = None) -> Tensor:
        """Create an array."""
        return jnp.array(a, dtype=dtype)

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

    def mod(self, x: Tensor, y: Tensor, dtype: Optional[str] = None) -> Tensor:
        return jnp.mod(x, y)

    def isclose(
        self, a: Tensor, b: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
    ) -> Tensor:
        return jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def equal(self, a: Tensor, b: Tensor) -> Tensor:
        return jnp.equal(a, b)

    def where(self, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        return jnp.where(condition, x, y)

    def any(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return jnp.any(a, axis=axis)

    def set_random_state(self, seed: Optional[int] = None, get_only: bool = False) -> Any:
        if seed is None:
            random_state = jax.random.PRNGKey(42)
        else:
            random_state = jax.random.PRNGKey(seed)
        if get_only is False:
            self.random_state = JaxRandomState(random_state)
        return JaxRandomState(random_state)

    def random_choice(
        self,
        a: Tensor,
        p: Optional[Tensor] = None,
        random_state: Optional[JaxRandomState] = None,
    ) -> Tensor:
        if random_state is None and self.random_state is None:
            random_state = self.set_random_state(get_only=True)
            return jax.random.choice(a=a, p=p, key=random_state.prngkey)
        elif random_state is None and self.random_state is not None:
            return jax.random.choice(a=a, p=p, key=self.random_state.prngkey)
        if not isinstance(random_state, JaxRandomState):
            raise TypeError("random_state must be of type JaxRandomState.")
        return jax.random.choice(a=a, p=p, key=random_state.prngkey)

    def jit(
        self,
        func: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
    ) -> Callable[..., Any]:
        return jax.jit(func, static_argnums=static_argnums)

    def cond(self, pred: bool, true_fn: Callable[..., Any], false_fn: Callable[..., Any]) -> Callable[..., Any]:
        return jax.lax.cond(pred, true_fn, false_fn)

    def fori_loop(
        self, lower: int, upper: int, body_fun: Callable[..., Any], init_val: Any, *args: Any, **kwargs: Any
    ) -> Any:
        return jax.lax.fori_loop(lower, upper, body_fun, init_val, *args, **kwargs)

    def set_element(self, a: Tensor, index: int, value: Any) -> None:
        a.at[index].set(value)

    def wrap_by_checkify(self, func: Callable[..., Any]) -> Callable[..., tuple[Error, Any]]:
        return checkify.checkify(func)

    def debug_assert_true(self, condition: bool, message: str, **kwargs) -> None:
        return checkify.check(condition, message, **kwargs)

    def logical_and(self, a: Tensor, b: Tensor) -> Tensor:
        return jnp.logical_and(a, b)

    def logical_or(self, a: Tensor, b: Tensor) -> Tensor:
        return jnp.logical_or(a, b)

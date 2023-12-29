from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple, Union

Tensor = Any
AbstractRandomState = Any


class AbstractBackend(ABC):
    """An abstract backend class for all backends to inherit from."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pi(self) -> float:
        """Return the value of pi."""
        raise NotImplementedError

    @abstractmethod
    def array(self, a: Any, dtype: Optional[str] = None) -> Tensor:
        """Create an array."""
        raise NotImplementedError

    @abstractmethod
    def eye(self, N: int, dtype: Optional[str] = None, M: Optional[int] = None) -> Tensor:
        """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
        raise NotImplementedError

    @abstractmethod
    def ones(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        """Return a new array of given shape and type, filled with ones."""
        raise NotImplementedError

    @abstractmethod
    def zeros(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
        """Return a new array of given shape and type, filled with zeros."""
        raise NotImplementedError

    @abstractmethod
    def copy(self, a: Tensor) -> Tensor:
        """Return a copy of the array."""
        raise NotImplementedError

    @abstractmethod
    def abs(self, a: Tensor) -> Tensor:
        """Return the absolute value elementwise."""
        raise NotImplementedError

    @abstractmethod
    def sin(self, a: Tensor) -> Tensor:
        """Return the elementwise sine of an array."""
        raise NotImplementedError

    @abstractmethod
    def cos(self, a: Tensor) -> Tensor:
        """Return the elementwise cosine of an array."""
        raise NotImplementedError

    @abstractmethod
    def tan(self, a: Tensor) -> Tensor:
        """Return the elementwise tangent of an array."""
        raise NotImplementedError

    @abstractmethod
    def size(self, a: Tensor) -> Tensor:
        """Return the number of elements in the array."""
        raise NotImplementedError

    @abstractmethod
    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        """Return the Kronecker product of two arrays."""
        raise NotImplementedError

    @abstractmethod
    def dtype(self, a: Tensor) -> str:
        """Return the data type of the array."""
        raise NotImplementedError

    @abstractmethod
    def tensordot(self, a: Tensor, b: Tensor, axes: Union[int, Tuple[Sequence[int], Sequence[int]]]) -> Tensor:
        """Return the tensor dot product along specified axes."""
        raise NotImplementedError

    @abstractmethod
    def moveaxis(self, a: Tensor, source: Sequence[int], destination: Sequence[int]) -> Tensor:
        """Move axes of an array to new positions."""
        raise NotImplementedError

    @abstractmethod
    def reshape(self, a: Tensor, shape: Sequence[int]) -> Tensor:
        """Return the reshaped array."""
        raise NotImplementedError

    @abstractmethod
    def eig(self, a: Tensor) -> Tensor:
        """Return the eigenvalues and right eigenvectors of a square array."""
        raise NotImplementedError

    @abstractmethod
    def eigh(self, a: Tensor) -> Tensor:
        """Return the eigenvalues and right eigenvectors of a Hermitian or symmetric matrix."""
        raise NotImplementedError

    @abstractmethod
    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Return the arithmetic mean of an array."""
        raise NotImplementedError

    @abstractmethod
    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        """Return the minimum value of an array."""
        raise NotImplementedError

    @abstractmethod
    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        """Return the maximum value of an array."""
        raise NotImplementedError

    @abstractmethod
    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        """Return the indices of the maximum values along an axis."""
        raise NotImplementedError

    @abstractmethod
    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        """Return the indices of the minimum values along an axis."""
        raise NotImplementedError

    @abstractmethod
    def real(self, a: Tensor) -> Tensor:
        """Return the real part of the complex argument."""
        raise NotImplementedError

    @abstractmethod
    def imag(self, a: Tensor) -> Tensor:
        """Return the imaginary part of the complex argument."""
        raise NotImplementedError

    @abstractmethod
    def dot(self, a: Tensor, b: Tensor) -> Tensor:
        """Return the dot product of two arrays."""
        raise NotImplementedError

    @abstractmethod
    def sqrt(self, a: Tensor) -> Tensor:
        """Return the non-negative square-root of an array, element-wise."""
        raise NotImplementedError

    @abstractmethod
    def sum(self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False) -> Tensor:
        """Return the sum of an array."""
        raise NotImplementedError

    @abstractmethod
    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        """Return evenly spaced values within a given interval."""
        raise NotImplementedError

    @abstractmethod
    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        """Return element-wise remainder of division."""
        raise NotImplementedError

    @abstractmethod
    def isclose(
        self, a: Tensor, b: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
    ) -> Tensor:
        """Return element-wise True if two arrays are element-wise equal within a tolerance."""
        raise NotImplementedError

    @abstractmethod
    def set_random_state(self, seed: Optional[int] = None, get_only: bool = False) -> Any:
        """Set the random state for the backend.

        Parameters
        ----------
        seed : Optional[int]
            The seed to use for the random state.
        get_only : bool
            If True, only return the random state. If False, set the random state globally.
            Default is False.

        Returns
        -------
        Any
        """
        raise NotImplementedError

    @abstractmethod
    def random_choice(
        self,
        a: Tensor,
        p: Optional[Tensor] = None,
        random_state: Optional[AbstractRandomState] = None,
    ) -> Tensor:
        """Generates a random sample from a given 1-D array.

        Parameters
        ----------
        a : Tensor
            1-D array-like or int.
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if a were np.arange(a)
        p : Optional[Tensor]
            The probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribution over all entries in a.
        random_state : Optional[AbstractRandomState]
            The random state to use for sampling. Default is None.

        Returns
        -------
        Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def jit(
        self,
        func: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
    ) -> Callable[..., Any]:
        """Return a function that is JIT compiled."""
        raise NotImplementedError

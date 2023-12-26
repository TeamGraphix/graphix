from __future__ import annotations

from typing import Union

from . import jax_backend, numpy_backend
from .abstract_backend import AbstractBackend

_BACKENDS = {
    "numpy": numpy_backend.NumPyBackend,
    "jax": jax_backend.JaxBackend,
}


def get_backend(backend: Union[str, AbstractBackend]) -> AbstractBackend:
    if isinstance(backend, AbstractBackend):
        return backend
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    return _BACKENDS[backend]()

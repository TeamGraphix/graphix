from __future__ import annotations

import importlib
from typing import Union

from . import jax_backend, numpy_backend
from .abstract_backend import AbstractBackend

_BACKENDS = {
    "numpy": numpy_backend.NumPyBackend,
    "jax": jax_backend.JaxBackend,
}

backend: AbstractBackend


def get_backend(backend: Union[str, AbstractBackend]) -> AbstractBackend:
    if isinstance(backend, AbstractBackend):
        return backend
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    return _BACKENDS[backend]()


def set_backend(backend_name: str = "numpy") -> AbstractBackend:
    """Set the backend to use for all computations."""
    global backend
    backend = get_backend(backend_name)
    # Class method decorators are evaluated at the time the class is defined,
    # not at the time the class is instantiated, so we need to reload the
    # statevec module to update the backend.
    from graphix import simulator
    from graphix.sim import statevec

    importlib.reload(simulator)
    importlib.reload(statevec)

    return backend


set_backend()

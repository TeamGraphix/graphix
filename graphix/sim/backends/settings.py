from __future__ import annotations

import sys

from .abstract_backend import AbstractBackend
from .backend_factory import get_backend

module_name = "graphix.sim"
backend: AbstractBackend
default_dtype = "complex128"


def set_backend(backend_name: str = "numpy") -> None:
    """Set the backend to use for all computations."""
    backend = get_backend(backend_name)
    for module in sys.modules:
        if module.startswith(module_name):
            setattr(sys.modules[module], "backend", backend)


set_backend()


def set_default_dtype(dtype: str = "complex128") -> None:
    """Set the default dtype for all computations."""
    for module in sys.modules:
        if module.startswith(module_name):
            setattr(sys.modules[module], "default_dtype", dtype)

    try:
        from jax import config
    except ModuleNotFoundError:
        config = None

    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
    if config is not None:
        if dtype == "complex128":
            config.update("jax_enable_x64", True)
        elif dtype == "complex64":
            config.update("jax_enable_x64", False)


set_default_dtype()

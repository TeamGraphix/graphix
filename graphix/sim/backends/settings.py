from __future__ import annotations

default_dtype: str


def set_default_dtype(dtype: str = "complex128") -> None:
    """Set the default dtype for all computations."""
    global default_dtype
    default_dtype = dtype

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

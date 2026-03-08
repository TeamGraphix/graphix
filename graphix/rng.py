"""Provide a default random-number generator if `None` is given."""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

_rng_local = threading.local()


def ensure_rng(rng: Generator | None = None, *, stacklevel: int = 1) -> Generator:
    """Return a default random-number generator if `None` is given.

    Parameters
    ----------
    rng: Generator | None, optional
        If set and not `None`, this value is returned.
        If `None` (the default), the default random-number generator is returned.
    stacklevel : int, optional
        Stack level to use for warnings. Defaults to 1, meaning that warnings
        are reported at this function's call site.

    Notes
    -----
    A warning is issued if the default random-number generator is used.
    """
    if rng is not None:
        return rng
    warnings.warn("Default random-number generator is used. Results are not reproducible.", stacklevel=stacklevel + 1)
    stored: Generator | None = getattr(_rng_local, "rng", None)
    if stored is not None:
        return stored
    rng = np.random.default_rng()
    # MEMO: Cannot perform type check
    setattr(_rng_local, "rng", rng)  # noqa: B010
    return rng

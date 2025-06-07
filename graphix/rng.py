"""Provide a default random-number generator if `None` is given."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

_rng_local = threading.local()


def ensure_rng(rng: Generator | None = None) -> Generator:
    """Return a default random-number generator if `None` is given."""
    if rng is not None:
        return rng
    stored: Generator | None = getattr(_rng_local, "rng", None)
    if stored is not None:
        return stored
    rng = np.random.default_rng()
    # MEMO: Cannot perform type check
    setattr(_rng_local, "rng", rng)  # noqa: B010
    return rng

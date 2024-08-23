from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

rng_local = threading.local()


def ensure_rng(rng: Generator | None = None) -> Generator:
    if rng is not None:
        return rng
    if rng := getattr(rng_local, "rng", None):
        return rng
    rng = np.random.default_rng()
    rng_local.rng = rng
    return rng

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


def ensure_rng(rng: Generator | None = None) -> Generator:
    if rng is not None:
        return rng
    return np.random.default_rng()

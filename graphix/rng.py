import numpy as np
from numpy.random import Generator


def ensure_rng(rng: Generator | None = None) -> Generator:
    if rng is not None:
        return rng
    return np.random.default_rng()

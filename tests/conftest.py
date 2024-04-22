import numpy as np
import pytest
from numpy.random import Generator

SEED = 42


@pytest.fixture()
def fx_rng() -> Generator:
    return np.random.default_rng(SEED)

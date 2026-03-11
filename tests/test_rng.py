from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import numpy as np
import pytest

from graphix.rng import ensure_rng


def test_identity() -> None:
    rng = np.random.default_rng()
    assert ensure_rng(rng) is rng


def test_default() -> None:
    with pytest.warns(UserWarning, match="Default random-number generator is used"):
        ensure_rng()


def test_new_thread() -> None:
    t = Thread(target=ensure_rng)
    with pytest.warns(UserWarning, match="Default random-number generator is used"):
        t.start()


@pytest.mark.filterwarnings("ignore:Default random-number generator is used")
def test_threadpool() -> None:
    with ThreadPoolExecutor() as executor:
        tasks = executor.map(lambda _: ensure_rng(), range(100))
        for _ in tasks:
            pass

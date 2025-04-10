from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import numpy as np

from graphix.rng import ensure_rng


def test_identity() -> None:
    rng = np.random.default_rng()
    assert ensure_rng(rng) is rng


def test_new_thread() -> None:
    t = Thread(target=ensure_rng)
    t.start()


def test_threadpool() -> None:
    with ThreadPoolExecutor() as executor:
        tasks = executor.map(lambda _: ensure_rng(), range(100))
        for _ in tasks:
            pass
